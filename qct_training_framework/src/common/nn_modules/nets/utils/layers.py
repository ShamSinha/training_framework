from typing import List

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock3D, LinearScheduler
from timm.models.layers.fast_norm import fast_layer_norm, is_fast_norm


class SELayer(nn.Module):
    def __init__(
        self,
        channel: int,
        reduction: int = 16,
        act_layer=nn.ReLU,
        dims: int = 3,
    ):
        """Squeeze-and-Excitation (SE) block.

        Args:
            channel (int): number of channels
            reduction (int): reduction ratio
            act_layer (nn.Module): activation layer
            dims (int): 2 or 3 for 2D or 3D inputs
        """
        super().__init__()
        assert dims in [2, 3], "dims must be 2 (2D inputs) or 3 (3D inputs)"
        self.dims = dims
        self.avg_pool = nn.AdaptiveAvgPool3d(1) if dims == 3 else nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            act_layer(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        if self.dims == 3:
            b, c, _, _, _ = x.size()
        else:
            b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1) if self.dims == 3 else self.fc(y).view(b, c, 1, 1)
        return x * y


def get_drop_out(
    drop_type: str, drop_prob: float = 0, drop_block_size: int = 3, total_steps: int = 10e3
):
    if drop_type == "dropout":
        return nn.Dropout(p=drop_prob)
    elif drop_type == "dropblock":
        return LinearScheduler(
            DropBlock3D(drop_prob=drop_prob, block_size=drop_block_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=int(total_steps),
        )


class PostRes(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        stride: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        act_layer=nn.ReLU,
        drop_prob: float = 0.1,
        drop_block_size: int = 3,
        total_steps: int = 10e3,
        drop_type: str = "dropout",
        norm: str = "batch",
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm3d(n_out) if norm == "batch" else create_group_norm(n_out)
        self.relu = act_layer()
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(n_out) if norm == "batch" else create_group_norm(n_out)
        self.se = SELayer(n_out, act_layer=act_layer)
        self.drop_block = get_drop_out(
            drop_type=drop_type,
            drop_prob=drop_prob,
            drop_block_size=drop_block_size,
            total_steps=total_steps,
        )

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out) if norm == "batch" else create_group_norm(n_out),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.drop_block(self.relu(out))
        return out


class LinearDropOut3d(nn.Dropout3d):
    def __init__(self, p=0.2, inplace=True, total_steps=15e3):
        super().__init__(p, inplace)
        self.total_steps = total_steps
        self.probs = np.linspace(0, p, total_steps)
        self.index = -1
        self.p = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.index += 1
        return F.dropout3d(
            input,
            self.probs[self.index] if self.index < len(self.probs) else self.p,
            self.training,
            self.inplace,
        )


class GroupViews(nn.Module):
    def __init__(
        self,
        in_chs: int,
        num_inputs: int,
        act_layer=nn.ReLU,
        drop_prob: float = 0.0,
        reduction: int = 16,
        drop_block_size: int = 3,
        total_steps: int = 5e3,
        norm: str = "batch",
        drop_type: str = "dropout",
    ):
        super().__init__()
        self.in_chs = in_chs
        # se layer to give priority to one of the views depending on the data
        # eg. for ggo min MIP might have higher weights and so on.
        # self.dropout = nn.Dropout3d(p=0.1, inplace=True)
        # self.dropout = LinearDropOut3d(p=0.2, inplace=True, total_steps=15000)
        self.dropout = get_drop_out(
            drop_type=drop_type,
            drop_prob=drop_prob,
            drop_block_size=drop_block_size,
            total_steps=total_steps,
        )
        # LinearScheduler(
        #         DropBlock3D(drop_prob=drop_prob, block_size=drop_block_size),
        #         start_value=0.,
        #         stop_value=drop_prob,
        #         nr_steps=int(total_steps)
        #     )
        self.se = SELayer(in_chs * num_inputs, reduction=reduction, act_layer=act_layer)
        # pooling at different scales then upscale and concat to get a context
        self.pooling1 = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.pooling1 = SoftPool3d(kernel_size = 2, stride=2)
        self.conv11_1 = nn.Sequential(
            nn.Conv3d(in_chs * num_inputs, in_chs, 1, 1),
            nn.BatchNorm3d(in_chs) if norm == "batch" else create_group_norm(in_chs),
        )
        self.pooling2 = nn.MaxPool3d(kernel_size=4, stride=4)
        # self.pooling2 = SoftPool3d(kernel_size = 4, stride=4)
        self.conv11_2 = nn.Sequential(
            nn.Conv3d(in_chs * num_inputs, in_chs, 1, 1),
            nn.BatchNorm3d(in_chs) if norm == "batch" else create_group_norm(in_chs),
        )
        self.relu = act_layer()
        # self.pooling3 = nn.MaxPool3d(kernel_size=8, stride=8) # might need for scan
        # self.upsample1 = nn.Upsample(scale_factor=2)
        # using deconvolution to upsample instead of straight forward upsampling
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(in_chs, in_chs, kernel_size=2, stride=2),
            nn.BatchNorm3d(in_chs) if norm == "batch" else create_group_norm(in_chs),
            act_layer(),
        )
        # self.se2 = SELayer(in_chs * 2, reduction=reduction, act_layer=act_layer)
        # self.upsample2 = nn.Upsample(scale_factor=4)

    def forward(self, x: List[torch.Tensor]):
        concated_ip = self.dropout(self.relu(torch.cat(x, axis=1)))  # concat in channel dimension
        se_op = self.se(concated_ip)  # TODO check effect of adding relu here
        pool_op1 = self.relu(self.conv11_1(self.relu(self.pooling1(se_op))))
        pool_op2 = self.relu(self.conv11_2(self.relu(self.pooling2(se_op))))
        upsample_pool_op2 = self.upsample1(pool_op2)

        concat_op = torch.concat([pool_op1, upsample_pool_op2], axis=1)  # self.se2

        return concat_op


def create_group_norm(in_channels, max_chs_in_grp: int = 16):
    return nn.GroupNorm(int(in_channels / max_chs_in_grp), num_channels=in_channels)


class Encoder(nn.Module):
    def __init__(
        self,
        in_chs: int = 1,
        act_layer=nn.ReLU,
        drop_prob: float = 0.0,
        drop_block_size: int = 3,
        total_steps: int = int(10e3),
        norm: str = "batch",
        drop_type: str = "dropout",
    ):
        super().__init__()
        self.in_chs = in_chs

        self.preBlock = nn.Sequential(
            nn.Conv3d(in_chs, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32) if norm == "batch" else create_group_norm(32),
            act_layer(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32) if norm == "batch" else create_group_norm(32),
            act_layer(),
        )
        self.resblock0 = PostRes(
            32,
            32,
            act_layer=act_layer,
            drop_prob=drop_prob,
            drop_block_size=drop_block_size,
            total_steps=total_steps,
            norm=norm,
            drop_type=drop_type,
        )
        # using conv with stride 2 to downsample instead of pooling layers
        self.downsample1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(32) if norm == "batch" else create_group_norm(32),
            act_layer(),  # same relu for all downsamples
        )
        self.resblock1 = PostRes(
            32,
            32,
            act_layer=act_layer,
            drop_prob=drop_prob,
            drop_block_size=drop_block_size,
            total_steps=total_steps,
            norm=norm,
            drop_type=drop_type,
        )

    def forward(self, x):
        return self.resblock1(self.downsample1(self.resblock0(self.preBlock(x))))


class MVTail(nn.Module):
    def __init__(
        self,
        in_chs: int,
        act_layer: nn.Module,
        drop_rate: float,
        out_channels: int,
        drop_block_size: int,
        total_steps: int = 10e3,
        norm: str = "batch",
        drop_type: str = "dropout",
    ) -> None:
        super().__init__()
        self.act_layer = act_layer
        self.drop_rate = drop_rate
        self.out_channels = out_channels
        self.postres1 = PostRes(
            in_chs,
            in_chs,
            act_layer=self.act_layer,
            drop_type=drop_type,
            drop_prob=drop_rate,
            drop_block_size=drop_block_size,
            total_steps=total_steps,
        )
        self.downsample1 = nn.Sequential(
            nn.Conv3d(in_chs, in_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(in_chs) if norm == "batch" else create_group_norm(in_chs),
        )
        self.postres2 = PostRes(
            in_chs,
            in_chs,
            act_layer=self.act_layer,
            drop_type=drop_type,
            drop_prob=drop_rate,
            drop_block_size=drop_block_size,
            total_steps=total_steps,
        )
        self.downsample2 = nn.Sequential(
            nn.Conv3d(in_chs, in_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(in_chs) if norm == "batch" else create_group_norm(in_chs),
        )
        self.postres3 = PostRes(
            in_chs,
            in_chs,
            act_layer=self.act_layer,
            drop_type=drop_type,
            drop_prob=drop_rate,
            drop_block_size=drop_block_size,
            total_steps=total_steps,
        )
        # self.downsample3 = nn.Conv3d(24, 12, kernel_size=2, stride=1)
        # self.dropout2 = nn.Dropout3d(p=self.drop_rate, inplace=True)
        self.dropout2 = get_drop_out(
            drop_type=drop_type,
            drop_prob=drop_rate,
            drop_block_size=drop_block_size,
            total_steps=total_steps,
        )
        # LinearScheduler(
        #         DropBlock3D(drop_prob=drop_rate, block_size=drop_block_size),
        #         start_value=0.,
        #         stop_value=drop_rate,
        #         nr_steps=int(total_steps)
        #     )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, self.out_channels)  # 1024 + (3*64 + 128)
        self.gelu1 = nn.GELU()  # same GELU for all downsamples
        self.gelu2 = nn.GELU()  # same GELU for all downsamples

    def forward(self, x):
        tail_op0 = self.postres1(x)
        tail_op1 = self.gelu1(self.downsample1(tail_op0))
        tail_op2 = self.postres2(self.dropout2(tail_op1))
        tail_op3 = self.postres3(self.gelu2(self.downsample2(tail_op2)))

        return self.fc(self.flatten(tail_op3))


class LayerNorm3d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial NCHW tensors."""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = (
            is_fast_norm()
        )  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x


def get_ConvNeXtBlock3D(convnext_block):
    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 4, 1, 2, 3)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1, 1))

        x = self.drop_path(x) + shortcut
        return x

    convnext_block.forward = forward.__get__(convnext_block, timm.models.convnext.ConvNeXtBlock)
    return convnext_block
