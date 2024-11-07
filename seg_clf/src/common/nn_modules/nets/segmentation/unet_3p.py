# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Norm, Pool
from monai.utils import InterpolateMode, deprecated_arg


class TwoConv(nn.Sequential):
    """two convolutions."""

    @deprecated_arg(
        name="dim",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(
            spatial_dims,
            in_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=1,
        )
        conv_1 = Convolution(
            spatial_dims,
            out_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=1,
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    @deprecated_arg(
        name="dim",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class SkipConcatUpDown(nn.Module):
    def __init__(
        self,
        spatial_dims,
        downsample_pool_strides,
        down_channels,
        upsample_scales,
        # up_channels,
        last_encoder_channels: int,
        skip_in_channels,
        skip_out_channels: int = 64,
        kernel_size: int = 3,
        conv_only: bool = True,
        dropout: float = 0.0,
        interp_mode: Union[InterpolateMode, str] = InterpolateMode.TRILINEAR,
    ) -> None:
        super().__init__()
        self.downsample_pool_strides = downsample_pool_strides
        self.upsample_scales = upsample_scales
        self.spatial_dims = spatial_dims
        self.skip_out_channels = skip_out_channels
        self.down_channels = down_channels
        self.last_encoder_channels = last_encoder_channels
        self.skip_in_channel = skip_in_channels

        self.skip_down_convs = nn.ModuleList(
            [
                nn.Sequential(
                    Pool["MAX", spatial_dims](kernel_size=k_s),
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=in_c,
                        out_channels=self.skip_out_channels,
                        kernel_size=kernel_size,
                        strides=1,
                        padding=None,
                        adn_ordering="NDA",
                        act="relu",
                        norm=Norm.BATCH,
                        dropout=dropout,
                        conv_only=conv_only,
                    ),
                )
                for k_s, in_c in zip(self.downsample_pool_strides, self.down_channels)
            ]
        )

        linear_mode = ["linear", "bilinear", "trilinear"]
        interp_mode = linear_mode[spatial_dims - 1]

        total_in = (
            1 + len(upsample_scales) + len(downsample_pool_strides)
        ) * self.skip_out_channels

        self.up_channels = [total_in] * (len(self.upsample_scales))
        self.up_channels[0] = self.last_encoder_channels
        self.skip_up_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=s_f, mode=interp_mode),
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=in_c,
                        out_channels=self.skip_out_channels,
                        kernel_size=kernel_size,
                        strides=1,
                        padding=None,
                        adn_ordering="NDA",
                        act="relu",
                        norm=Norm.BATCH,
                        dropout=dropout,
                        conv_only=conv_only,
                    ),
                )
                for s_f, in_c in zip(upsample_scales, self.up_channels)
            ]
        )

        self.skip_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=skip_in_channels,
            out_channels=self.skip_out_channels,
            kernel_size=kernel_size,
            strides=1,
            padding=None,
            # adn_ordering="NDA",
            # act="relu",
            # norm=Norm.BATCH,
            # dropout=dropout,
            conv_only=conv_only,
        )

        self.merge_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=total_in,
            out_channels=total_in,
            kernel_size=kernel_size,
            strides=1,
            padding=None,
            adn_ordering="NDA",
            act="relu",
            norm=Norm.BATCH,
            dropout=dropout,
            conv_only=False,
        )

    def forward(
        self, x_in, e_fts, d_fts
    ):  # Same level encoder output, List of encoder fts above this, List of decoder fts below this.
        assert len(e_fts) == len(self.downsample_pool_strides)
        assert len(d_fts) == len(self.upsample_scales)
        early_fts = [self.skip_down_convs[i](e_ft) for i, e_ft in enumerate(e_fts)]
        late_fts = [self.skip_up_convs[i](d_ft) for i, d_ft in enumerate(d_fts)]
        skip_ft = [self.skip_conv(x_in)]
        all_fts = torch.cat(early_fts + late_fts + skip_ft, dim=1)
        final_ft = self.merge_conv(all_fts)
        return final_ft


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions."""

    @deprecated_arg(
        name="dim",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x


class Unet3p(nn.Module):
    @deprecated_arg(
        name="dimensions",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (64, 128, 256, 512, 1024),
        skip_out_channels: int = 64,
        act: Union[str, tuple] = (
            "LeakyReLU",
            {"negative_slope": 0.1, "inplace": True},
        ),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        dimensions: Optional[int] = None,
    ):

        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        # fea = ensure_tuple_rep(features, 6)
        fea = features
        print(f"BasicUNet features: {fea}.")

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        # 64, 128, 256, 512, 1024
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        # down_layers = nn.ModuleList([Down(spatial_dims, fea[i],
        # fea[i+1], act, norm, bias, dropout) for i in range(len(fea)-1)])

        # Current level -> l # 1 indexing, total -> n
        self.up1 = SkipConcatUpDown(
            spatial_dims=spatial_dims,
            downsample_pool_strides=[8, 4, 2],  # [2**(i) for i in range(n-2, 0, -1)]
            down_channels=fea[:3],  # fea[]
            upsample_scales=[2],
            last_encoder_channels=fea[4],  # fea[-1]
            skip_in_channels=fea[3],  # fea[l]
            skip_out_channels=skip_out_channels,
        )

        self.up2 = SkipConcatUpDown(
            spatial_dims=spatial_dims,
            downsample_pool_strides=[4, 2],
            down_channels=fea[:2],
            upsample_scales=[4, 2],
            last_encoder_channels=fea[4],
            skip_in_channels=fea[2],
            skip_out_channels=skip_out_channels,
        )

        self.up3 = SkipConcatUpDown(
            spatial_dims=spatial_dims,
            downsample_pool_strides=[2],
            down_channels=fea[:1],
            upsample_scales=[8, 4, 2],
            last_encoder_channels=fea[4],
            skip_in_channels=fea[1],
            skip_out_channels=skip_out_channels,
        )

        self.up4 = SkipConcatUpDown(
            spatial_dims=spatial_dims,
            downsample_pool_strides=[],
            down_channels=[],
            upsample_scales=[16, 8, 4, 2],
            last_encoder_channels=fea[4],
            skip_in_channels=fea[0],
            skip_out_channels=skip_out_channels,
        )

        in_channels = len(features) * skip_out_channels
        self.final_conv = Conv["conv", spatial_dims](in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        # Same level encoder output, List of encoder fts above this, List of decoder fts below this.
        u3 = self.up1(x3, [x0, x1, x2], [x4])
        u2 = self.up2(x2, [x0, x1], [x4, u3])
        u1 = self.up3(x1, [x0], [x4, u3, u2])
        u0 = self.up4(x0, [], [x4, u3, u2, u1])
        seg_logits = self.final_conv(u0)
        return seg_logits


if __name__ == "__main__":
    model = Unet3p(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
    )
    # model.eval()
    model.cuda()
    batch_size = 2
    x = torch.rand((batch_size, 1, 32, 64, 64)).cuda()
    # with torch.no_grad():
    seg_logits = model(
        x
    )  # B X out_channels X D X H X W , [B X class_head_num_cls[i] for i in range(len(class_head_num_cls))]
    print(seg_logits.shape)
