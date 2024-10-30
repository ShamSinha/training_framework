from typing import List, Optional

import torch
import torch.nn as nn

from src.common.nn_modules.nets.utils.layers import SELayer

from ..utils.convert_3d import convert_3d


class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = SELayer(out_c)  # Squeeze_Excitation(out_c) #CBAM(out_c, 3, 8)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y


class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = SELayer(out_c)  # Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y


class Qnet(nn.Module):
    def __init__(
        self,
        num_classes: Optional[int] = None,
        encoder_channels: List[int] = [1, 16, 32, 64, 128],
        return_feats: bool = False,
        apply_avg_pool: bool = True,
        convert_3d: bool = True,
    ):
        """Merging all the new things into Unet.

        Attention. ASPP. SELayer. ROI Encoder.
        With the default config model should have 16.42M parameters.
        Args:
            num_classes (int): Number of classes to predict.
            encoder_channels (List[int], optional): Encoder channels. Defaults to [1, 16, 32, 64, 128, 256].
            return_feats (bool): return features. For MTL backbone keep this True.
            apply_avg_pool (bool): apply average pooling. Applicable only for MTL.
            convert_3d (bool, optional): Whether to convert 2d to 3d. Defaults to True.
        Example:
        ```
        model = Qnet(num_classes=2)
        crop_batch = torch.randn(2, 1, 32, 64, 64)
        roi_batch = (torch.randn(2, 1, 32, 64, 64)>0.5).float()
        op = model(crop_batch, roi_batch)
        op.shape
        ```
        """
        super().__init__()
        # either num_classes is not None else return_feats must be True
        assert num_classes is not None or return_feats is True
        self.encoder_channels = encoder_channels
        self.num_classes = num_classes
        self.return_feats = return_feats
        self.apply_avg_pool = apply_avg_pool
        self.convert_3d = convert_3d

        self.build_model()

    def apply_3d(self, x):
        if self.convert_3d:
            x = convert_3d(x)
        return x

    def build_model(self):
        encoder_c = self.encoder_channels
        self.crop_c1 = convert_3d(
            self.apply_3d(Stem_Block(encoder_c[0], encoder_c[1], stride=1))
        )
        self.crop_c2 = convert_3d(
            self.apply_3d(ResNet_Block(encoder_c[1], encoder_c[2], stride=2))
        )
        self.crop_c3 = convert_3d(
            self.apply_3d(ResNet_Block(encoder_c[2], encoder_c[3], stride=2))
        )
        self.crop_c4 = convert_3d(
            self.apply_3d(ResNet_Block(encoder_c[3], encoder_c[4], stride=2))
        )

        self.mask_c1 = convert_3d(
            self.apply_3d(Stem_Block(encoder_c[0], encoder_c[1], stride=1))
        )
        self.mask_c2 = convert_3d(
            self.apply_3d(ResNet_Block(encoder_c[1], encoder_c[2], stride=2))
        )

        self.drop_op = self.apply_3d(nn.Dropout2d(p=0.1))
        self.se_c1 = SELayer(encoder_c[1], 4, dims=3 if self.convert_3d else 2)
        self.se_c2 = SELayer(encoder_c[2], 8, dims=3 if self.convert_3d else 2)

        self.avg_pool = (
            self.apply_3d(nn.AdaptiveAvgPool2d(1))
            if self.apply_avg_pool
            else nn.Identity()
        )
        self.flatten = nn.Flatten(1) if self.apply_avg_pool else nn.Identity()

        if self.return_feats:
            self.mlp = nn.Identity()
        else:
            self.mlp = nn.Linear(encoder_c[4], self.num_classes, bias=True)

        # self.classification_head = convert_3d(
        #     nn.Sequential(
        #         nn.Flatten(),
        #         nn.Linear(encoder_c[4] * (2**3), self.num_classes, bias=False),
        #     )
        # )

    def forward(self, images: torch.Tensor, roi_mask: Optional[torch.Tensor] = None):
        # crop_op1 = self.crop_c1(images)
        # mask_op1 = self.mask_c1(roi_mask)
        # crop_op2 = self.crop_c2(self.se_c1(crop_op1 + mask_op1))
        # mask_op2 = self.mask_c2(mask_op1)
        # crop_op3 = self.crop_c3(self.se_c2(crop_op2 + mask_op2))
        # crop_op4 = self.crop_c4(crop_op3)
        # return self.mlp(self.flatten(self.avg_pool(crop_op4)))
        crop_op1 = self.crop_c1(images)
        crop_op2 = self.crop_c2(crop_op1)
        crop_op3 = self.crop_c3(crop_op2)
        crop_op4 = self.crop_c4(crop_op3)
        return self.mlp(self.flatten(self.avg_pool(crop_op4)))
