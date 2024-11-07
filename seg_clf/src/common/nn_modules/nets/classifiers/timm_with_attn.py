from typing import Optional

import timm
import torch
import torch.nn as nn
from loguru import logger

from ..utils.convert_3d import convert_3d
from ..utils.layers import SELayer


class Timm3DWithAttn(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_chans: int,
        num_classes: int,
        act_layer: nn.Module = nn.ReLU,
        drop: float = 0.0,
        bias: bool = False,
        use_mask: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.act_layer = act_layer
        self.drop = drop
        self.bias = bias
        self.use_mask = use_mask
        self._create_model()

    def _create_model(self):
        if self.use_mask:  # mask attn
            self.stem1 = nn.Sequential(
                nn.Conv3d(
                    self.in_chans, self.in_chans * 32, kernel_size=3, stride=1, padding=0
                ),  # stride 1 pad 0
                nn.BatchNorm3d(self.in_chans * 32),
                nn.ReLU(inplace=True),
            )
            self.stem2 = nn.Sequential(
                nn.Conv3d(self.in_chans, self.in_chans * 32, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm3d(self.in_chans * 32),
                nn.ReLU(inplace=True),
            )
            self.se_l = SELayer(self.in_chans * 32, 10)  # 32
            self.dropout = nn.Dropout3d(p=0.05)
            logger.info("Using mask head for Timm classifier.")
        backbone = timm.create_model(
            self.backbone,
            pretrained=False,
            num_classes=0,
            in_chans=self.in_chans * 32 if self.use_mask else self.in_chans,
            drop_rate=self.drop,
        )
        self.backbone = convert_3d(backbone)
        self.backbone.global_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten(1)
        self.mlp = nn.Linear(backbone.num_features, self.num_classes, bias=True)

    def forward(self, x: torch.Tensor, roi_mask: Optional[torch.Tensor] = None):  # noqa
        new_x = self.dropout(self.stem1(x) + self.stem2(roi_mask)) if self.use_mask else x
        new_x = self.se_l(new_x) if self.use_mask else x
        return self.mlp(self.flatten(self.backbone(new_x)))
