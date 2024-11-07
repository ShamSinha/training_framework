"""Default monai models does not support mask."""
from typing import Optional

import monai.networks.nets as monainets
import torch
import torch.nn as nn
from loguru import logger

from ..utils.layers import SELayer


class MonaiModelWithAttn(nn.Module):
    def __init__(self, model_arch: str, in_channels: int, use_mask=False, **model_args) -> None:
        super().__init__()
        self.model_arch = model_arch
        self.model_args = model_args["model_args"]
        self.use_mask = use_mask
        self.in_channels = in_channels
        self._create_model()

    def _create_model(self):
        if self.use_mask:  # mask attn
            self.stem1 = nn.Sequential(
                nn.Conv3d(
                    self.in_channels, self.in_channels * 32, kernel_size=3, stride=1, padding=0
                ),
                nn.BatchNorm3d(self.in_channels * 32),
                nn.ReLU(inplace=True),
            )
            self.stem2 = nn.Sequential(
                nn.Conv3d(
                    self.in_channels, self.in_channels * 32, kernel_size=3, stride=1, padding=0
                ),
                nn.BatchNorm3d(self.in_channels * 32),
                nn.ReLU(inplace=True),
            )
            self.act = nn.ReLU()
            self.se_l = SELayer(self.in_channels * 32, 10)  # 24
            self.dropout = nn.Dropout3d(p=0.05)
            logger.info("Using mask head for Monai classifier.")
            self.net = getattr(monainets, self.model_arch)(
                in_channels=self.in_channels * 32, **self.model_args
            )
        else:
            self.net = getattr(monainets, self.model_arch)(
                in_channels=self.in_channels, **self.model_args
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):  # noqa
        new_x = self.dropout(self.stem1(x) + self.stem2(mask)) if self.use_mask else x
        new_x = self.se_l(new_x) if self.use_mask else x

        return self.net(new_x)
