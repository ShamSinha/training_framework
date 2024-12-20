# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/networks/05_convnextv2.ipynb.

# %% auto 0
__all__ = ['ConvNextV2BackbonewithFPN3D']

# %% ../../nbs/networks/05_convnextv2.ipynb 2
import torch
import torch.nn as nn
import fastcore.all as fc

from collections import OrderedDict
from medct.convnextv2 import ConvNextV2Model3d, ConvNextV2Config3d
from monai.networks.blocks.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

# %% ../../nbs/networks/05_convnextv2.ipynb 14
class ConvNextV2BackbonewithFPN3D(nn.Module):
    def __init__(self, backbone_cfg, returned_layers=[1, 2], out_channels=256, extra_blocks=False):
        super().__init__()
        fc.store_attr(names=["backbone_cfg", "returned_layers", "out_channels", "extra_blocks"])
        from omegaconf import DictConfig, OmegaConf #during inference self.backbone_cfg is DictConfig which is not supported by transformers.
        if isinstance(self.backbone_cfg, DictConfig):
            self.backbone_cfg = OmegaConf.to_object(self.backbone_cfg)
        self.cfg = ConvNextV2Config3d(**self.backbone_cfg)
        self.body = ConvNextV2Model3d(self.cfg)
        self.fpn = FeaturePyramidNetwork( 
            spatial_dims=3, 
            in_channels_list=self.cfg.hidden_sizes,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(3) if extra_blocks else None,
        )
        
    def forward(self, x):
        out = self.body(x, output_hidden_states=True).hidden_states
        out = OrderedDict({f"layer{k}": v for k, v in enumerate(out) if k in self.returned_layers})
        y = self.fpn(out)
        return y
