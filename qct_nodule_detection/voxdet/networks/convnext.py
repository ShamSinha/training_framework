# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/networks/04_convnext.ipynb.

# %% auto 0
__all__ = ['LayerNorm3d', 'drop_path', 'DropPath', 'ConvNextBlock', 'ConvNextStage', 'ConvNext', 'convnext10', 'convnext18',
           'convnext50', 'convnext_fpn3d_feature_extractor']

# %% ../../nbs/networks/04_convnext.ipynb 1
import torch 
import torch.nn as nn
import torch.nn.functional as F
import fastcore.all as fc

from .fpn import BackbonewithFPN3D
from .res_se_net import conv3d

# %% ../../nbs/networks/04_convnext.ipynb 5
class LayerNorm3d(nn.LayerNorm):
    """ LayerNorm for channels of '3D' spatial NCDHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1) #NCDHW -> NDHWC
        #(0, 2, 3, 1) -> NCHW -> NHWC 
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3) # NDHWC -> NCDHW
        return x

# %% ../../nbs/networks/04_convnext.ipynb 6
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    copied from https://github.com/rwightman/pytorch-image-models/blob/7d9e321b761a673000af312ad21ef1dec491b1e9/timm/layers/drop.py#L137
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

# %% ../../nbs/networks/04_convnext.ipynb 7
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        fc.store_attr()
    __repr__ = fc.basic_repr("drop_prob, scale_by_keep")
    def forward(self, x):return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# %% ../../nbs/networks/04_convnext.ipynb 17
class ConvNextBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm3d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        inputs = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = inputs + self.drop_path(x)
        return x

# %% ../../nbs/networks/04_convnext.ipynb 21
class ConvNextStage(nn.Module):
    def __init__(self, dims, layers, dp_rates=0, normalize=None): 
        fc.store_attr()
        super().__init__()
        if not isinstance(dp_rates, list): dp_rates = [x.item() for x in torch.linspace(0, dp_rates, layers)]  
        for i in range(self.layers): 
            setattr(self, f"layer{i}", ConvNextBlock(dims, drop_path=dp_rates[i]))
            if self.normalize is not None: setattr(self, f"norm{i}", self.normalize(dims))
            
    def forward(self, x): 
        for i in range(self.layers): x = getattr(self, f"layer{i}")(x)
        if self.normalize is not None: x = getattr(self, f"norm{i}")(x)
        return x 

# %% ../../nbs/networks/04_convnext.ipynb 26
class ConvNext(nn.Module):
    def __init__(self, ic=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], c1_ks=(4, 4, 4), c1_s=(2, 4, 4),
                 drop_path_rate=0.):
        fc.store_attr()
        super().__init__()
        dp_rates = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(depths))]
        pad = 0 if c1_s[0] == c1_s[1] else 1 
        for i in range(len(self.depths)):  
            if i == 0:  
                setattr(self, f"base", conv3d(ic, dims[0], ks=c1_ks, stride=c1_s, norm=LayerNorm3d, padding=(pad, 0, 0)))
            else:
                setattr(self, f"downsample{i}", nn.Conv3d(dims[i-1], dims[i], kernel_size=2, stride=2, padding=(0, 0, 0)))
            
            dp_rates_layer = dp_rates[sum(depths[:i]): sum(depths[:i+1])]
            setattr(self, f"stage{i+1}", ConvNextStage(dims[i], layers=depths[i], dp_rates=dp_rates_layer))            
        
    
    def forward(self, x): 
        out = x 
        for i in range(len(self.dims)):
            if i==0: out = self.base(out)
            else: out = getattr(self, f"downsample{i}")(out)
            out = getattr(self, f"stage{i+1}")(out)
        return out

# %% ../../nbs/networks/04_convnext.ipynb 36
def convnext10(ic, dims=[96, 192, 384, 768], c1_ks=(4, 4, 4), c1_s=(2, 4, 4), drop_path_rate=0.):
    c10 = ConvNext(ic=ic, depths=(1, 1, 1, 1), dims=dims, c1_ks=c1_ks, c1_s=c1_s, drop_path_rate=drop_path_rate)
    return c10

# %% ../../nbs/networks/04_convnext.ipynb 37
def convnext18(ic, dims=[96, 192, 384, 768], c1_ks=(4, 4, 4), c1_s=(2, 4, 4), drop_path_rate=0.):
    c18 = ConvNext(ic=ic, depths=(2, 2, 2, 2), dims=dims, c1_ks=c1_ks, c1_s=c1_s, drop_path_rate=drop_path_rate)
    return c18

# %% ../../nbs/networks/04_convnext.ipynb 38
def convnext50(ic, dims=[96, 192, 384, 768], c1_ks=(4, 4, 4), c1_s=(2, 4, 4), drop_path_rate=0.):
    c50 = ConvNext(ic=ic, depths=(3, 3, 9, 3), dims=dims, c1_ks=c1_ks, c1_s=c1_s, drop_path_rate=drop_path_rate)
    return c50

# %% ../../nbs/networks/04_convnext.ipynb 41
def convnext_fpn3d_feature_extractor(backbone, out_channels=256, returned_layers=[1, 2, 3], extra_blocks:bool=False):
    in_channels_stage2 = backbone.dims[-1] // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    return_layers = {f"stage{k}": str(v) for v, k in enumerate(returned_layers)}
    return BackbonewithFPN3D(backbone, return_layers, in_channels_list, out_channels, extra_blocks)
