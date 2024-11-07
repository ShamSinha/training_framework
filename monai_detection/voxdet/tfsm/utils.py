# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/tfsm/02_transform_utils.ipynb.

# %% auto 0
__all__ = ['chwd_2_corner', 'corner_2_chwd']

# %% ../../nbs/tfsm/02_transform_utils.ipynb 2
import torch
import numpy as np 
from typing import Union

# %% ../../nbs/tfsm/02_transform_utils.ipynb 4
def chwd_2_corner(bbox: Union[torch.Tensor, np.asarray]):
    #convert xc, yc, zc, h, w, d to x1 y1 z1 x2 y2 z2
    # Note that it 
    out = bbox.copy().astype(float) if isinstance(bbox, np.ndarray) else bbox.clone()
    out[:, :3] = bbox[:, :3] - (bbox[:, 3:]/2)
    out[:, 3:] = bbox[:, :3] + (bbox[:, 3:]/2)
    return out 

# %% ../../nbs/tfsm/02_transform_utils.ipynb 6
def corner_2_chwd(bbox: Union[torch.Tensor, np.asarray]):
    out = bbox.copy().astype(float) if isinstance(bbox, np.ndarray) else bbox.clone()
    hwd = (bbox[:, 3:] - bbox[:, :3])
    out[:, 3:] = hwd
    out[:, :3] = bbox[:, :3]+(hwd/2)
    return out