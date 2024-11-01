# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/bbox_func/02_reg_loss.ipynb.

# %% auto 0
__all__ = ['COMPUTE_DTYPE', 'StrEnum', 'LossReduction', 'RegLoss']

# %% ../../nbs/bbox_func/02_reg_loss.ipynb 2
import torch
import torch.nn.functional as F
import fastcore.all as fc
from enum import Enum
from torch.nn.modules.loss import _Loss
from typing import Union
from . import bbox_iou #import cal_diou_pair, cal_giou_pair

# %% ../../nbs/bbox_func/02_reg_loss.ipynb 4
class StrEnum(str, Enum):
    """Enum subclass that converts its value to a string."""
    def __str__(self): return self.value
    def __repr__(self): return self.value

# %% ../../nbs/bbox_func/02_reg_loss.ipynb 5
class LossReduction(StrEnum):
    NONE = "none"
    MEAN = "mean"
    SUM = "sum"

# %% ../../nbs/bbox_func/02_reg_loss.ipynb 7
COMPUTE_DTYPE = torch.float32

# %% ../../nbs/bbox_func/02_reg_loss.ipynb 9
class RegLoss(_Loss):
    def __init__(self, \
                 reduction: Union[LossReduction, str] = LossReduction.MEAN, \
                 iou_loss: str="diou", \
                 l1_loss: bool= True) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        fc.store_attr()
        
    
    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape != inputs.shape:raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        box_dtype = inputs.dtype
        iou = getattr(bbox_iou, f'cal_{self.iou_loss}_pair')(target.to(dtype=COMPUTE_DTYPE), \
                                                             inputs.to(dtype=COMPUTE_DTYPE)) # type: ignore
        iou_loss= self._reduce(1.0 - iou).to(box_dtype)
        if self.l1_loss:
            l1_loss= F.l1_loss(target.to(dtype=COMPUTE_DTYPE), inputs.to(dtype=COMPUTE_DTYPE))
            l1_loss = self._reduce(l1_loss).to(box_dtype)
            return iou_loss, l1_loss 
        return iou_loss
    
    def _reduce(self, loss):
        if self.reduction == LossReduction.MEAN.value: loss = loss.mean()
        elif self.reduction == LossReduction.SUM.value: loss = loss.sum()
        elif self.reduction == LossReduction.NONE.value: pass
        else: raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return loss
