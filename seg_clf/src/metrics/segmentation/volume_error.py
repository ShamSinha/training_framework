from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class VolumeError(Metric):
    volume_errors: List[Tensor]
    full_state_update: bool = True

    def __init__(
        self,
        cls_idx: int = 1,
        rediction_method: str = "mean",
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any]
    ):
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        self.rediction_method = rediction_method
        self.cls_idx = cls_idx
        self.add_state("volume_errors", default=[], dist_reduce_fx="sum")

    def update(self, preds: List[Tensor], target: List[Tensor], voxel_vols: List[Tensor]) -> None:
        pred_volume = torch.tensor(
            [
                pred[self.cls_idx].sum().item() * vox_vol.item()
                for pred, vox_vol in zip(preds, voxel_vols)
            ]
        )
        label_volume = torch.tensor(
            [
                pred[self.cls_idx].sum().item() * vox_vol.item()
                for pred, vox_vol in zip(target, voxel_vols)
            ]
        )
        volume_error = torch.abs(pred_volume - label_volume) / label_volume
        self.volume_errors.append(volume_error)

    def compute(self):
        if self.volume_errors == []:
            return None
        return torch.cat(self.volume_errors).mean()
