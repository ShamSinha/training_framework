from typing import Any, Dict, List, Optional, Literal

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from loguru import logger
from monai.networks import one_hot

class Dice3D(Metric):
    dice_scores: List[Tensor]
    full_state_update: bool = True

    def __init__(
        self,
        num_classes: int,
        conf_thresholds: List[float] = [0.5],
        compute_on_step: Optional[bool] = None,
        ignore_background: bool = False,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        **kwargs: Dict[str, Any]
    ):
        """_summary_

        Args:
            conf_thresholds (Optional[List[float]], optional): _description_. Defaults to None.
            compute_on_step (Optional[bool], optional): _description_. Defaults to None.
            ignore_background (bool): ignore dice computation for background pixels

        Example:
        ```
        import torch
        pred = [torch.ones(2, 3, 3, 3)]
        gt = [torch.cat([torch.zeros(1, 3, 3, 3), torch.ones(1, 3, 3, 3)], dim=0)]
        dice_mat = Dice3D(ignore_background=False)
        dice_mat.update(preds=pred, target=gt)
        print(dice_mat.compute())
        ```
        """
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        self.num_classes = num_classes
        self.conf_thresholds = conf_thresholds
        self.ignore_background = ignore_background
        self.average = average
        self.add_state("dice_scores", default=[], dist_reduce_fx=None)

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:  # type: ignore
        """Add detections and ground truth to the metric.
        Args:
            preds: A list consisting of Tensors. Each Tensor are prediction corresponding to one image.
            - Each probability Tensor should be in following format [num_class, z, y, x]
            - Each probability Tensors value range should be in [0, 1]
            target: A list consisting of tensors, Each Tensor are prediction corresponding to one image.
            - Each Tensor should be in following format [num_class, z, y, x]
            - values should be either 1 or 0
        """
        # _input_validator(preds, target, self.num_classes)

        if target[-1].shape[0] == 1 :
            target = one_hot(target,self.num_classes)
            
        assert target.shape == preds.shape

        self.input_device = preds.device
        self.dice_scores.append(self._get_intermediate_dice(preds, target))

    def _get_intermediate_dice(
        self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]
    ):
        nb_classes = len(self._get_classes())
        nb_confs = len(self.conf_thresholds)
        nb_imgs = len(preds)
        dice_score = -1 * torch.ones(nb_imgs, nb_classes, nb_confs).to(self.input_device)
        iou_score = -1 * torch.ones(nb_imgs, nb_classes, nb_confs).to(self.input_device)

        # logger.debug(f"nb_classes: {nb_classes}, nb_confs: {nb_confs}, nb_imgs: {nb_imgs}")
        # logger.debug(f"dice_score: {dice_score.shape}, iou_score: {iou_score.shape}")
        
        for idx_cls in range(nb_classes):
            if self.ignore_background and idx_cls == 0:
                continue
            for idx_conf in range(nb_confs):
                dice_score, iou_score = self._calculate_dice_score(
                    dice_score=dice_score,
                    iou_score=iou_score,
                    class_idx=idx_cls,
                    conf_idx=idx_conf,
                    pred_probs=preds,
                    gt_labels=target,
                )
        return dice_score

    def _get_classes(self):
        if self.num_classes is not None:
            return np.arange(self.num_classes)
        return []

    @staticmethod
    def __compute_dice(pred_mask, gt_mask):
        """
        Compute dice score given two binary mask
        dice score = 2*area(inter(pred_mask, gt_mask)) / area(gt_mask) + area(pred_mask)
        """
        return (2 * torch.sum(pred_mask * gt_mask)) / (torch.sum(pred_mask) + torch.sum(gt_mask))

    @staticmethod
    def __compute_iou(pred_mask, gt_mask):
        """Compute IoU of the predicted binary mask and ground truth mask."""

        return torch.sum(pred_mask * gt_mask) / torch.sum(torch.logical_or(pred_mask, gt_mask))

    def _calculate_dice_score(
        self,
        dice_score,
        iou_score,
        class_idx,
        conf_idx,
        pred_probs: List[Tensor],
        gt_labels: List[Tensor],
    ):
        for index, (detection_prob, gt_mask) in enumerate(zip(pred_probs, gt_labels)):
            det_binary = detection_prob[class_idx] > self.conf_thresholds[conf_idx]
            gt_binary = gt_mask[class_idx]
            dice_score[index, class_idx, conf_idx] = self.__compute_dice(det_binary, gt_binary)
            iou_score[index, class_idx, conf_idx] = self.__compute_iou(det_binary, gt_binary)

        return dice_score, iou_score

    def compute(self):
        if self.dice_scores == []:
            return None
        dice_scores = torch.cat(self.dice_scores, dim=0)
        if self.average == "micro":
            return dice_scores[(dice_scores >= 0) & (~dice_scores.isnan())].mean()
        elif self.average == "macro":
            return (torch.mean(dice_scores , 0)).mean()
        elif self.average in ['none' , None] :
            return torch.mean(dice_scores , 0)[:,0]
       
# if __name__ == "__main__":
#     import torch

#     pred = [torch.ones(2, 3, 3, 3)]
#     gt = [torch.cat([torch.zeros(1, 3, 3, 3), torch.ones(1, 3, 3, 3)], dim=0)]
#     dice_mat = Dice3D(ignore_background=False)
#     dice_mat.update(preds=pred, target=gt)
#     print(dice_mat.compute())