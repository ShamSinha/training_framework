import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from ..abstract_metric import AbstractMetric
from .auc_plot import plot_auc_curve


class AUC(AbstractMetric):
    pred_probs: List[Tensor]
    gts: List[Tensor]
    full_state_update: bool = True

    def __init__(
        self,
        num_classes: int,
        return_plot: bool = True,
        apply_softmax: bool = True,
        dist_sync_on_step: bool = False,
        **kwargs: Dict[str, Any]
    ):
        """Compute AUC for a given number of classes.

        Args:
            num_classes: number of classes
            return_plot: if to return plot or not
            dist_sync_on_step: whether to synchronize metric state across processes at each ``forward()``
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        self.return_plot = return_plot
        self.num_classes = num_classes
        self.class_ids = list(range(num_classes))
        self.apply_softmax = apply_softmax
        self.add_state("pred_probs", default=[], dist_reduce_fx= None)
        self.add_state("gts", default=[], dist_reduce_fx= None)

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.pred_probs.append(preds if not self.apply_softmax else torch.softmax(preds, dim=1))
        self.gts.append(target)

    def get_auc(
        self,
        labels: Union[torch.Tensor, np.ndarray],
        pred_probs: Union[torch.Tensor, np.ndarray],
        **kwargs
    ):
        if len(np.unique(self.tensor2np(labels))) < self.num_classes:
            return torch.tensor(-1, dtype=torch.float32)
        return torch.tensor(
            roc_auc_score(self.tensor2np(labels), self.tensor2np(pred_probs), **kwargs)
        )

    def compute(self) -> Tensor:
        pred_probs = torch.cat(self.pred_probs, dim=0)
        gts = torch.cat(self.gts, dim=0)
        result = {}
        if (
            self.num_classes > 2
        ):  # individual class AUC only for multi-class else they are same as ovo results
            for class_id in self.class_ids:
                prediction_prob = pred_probs[:, class_id]
                labels = gts == class_id
                result[class_id] = self.get_auc(labels, prediction_prob)
        if self.num_classes > 2:
            result["ovo"] = self.get_auc(
                labels=gts, pred_probs=pred_probs, average="macro", multi_class="ovo"
            )
        else:  # the probability of the class with the “greater label” should be provided
            result["ovo"] = self.get_auc(labels=gts, pred_probs=pred_probs[:, 1])
        # get the AUC plot
        if self.return_plot and len(np.unique(self.tensor2np(gts))) == self.num_classes:
            # if len(gts)>10:
            #     breakpoint()
            # only return plot if all classes are present in the data
            # plot_auc_curve(self.tensor2np(gts), self.tensor2np(pred_probs), n_classes=self.num_classes)
            result["plot"] = plot_auc_curve(
                self.tensor2np(gts), self.tensor2np(pred_probs), n_classes=self.num_classes
            )
        return result
