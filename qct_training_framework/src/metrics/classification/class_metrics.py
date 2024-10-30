from typing import List, Dict, Optional, Any

import numpy as np
import torch
from loguru import logger
from .auc_plot import plot_auc_curve
from .plots import sens_spec_at_diff_th ,plot_auc

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
    average_precision_score
)
from torchmetrics import Metric
from torch import Tensor

class ClassificationMetrics(Metric):
    preds: List[Tensor]
    target: List[Tensor]
    full_state_update: bool = True

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any]
    ):
        """
        returns:
        1. One vs all AUC for each of the classes.
        2. Computes the average AUC of all possible pairwise combinations of classes.
        """
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        self.ignore_index = ignore_index
        self.class_ids = np.arange(num_classes)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.input_device = preds.device
        indices_to_remove = torch.argwhere(target == self.ignore_index)
        remaining_indices = torch.tensor(
            [i for i in range(len(preds)) if i not in indices_to_remove]
        ).to(self.input_device)

        self.target.append(torch.index_select(target, 0, remaining_indices))
        self.preds.append(torch.index_select(preds, 0, remaining_indices))

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        preds = preds.detach().cpu().clone().numpy()
        target = target.detach().cpu().clone().numpy()

        result = {}
        if (
            len(self.class_ids) > 2
        ):  # indivisual class AUC only for multiclass else they are same as ovo results
            for class_id in self.class_ids:
                prediction_prob = preds[:, class_id]
                labels = target == class_id
                result[class_id] = roc_auc_score(labels, prediction_prob)
        if len(self.class_ids) > 2:
            result["ovo"] = roc_auc_score(
                y_true=target, y_score=preds, average="macro", multi_class="ovo"
            )
            precision, recall, fscore, support = precision_recall_fscore_support(
                target, preds, labels=self.class_ids, average=None
            )

            avg_precision = np.average(precision)
            avg_recall = np.average(recall)
            avg_fscore = np.average(fscore)

            result["avg_precision"] = avg_precision
            result["avg_recall"] = avg_recall
            result["avg_fscore"] = avg_fscore
        
        else:  # the probability of the class with the “greater label” should be provided
            prediction_prob = preds[:, 1]
            result["ovo"] = roc_auc_score(y_true=target, y_score=prediction_prob)

            # result["auc_plot"] = plot_auc_curve(
            #     target, preds, n_classes=len(self.class_ids)
            # )

            result["auc_plot"] = plot_auc(target, prediction_prob)
            
            result["sens_spec_at_th_plot"] = sens_spec_at_diff_th(target,prediction_prob)

            conf = confusion_matrix(
                target, prediction_prob >= 0.5, labels=self.class_ids
            )
            tn, fp, fn, tp = conf.flatten()

            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            youden = sensitivity + specificity - 1
            result["sensitivity"] = sensitivity
            result["specificity"] = specificity
            result["youden"] = youden
            result["ap"] = average_precision_score(target, prediction_prob)
        
        return result

