from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from lightning.pytorch import LightningModule, Trainer
from torchmetrics.classification import Accuracy

from ..metrics.classification.auc_plot import plot_auc_curve
from ..metrics.classification.class_metrics import CharacteristicsAUC
from ..metrics.classification.sencitivity_specificity import (
    compute_sencitivity_specificity,
    get_sensitivity_specificity_list,
)
from .clearml_callback import ClearMLCallBack


class ClsSubAnalysis(ClearMLCallBack):
    def __init__(
        self,
        num_classes: int,
        subgroups_keys: List[str],
        outputs_key: str,
        labels_key: str,
        metrics: List[str] = ["AUC", "Accuracy", "Senc_Spec"],
        threshold: float = 0.5,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        self.subgroups_keys = subgroups_keys
        self.outputs_key = outputs_key
        self.labels_key = labels_key
        self.metrics = metrics
        self.num_classes = num_classes
        self.threshold = threshold
        self.top_k = top_k
        self.logger = None

    def _extract_data(self, outputs: List[Dict[str, torch.Tensor]]):
        # extract metadata subgroups
        metadata = {}
        for meta_key in self.subgroups_keys:
            metadata[meta_key] = torch.cat([batch_output[meta_key] for batch_output in outputs])
        # extract outputs & labels
        # keeping as list instead of tensor because batch size can be different across batches
        predictions = [
            prediction for batch_output in outputs for prediction in batch_output[self.outputs_key]
        ]
        labels = [
            label.item() for batch_output in outputs for label in batch_output[self.labels_key]
        ]

        return metadata, predictions, labels

    def _compute_metrics(self, predictions: List[torch.Tensor], labels: List[torch.Tensor]):
        """Add any new metrics here."""
        results = {"Sample Size": len(labels)}
        predictions = torch.stack(predictions).float()
        predictions_arr: torch.Tensor = torch.nn.Softmax(dim=-1)(predictions)
        prediction_classes_arr = (
            predictions_arr.argmax(dim=-1).cpu()
            if predictions_arr.device.type == "cuda"
            else predictions_arr.argmax(dim=-1)
        )
        for metric in self.metrics:
            if metric == "AUC":
                # predictions_arr = torch.stack(predictions)
                auc_metric = CharacteristicsAUC(
                    class_ids=np.arange(self.num_classes).tolist(),
                    gt_classes=labels,
                    pred_probs=predictions_arr.cpu().numpy()
                    if predictions_arr.device.type == "cuda"
                    else predictions_arr.numpy(),
                )
                try:
                    auc = auc_metric.compute()
                except Exception as e:
                    logger.warning(e)
                    auc = -1
                if auc != -1:
                    results.update({f"AUC_{k}": v for k, v in auc.items()})

            elif metric == "Accuracy":
                acc_metric = Accuracy(
                    num_classes=self.num_classes,
                    threshold=self.threshold,
                    task="multiclass",
                    top_k=self.top_k,
                )
                acc_metric(
                    predictions_arr.cpu()
                    if predictions_arr.device.type == "cuda"
                    else predictions_arr,
                    torch.tensor(labels),
                )
                acc = acc_metric.compute()
                results.update({"Accuracy": acc.item()})
            elif metric == "Senc_Spec":
                result = compute_sencitivity_specificity(
                    class_names=np.arange(self.num_classes).tolist(),
                    gt_class_ids=np.array(labels),
                    pred_class_ids=prediction_classes_arr,
                )
                results.update(result)
            else:
                raise NotImplementedError(f"metric {metric} is not Implemented.")
        return results

    def _compute(
        self,
        meta_values: Union[List, torch.Tensor],
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
    ):
        results = {}
        unique_meta_values = torch.unique(torch.tensor(meta_values)).tolist()
        for unique_key in unique_meta_values:
            indices = torch.where(meta_values == unique_key)[0].tolist()
            filtered_predictions = list(map(predictions.__getitem__, indices))
            filtered_labels = list(map(labels.__getitem__, indices))
            results[unique_key] = self._compute_metrics(filtered_predictions, filtered_labels)

        return results

    def _cumulative_metrics(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        current_epoch: int,
        phase: str,
    ):
        """Compute the mentioned metrics on the whole dataset."""
        results = {
            f"Full {phase} Dataset": [
                self._compute_metrics(predictions=predictions, labels=labels)
            ]
        }
        predictions_arr: torch.Tensor = torch.nn.Softmax(dim=-1)(torch.stack(predictions))
        predictions_arr = (
            predictions_arr.cpu() if predictions_arr.device.type == "cuda" else predictions_arr
        )
        senc_spec_l = get_sensitivity_specificity_list(
            num_classes=self.num_classes,
            gt_class_idx_l=np.array(labels),
            pred_class_prob_l=predictions_arr.numpy(),
        )
        results["SencSpecList"] = senc_spec_l
        fig = plot_auc_curve(
            gt_class_idx_l=np.array(labels),
            pred_class_prob_l=predictions_arr.numpy(),
            n_classes=self.num_classes,
        )
        self.logger.report_matplotlib_figure(
            title="AUC", series=phase, iteration=current_epoch, figure=fig
        )
        self._log2clearml(results, phase, current_epoch)

    def _shared_computation(
        self, outputs: List[Dict[str, torch.Tensor]], phase: str, current_epoch: int
    ):
        metadata, predictions, labels = self._extract_data(outputs)
        self._cumulative_metrics(
            predictions=predictions, labels=labels, phase=phase, current_epoch=current_epoch
        )
        results = {}
        for metadata_key, metadata_values in metadata.items():
            results[metadata_key] = self._compute(metadata_values, predictions, labels)
        return results

    def _log2clearml(self, results: Dict, phase: str, iteration: int):
        if self.logger is None:
            logger.warning(
                "make sure `_setup_clearml_task` was called before calling this function."
            )
            logger.warning(f"Skipping logging for phase {phase} iteration {iteration}")
            return
        for meta, result in results.items():
            self.logger.report_table(
                phase, meta, iteration=iteration, table_plot=pd.DataFrame(result).T
            )

    # def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    #     if self.logger is None:
    #         self._setup_clearml_task(trainer=trainer)
    #     if trainer.max_epochs - 1 == trainer.current_epoch:
    #         # for validation only log the last epoch results
    #         results = self._shared_computation(
    #             trainer.model.validation_outputs, "val", trainer.current_epoch
    #         )
    #         self._log2clearml(results, "val", trainer.current_epoch)

    # def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    #     if self.logger is None:
    #         self._setup_clearml_task(trainer=trainer)
    #     if trainer.max_epochs - 1 == trainer.current_epoch:
    #         # for train only log the last epoch results
    #         results = self._shared_computation(
    #             trainer.model.training_outputs, "train", trainer.current_epoch
    #         )
    #         self._log2clearml(results, "train", trainer.current_epoch)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.logger is None:
            self._setup_clearml_task(trainer=trainer)
        results = self._shared_computation(
            trainer.model.test_outputs, "test", trainer.current_epoch
        )
        self._log2clearml(results, "test", trainer.current_epoch)
