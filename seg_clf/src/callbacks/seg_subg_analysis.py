from typing import Dict, List, Union

import pandas as pd
import torch
from loguru import logger
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from lightning.pytorch import LightningModule, Trainer
from tqdm import tqdm

from .clearml_callback import ClearMLCallBack


class SegSubAnalysis(ClearMLCallBack):
    def __init__(
        self,
        subgroups_keys: List[str],
        outputs_key: str,
        labels_key: str,
        metrics: List[str] = ["Dice", "VolumeError"],
    ) -> None:
        """
        Subgroup analysis for segmentation models.
        Args:
        -----
        subgroups_keys: which subgroup keys to select.
        outputs_key: prediction key
        labels_key: gt label key
        metrics: metrics to compute per subgroup.
        Returns:
        --------
        Logs the subgroup metrics of last validation epoch and test epoch to clearml `PLOTS`
        tab.
        """
        super().__init__()
        self.subgroups_keys = subgroups_keys
        self.outputs_key = outputs_key
        self.labels_key = labels_key
        self.metrics = metrics
        self.logger = None

    def _extract_data(self, outputs: List[Dict[str, torch.Tensor]]):
        # extract metadata subgroups
        metadata = {}
        for meta_key in self.subgroups_keys:
            metadata[meta_key] = torch.cat([batch_output[meta_key] for batch_output in outputs])
        metadata["voxel_vol"] = torch.cat([batch_output["voxel_vol"] for batch_output in outputs])
        # extract outputs & labels
        # keeping as list instead of tensor because batch size can be different across batches
        predictions = [
            prediction for batch_output in outputs for prediction in batch_output[self.outputs_key]
        ]
        labels = [label for batch_output in outputs for label in batch_output[self.labels_key]]

        return metadata, predictions, labels

    def _compute_metrics(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        filtered_voxel_vols: List[torch.Tensor],
    ):
        """Add any new metrics here."""
        results = {}
        for metric in self.metrics:
            if metric == "Dice":
                dice_metric = DiceMetric(
                    include_background=False, reduction="mean", get_not_nans=False
                )
                post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
                post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
                seg_outputs = [post_pred(i) for i in predictions]
                seg_labels = [post_label(i) for i in labels]
                dice_metric(y_pred=seg_outputs, y=seg_labels)
                dice_score = dice_metric.aggregate().item()
                sample_size = len(seg_outputs)
                results.update({"Dice Score": dice_score, "Sample Size": sample_size})
            elif metric == "VolumeError":
                post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
                post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
                seg_outputs = [post_pred(i) for i in predictions]
                seg_labels = [post_label(i) for i in labels]
                pred_volume = torch.tensor(
                    [
                        pred[1].sum().item() * vox_vol.item()
                        for pred, vox_vol in zip(seg_outputs, filtered_voxel_vols)
                    ]
                )
                label_volume = torch.tensor(
                    [
                        pred[1].sum().item() * vox_vol.item()
                        for pred, vox_vol in zip(seg_labels, filtered_voxel_vols)
                    ]
                )
                sample_size = len(seg_outputs)
                volume_error = torch.mean(
                    torch.abs(pred_volume - label_volume) / label_volume
                ).item()
                results.update({"Dice Score mean": volume_error, "Sample Size": sample_size})

            else:
                raise NotImplementedError(f"metric {metric} is not Implemented.")
        return results

    def _compute(
        self,
        meta_values: Union[List, torch.Tensor],
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        voxel_vol: List[torch.Tensor],
    ):
        meta_values = (
            meta_values.clone().detach() if isinstance(meta_values, torch.Tensor) else meta_values
        )
        results = {}
        unique_meta_values = torch.unique(torch.tensor(meta_values)).tolist()
        for unique_key in unique_meta_values:
            indices = torch.where(meta_values == unique_key)[0].tolist()
            filtered_predictions = list(map(predictions.__getitem__, indices))
            filtered_labels = list(map(labels.__getitem__, indices))
            filtered_voxel_vols = list(map(voxel_vol.__getitem__, indices))
            results[unique_key] = self._compute_metrics(
                filtered_predictions, filtered_labels, filtered_voxel_vols
            )

        return results

    def _shared_computation(self, outputs: List[Dict[str, torch.Tensor]]):
        # returns results in following format
        # {"meta": {class_name: {metric_score: x, Sample Size: y}}}
        metadata, predictions, labels = self._extract_data(outputs)
        voxel_vol = metadata.pop("voxel_vol")
        results = {}
        for metadata_key, metadata_values in tqdm(
            metadata.items(), desc="Processing subgroup metadata", leave=False, total=len(metadata)
        ):
            results[metadata_key] = self._compute(
                metadata_values, predictions, labels, voxel_vol=voxel_vol
            )
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
    #         results = self._shared_computation(trainer.model.validation_outputs)
    #         self._log2clearml(results, "val", trainer.current_epoch)

    # def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    #     if self.logger is None:
    #         self._setup_clearml_task(trainer=trainer)
    #     if trainer.max_epochs - 1 == trainer.current_epoch:
    #         # for validation only log the last epoch results
    #         results = self._shared_computation(trainer.model.training_outputs)
    #         self._log2clearml(results, "train", trainer.current_epoch)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.logger is None:
            self._setup_clearml_task(trainer=trainer)
        results = self._shared_computation(trainer.model.test_outputs)
        self._log2clearml(results, "test", trainer.current_epoch)
