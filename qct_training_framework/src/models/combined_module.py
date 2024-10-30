from typing import Dict, List, Optional, Union, Callable

import lightning.pytorch as pl
import torch
import torchmetrics
import os
from omegaconf import DictConfig
from loguru import logger
from clearml import Task
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class CombinedLitModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        activation: torch.nn.Module,
        metric: torchmetrics.Metric,
        knowledge_distillation = False,
        # metric: Union[List[Callable], DictConfig[str, List[Callable]]],
        image_key: str = "image",
        label_keys: Union[str, List[str]] = "label",
        class_names: Dict[str, List[str]] = None,
        additional_keys: Optional[List[str]] = None,
        subgroups_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.activation = activation
        self.knowledge_distillation = knowledge_distillation
        self.image_key = image_key
        self.label_keys = label_keys
        self.class_names = class_names
        self.additional_keys = additional_keys

        self.li = []

        self.train_metrics = metric.clone()
        self.val_metrics = metric.clone()
        self.test_metrics = metric.clone()
        self.metrics = {
            "train_metric": self.train_metrics,
            "val_metric": self.val_metrics,
            "test_metric": self.test_metrics,
        }
        # self.logger.add(os.path.join("/home/users/shubham.kumar/projects/qct_training_framework", "train.log"), level="DEBUG")

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)

    def shared_step(self, batch, phase: str):
        images = batch[self.image_key]
        batch_size = images.shape[0]

        if isinstance(self.label_keys, str):
            labels = batch[self.label_keys].squeeze()
            
        else:
            labels = {key: batch[key].squeeze() for key in self.label_keys}

        if self.additional_keys is not None :
            try :
                additional_data = {key: batch[key] for key in self.additional_keys} 
            except : 
                additional_data = {}
        else :
            additional_data = {}
        
        model_outputs = self(images)

        loss = self.criterion(model_outputs, labels)

        if self.knowledge_distillation :
            keys_to_remove = [key for key in model_outputs.keys() if key.startswith("teacher")]
            for key in keys_to_remove:
                model_outputs.pop(key)

            req_dict = {}
            for key, value in model_outputs.items():
                if key.startswith("student_"):
                    new_key = key.split("_")[1]  # Extract the label key
                    req_dict[new_key] = value

            if len(req_dict) == 1 :
                model_outputs = next(iter(req_dict.values()))
            else: 
                model_outputs = req_dict
            
                   
        self.log(
            f"{phase}_loss",
            loss,
            prog_bar=True,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        if isinstance(model_outputs, dict):
            pred = {}
            for key, value in model_outputs.items():
                pred[key] = self.activation(value)
        else:
            pred = self.activation(model_outputs)

        self.metrics[f"{phase}_metric"].update(pred, labels)

        return {"loss": loss}

    def shared_epoch_end(self, phase: str):
        metrics = self.metrics[f"{phase}_metric"].compute()

        if isinstance(metrics, dict) :
            metrics = get_flattend_metrics(metrics)
            for key, value in metrics.items():
                if phase != "train" :
                    if "plot" in key and value is not None: 
                        # self.log_plots_in_clearml(key,value)
                        continue
                else :
                    if "plot" in key:
                        continue
                if value is None :
                    continue
                if value.ndim == 1:  ## metric average is "none" or None
                    self.log(f"{phase}_{key}", value.mean(), sync_dist=True)
                    for label_key in self.class_names.keys():
                        classes = self.class_names[label_key]
                        if len(classes) == value.shape[0]:
                            for i, class_name in enumerate(classes):
                                self.log(
                                    f"{phase}_{label_key}_{class_name}",
                                    value[i],
                                    sync_dist=True,
                                )
                else:
                    self.log(f"{phase}_{key}", value, sync_dist=True)
        else :
            self.log(f"{phase}_{self.label_keys}", metrics, sync_dist=True)


    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("val")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def configure_optimizers(self):
        
        # logger.debug(f"parameters : {self.net.parameters()} ")
        optimizer = self.optimizer(params=self.parameters())

        try:
            scheduler = self.scheduler(
                optimizer=optimizer, total_steps=self.trainer.estimated_stepping_batches
            )  # for one cycle lr we need to pass total_steps
        except:  # noqa
            scheduler = self.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
        # scheduler = self.scheduler(optimizer=optimizer)
        # return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss",
        #         "interval": "epoch",
        #         "frequency": 1,
        #     },
        # }
    # def log_plots_in_clearml(self, key ,plot_metrics):
    #     logger.debug(self.trainer.logger.hparams)
    #     task_id = self.trainer.logger.hparams["clearml_task_id"]
    #     if task_id:
    #         task = Task.get_task(task_id=task_id)
    #         clearml_logger = task.get_logger()
    #         clearml_logger.report_matplotlib_figure(
    #             title="Manual Reporting", series=key, iteration=None, figure=plot_metrics
    #         )
            
# Function to flatten the nested dictionary
def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_flattend_metrics(metrics: Dict):
    # Flatten the nested dictionary with custom keys
    custom_keys_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened_value = flatten_dict({sub_key: sub_value}, parent_key=key)
                custom_keys_metrics.update(flattened_value)
        else:
            custom_keys_metrics[key] = value

    return custom_keys_metrics

