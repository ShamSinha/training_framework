from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import lightning.pytorch as pl
import torch
from clearml import Task
from loguru import logger
from omegaconf import DictConfig
import ast
from ..metrics.subgroup_metric import SubgroupMetric

class LightningModel(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: Union[torch.nn.Module, DictConfig[str, torch.nn.Module]],
        metric: Union[List[Callable], DictConfig[str, List[Callable]]],
        image_key: str = "image",
        label_key: str = "label",
        additional_keys: Optional[Dict] = None,
        subgroups_dicts: Optional[DictConfig[str, List[int]]] = None,
        metric_additional_keys: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Common lightning module for all models training.

        Args:
            net: Model to train.
            optimizer: Optimizer to use.
            scheduler: Scheduler to use.
            criterion: Loss function to use. In case MTL is used, it should be a dict of loss functions ({task1: L1, task2: L2}).
            metric: Metric to use. In case MTL is used, it should be a dict of metrics ({task1: [M1, M2], task2: [M1, M2]}).
            additional_keys: Additional keys to pass to the model.
            subgroups_dicts: Dict of subgroups to use for subgroup metrics. Should be in the form {subgroup_name: [list of possible values]}.
            metric_additional_keys: Additional keys to pass to the metric. Eg. {"VolumeError": "voxel_vols"}
        """
        super().__init__()
        self.net = net
        self.criterion = criterion
        # train and val metrics are standard metrics

        logger.debug(f"{metric}")
        logger.debug(f"{isinstance(metric, DictConfig) }")

        self.metric = torch.nn.ModuleDict(
            {
                phase: (
                    torch.nn.ModuleDict(
                        {
                            k: torch.nn.ModuleList([deepcopy(met) for met in v])
                            for k, v in metric.items()
                        }
                    )
                    if isinstance(metric, DictConfig) 
                    else torch.nn.ModuleList([deepcopy(met) for met in metric])
                )
                for phase in ["train_met", "val_met"]
            }
        )
        # test metrics are subgroup metrics
        self.metric["test_met"] = (
            torch.nn.ModuleDict(
                {
                    k: torch.nn.ModuleList(
                        [SubgroupMetric(subgroups_dicts, [deepcopy(met) for met in v])]
                    )
                    for k, v in deepcopy(metric).items()
                }
            )
            if isinstance(metric, DictConfig)
            else torch.nn.ModuleList(
                [SubgroupMetric(subgroups_dicts, [deepcopy(met) for met in metric])]
            )
        )
        self.subgroup_keys = (
            list(subgroups_dicts.keys()) if isinstance(subgroups_dicts, DictConfig) else None
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.additional_keys = additional_keys
        self.image_key = image_key
        self.label_key = label_key
        self.metric_additional_keys = metric_additional_keys

        self.is_mtl = getattr(self.net, "is_mtl", False)
        self.label_keys = getattr(self.net, "tasks_list", None)
        if self.is_mtl:
            self.net.device = self.device

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)

    def shared_step(self, batch, batch_idx, phase: str):
        inputs = batch[self.image_key]
        batch_size = inputs.shape[0]
        additional_data = (
            {key: batch[key] for key in self.additional_keys}
            if self.additional_keys is not None
            else {}
        )
        outputs = self.forward(inputs, **additional_data)
        subgroup_dicts = (
            {key: batch[key] for key in self.subgroup_keys}
            if (self.subgroup_keys is not None and phase == "test")
            else {}
        )
        metric_additional_data = (
            {
                metric_name: {val: batch[val] for val in values}
                for metric_name, values in self.metric_additional_keys.items()
            }
            if self.metric_additional_keys is not None
            else {}
        )
        
        # changes for slicewise 
        img_out = outputs['scan_label']
        # logger.debug(batch['scan_label'])
        
        img_loss = self.criterion(img_out, batch['scan_label'])
        def multisoftmax_loss( outputs_list, targets_list, criterion=None):
            """Multi Softmax Loss.

            outputs_list(list): list of b x 2 of length c.
            targets_list(list): list of b of length c.
            """
            if targets_list == -1: 
                return 0
            #losses = []
            loss_sw = 0
            if criterion is None:
                criterion = nn.CrossEntropyLoss()

            for i, (x, y) in enumerate(zip(outputs_list, targets_list)):
                y = y
                x = x
                #losses.append(criterion(x, y))
                if torch.any(y == -1):
                    loss_sw += 0
                else :
                    try:
                        loss_sw += criterion(x, y)
                    except Exception as e:
                        breakpoint()
            return loss_sw
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # targets_list = []
        # for slicegt in batch['slicewise_gt']:
        #     targets = ast.literal_eval(slicegt)
        #     targets_tensor = torch.tensor(targets).to(device)
        #     targets_list.append(targets_tensor)


        # slicewise_loss = multisoftmax_loss(slice_out, targets_list)

        loss = img_loss #+ slicewise_loss
        outputs = img_out
        for i in range(len(self.metric[phase + "_met"])):
            if phase == "test":
                # if test/subgroup metric, pass metric_additional_data
                # it will be passed to the appropriate metric
                self.metric[phase + "_met"][i](
                    outputs, batch[self.label_key], metric_additional_data, **subgroup_dicts
                )
            else:
                add_ips = metric_additional_data.get(
                    self.metric[phase + "_met"][i].__class__.__name__, {}
                )
                self.metric[phase + "_met"][i](
                    outputs, batch[self.label_key], **add_ips, **subgroup_dicts
                )
        self.log(
            f"{phase}_loss",
            loss.item(),
            # img_loss.item(),
            # slicewise_loss.item(),
            prog_bar=True,
            batch_size=batch_size,
            on_step=False,  # too much logs, set to True if u want to check on step calculation
            on_epoch=True,
            sync_dist= True
        )
        return loss

    def shared_epoch_end(self, phase: str):
        # used to log the train and val metrics
        if self.is_mtl:  # multi task learning
            for i, task in enumerate(self.net.tasks_list):
                for i in range(len(self.metric[phase + "_met"][task])):
                    metric_results = self.metric[phase + "_met"][task][i].compute()
                    self.metric[phase + "_met"][task][i].reset()  # reset the metric
                    if isinstance(metric_results, Dict):
                        for k, v in metric_results.items():
                            if k == "plot":
                                # only log the plot during test phase
                                continue
                            self.log(
                                f"{phase}_{task}_{self.metric[phase+'_met'][task][i].__class__.__name__}_{k}",
                                v,
                                prog_bar=True,
                                on_step=False,
                                on_epoch=True,
                                sync_dist= True
                            )
                    else:
                        self.log(
                            f"{phase}_{task}_{self.metric[phase+'_met'][task][i].__class__.__name__}",
                            metric_results,
                            prog_bar=True,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True
                        )
        else:
            for i in range(len(self.metric[phase + "_met"])):
                metric_results = self.metric[phase + "_met"][i].compute()
                self.metric[phase + "_met"][i].reset()  # reset the metric
                if isinstance(metric_results, Dict):
                    for k, v in metric_results.items():
                        if k == "plot":
                            # only log the plot during test phase
                            continue
                        self.log(
                            f"{phase}_{self.metric[phase+'_met'][i].__class__.__name__}_{k}",
                            v,
                            prog_bar=True,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True
                        )
                else:
                    self.log(
                        f"{phase}_{self.metric[phase+'_met'][i].__class__.__name__}",
                        metric_results,
                        prog_bar=True,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True
                    )
        plt.close("all")

    def on_train_epoch_end(self):
        self.shared_epoch_end("train")

    def on_validation_epoch_end(self):
        self.shared_epoch_end("val")

    def log_single_task(
        self, logger, task_metric: Dict, task_name: Optional[str] = None, phase: str = "test"
    ):
        if "SencitivitySpecificityList" in task_metric:  # senc spec on full dataset
            senc_spec_df = pd.DataFrame(task_metric.pop("SencitivitySpecificityList")).T
            senc_spec_df = senc_spec_df.applymap(lambda x: x.numpy())
            logger.report_table(
                f"{phase}-{task_name}",
                "Sensitivity Specificity",
                iteration=0,
                table_plot=senc_spec_df,
            )
            # remove the subgroup analysis for the senc and spec list
            task_metric = {
                k: v for k, v in task_metric.items() if "SencitivitySpecificityList" not in k
            }
        # log metrics to clearml
        overall_metrics_dict = {  # metrics that returns tensors as outputs
            name: round(value.item(), 2)
            for name, value in task_metric.items()
            if "_" not in name
            if not isinstance(value, Dict)
        }
        overall_dict_metrics = {  # metrics that returns dict as outputs
            f"{name}-{name_}": round(value.item(), 2)
            for name, value in task_metric.items()
            if "_" not in name
            if isinstance(value, Dict)
            for name_, value in value.items()
            if name_ != "plot"
        }
        overall_metrics_dict.update(overall_dict_metrics)  # full dataset results

        overall_metrics = pd.DataFrame([overall_metrics_dict], index=["Full Dataset Result"]).T
        iteration = 0
        logger.report_table(  # log full dataset results
            f"{phase}-{task_name}",
            "Full Dataset results",
            iteration=iteration,
            table_plot=overall_metrics,
        )
        plots = {  # log plots
            f"{name}": value
            for name, value in task_metric.items()
            if "_" not in name and isinstance(value, Dict)
            for name_, value in value.items()
            if name_ == "plot" and value is not None
        }
        for name, value in plots.items():
            logger.report_matplotlib_figure(
                f"plots-{task_name}",
                f"{phase}-{name}",
                iteration=iteration,
                figure=value,
            )
        # log subgroup results
        subgrp_metrics = {}
        for name, value in task_metric.items():
            if "_" in name:  # subgroup metric
                subgrp_name, cls_idx, metric_name = name.split("_")
                if subgrp_name not in subgrp_metrics:
                    subgrp_metrics[subgrp_name] = {}
                if cls_idx not in subgrp_metrics[subgrp_name]:
                    subgrp_metrics[subgrp_name][cls_idx] = {}
                if isinstance(value["result"], Dict):  # if metric returns dict
                    for name_, value_ in value["result"].items():
                        if name_ != "plot":
                            # log numerical metrics
                            subgrp_metrics[subgrp_name][cls_idx].update(
                                {
                                    f"{metric_name}-{name_}": round(value_.item(), 2),
                                    "sample_size": value["sample_size"].item(),
                                }
                            )
                        # elif value_ is not None:
                        #     # log plots
                        #     logger.report_matplotlib_figure(
                        #         f"plots-{task_name}-{subgrp_name}",
                        #         f"{phase}-{cls_idx}-{metric_name}",
                        #         iteration=iteration,
                        #         figure=value_,
                        #     )
                else:  # if metric returns tensor
                    subgrp_metrics[subgrp_name][cls_idx].update(
                        {
                            metric_name: round(value["result"].item(), 2)
                            if value["result"] is not None
                            else 0,
                            "sample_size": value["sample_size"].item(),
                        }
                    )
        for task, result in subgrp_metrics.items():
            logger.report_table(
                f"{phase}-{task_name}",
                task,
                iteration=iteration,
                table_plot=pd.DataFrame(result).T,
            )

    def log2clearml(self, test_metric_results):
        # log test metrics to clearml
        task_id = self.trainer.logger.hparams["clearml_task_id"]
        if task_id:
            task = Task.get_task(task_id=task_id)
            clearml_logger = task.get_logger()
            if self.is_mtl:
                for task in self.net.tasks_list:
                    self.log_single_task(clearml_logger, test_metric_results[task], task)
            else:
                self.log_single_task(clearml_logger, test_metric_results)
        else:
            logger.warning(
                "`clearml_task_id` is not logged to logger hparams. Did you set `skip_clearml` to `False`?"
            )
            logger.info("TODO: Implement this!!!!. Printing to stdout for now.")
            print(test_metric_results)
        plt.close("all")

    def on_test_epoch_end(self) -> None:
        test_metric_results = {}
        if self.is_mtl:
            for task in self.net.tasks_list:
                test_metric_results[task] = self.metric["test_met"][task][0].compute()
        else:
            test_metric_results = self.metric["test_met"][
                0
            ].compute()  # test metric is always single element list.
        self.log2clearml(test_metric_results)

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        if self.is_mtl:
            return self.net.backward(loss)
        return super().backward(loss, *args, **kwargs)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if hasattr(self, "cfg"):
            checkpoint["cfg"] = self.cfg
        else:
            logger.warning(
                "cfg not found. Add cfg to lightning_module_instance to save along with checkpoint"
            )

    def configure_optimizers(self):
       # breakpoint()
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
