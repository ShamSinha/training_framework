from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
from monai.utils.type_conversion import convert_to_tensor
from omegaconf import DictConfig
from torchmetrics import Metric


class SubgroupMetric(torch.nn.Module):
    def __init__(
        self,
        subgroups: Optional[DictConfig[str, List[int]]],
        metrics: List[Metric],
    ):
        """
        Subgroup metric class to compute metrics for subgroups of the dataset.
        Wont work for multi-gpu testing.
        Args:
            subgroups: Dictionary of subgroups. Keys are the subgroup names and values are the list of subgroup values.
            metrics: List of metrics to compute for each subgroup.
        Example:
        ```
            from torchmetrics import Accuracy
            from torchmetrics.classification import  MulticlassSpecificity
            subgroups = {
                "age": [0, 1, 2, 3],
                "gender": [1, 1, 0, 1, 0],
            }
            subgroup_metric = SubgroupMetric(
                subgroups=subgroups,
                metrics=[
                Accuracy(task="multiclass", num_classes=3),
                MulticlassSpecificity(task="multiclass", num_classes=3)
                ]
            )
            subgroup_metric(torch.randn(2,3), torch.ones(2, dtype=torch.long), age=torch.tensor([1,2]), gender=torch.tensor([1,2]))
            subgroup_metric.compute()
        ```
        """
        super().__init__()
        self.subgroups = subgroups if subgroups is not None else {}
        self.subgroup_names = list(self.subgroups.keys())
        self.metrics = torch.nn.ModuleList([deepcopy(met) for met in metrics])
        self.subgroup_metrics = torch.nn.ModuleDict({})
        self.sample_size = {}
        for subgroup_name, subgroup_values in self.subgroups.items():
            for subgroup_value in subgroup_values:
                self.subgroup_metrics[f"{subgroup_name}_{subgroup_value}"] = torch.nn.ModuleList(
                    [deepcopy(met) for met in metrics]
                )
                self.sample_size[f"{subgroup_name}_{subgroup_value}"] = 0
            self.subgroup_metrics[f"{subgroup_name}_-1"] = torch.nn.ModuleList(
                [deepcopy(met) for met in metrics]
            )  # to handle None cases
            self.sample_size[f"{subgroup_name}_-1"] = 0

    def tensor2list(self, tensor: torch.Tensor):
        """Converts both cpu and gpu tensors to list."""
        return (
            tensor.cpu().detach().numpy().tolist()
            if tensor.is_cuda
            else tensor.detach().numpy().tolist()
        )

    def update_tensor(
        self, preds: torch.Tensor, target: torch.Tensor, additional_data_dicts: Dict = {}, **kwargs
    ):
        # full dataset metric update
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            add_data = additional_data_dicts.get(metric_name, {})
            metric.update(preds, target, **add_data)
        # subgroup metric update
        for sub_name in self.subgroup_names:
            subgroup = (
                self.tensor2list(kwargs[sub_name]) if sub_name in kwargs else [-1] * len(preds)
            )
            for i in range(len(subgroup)):
                self.sample_size[f"{sub_name}_{subgroup[i]}"] += 1
                for metric in self.subgroup_metrics[f"{sub_name}_{subgroup[i]}"]:
                    add_data = additional_data_dicts.get(metric.__class__.__name__, {})
                    if add_data != {}:
                        # if there are any metrics that needs additional data
                        metric.update(
                            preds[i].unsqueeze(0),
                            target[i].unsqueeze(0),
                            **{
                                k: torch.Tensor([v[i]]).to(target[i].device)
                                for k, v in add_data.items()
                            },
                        )
                    else:
                        metric.update(preds[i].unsqueeze(0), target[i].unsqueeze(0))

    def update(
        self,
        preds: Union[torch.Tensor, List[torch.Tensor]],
        target: Union[torch.Tensor, List[torch.Tensor]],
        additional_data_dicts: Dict[str, Dict[str, torch.Tensor]] = {},
        **kwargs,
    ):
        if isinstance(preds, torch.Tensor):
            self.update_tensor(preds, target, additional_data_dicts, **kwargs)
        else:
            raise TypeError("preds and target should be either torch.Tensor or List[torch.Tensor]")

    def forward(
        self, preds: torch.Tensor, target: torch.Tensor, additional_data_dicts: Dict = {}, **kwargs
    ):
        self.update(preds, target, additional_data_dicts, **kwargs)

    @staticmethod
    def get_default_metric_result(metric: Metric):
        # for some of the subgroups there might not be any samples
        # in that case, we return -1 as the metric result
        try:
            return metric.compute()
        except Exception as e:
            # print(f"Error in computing metric {metric.__class__.__name__}: {e}")
            return -1

    def get_subgroup_name(self, name: str):
        # replace _ to - in subgroup names except the last one
        names = name.split("_")
        return "-".join(names[:-1]) + "_" + names[-1]

    def compute(self):
        # full dataset metric compute
        # print(self.count)
        metrics = {}
        for metric in self.metrics:
            metrics[metric.__class__.__name__] = convert_to_tensor(
                self.get_default_metric_result(metric)
            )
        # subgroup metric compute
        for sub_metric_name in self.subgroup_metrics:
            for metric in self.subgroup_metrics[sub_metric_name]:
                metrics[
                    f"{self.get_subgroup_name(sub_metric_name)}_{metric.__class__.__name__}"
                ] = {
                    "result": convert_to_tensor(self.get_default_metric_result(metric)),
                    "sample_size": torch.tensor(self.sample_size[sub_metric_name]),
                }

        return metrics
