from typing import Dict, Iterable, Tuple, List
import torch
from torchmetrics import Metric ,Accuracy
from torch.nn import Module, ModuleDict, ModuleList
from monai.networks import one_hot
from collections import OrderedDict
from loguru import logger

class CombinedDiffInputMetrics(Metric):
    full_state_update = True
    def __init__(
        self,
        metrics_func: Dict[str, List[Metric]],
        num_classes: Dict[str , int],
        kwargs = {}
    ) -> None:
        
        """
        The CombinedMetricsDiffInput class is a subclass of the PyTorch nn.Module class that 
        combine the all the metrics from different tasks in one metrics dictionary.  
        """
        super().__init__()
        self.num_classes = num_classes
        self.metrics = {}

        for key in metrics_func.keys() : 
            metrics = metrics_func[key]
            for m in metrics : 
                new_key = f"{key}_{(type(m).__name__)}"
                self.metrics[new_key] = m

        self.kwargs = kwargs
        
    def items(self) -> Iterable[Tuple[str, Module]]:
        """Return an iterable of the ModuleDict key/value pairs."""
        return self.metrics.items()

    def update(self, output: Dict[str, torch.Tensor], label: Dict[str, torch.Tensor]):
        """Update the metric state."""

        for metric_name, m in self.items():
            for key in output.keys() :
                if metric_name[:len(key)] == key : 
                    m.to(output[key].device)
                    if label[key].shape[0] ==0 :
                        continue 
                    # if output[key].ndim == 2 : 
                    #     if self.num_classes[key] == 2 and output[key].shape[1] == 2 : # binary
                    #         m.update(output[key][:,1],  label[key])
                    #     else :
                    #         m.update(output[key],  label[key])
                    # else:
                    try : 
                        m.update(output[key],  label[key])
                    except :
                        m.update(output[key],  one_hot(label[key] , self.num_classes[key]))

    def compute(self):
        """Compute the result for each metric in the collection."""
        res = {k: m.compute() for k, m in self.items()}
        # overall dice score
        if "dice" in self.kwargs : 
            dice = []
            for key in res.keys():
                if res[key].ndim == 0 : # metrics is scalar that is achieved using average in "micro" and "macro" mode
                    dice.append(res[key])
                elif res[key].ndim == 1 : #metrics is vector that is achieved using average in "none" or None mode
                    dice.append(res[key].mean())

            res["dice"] = torch.stack(dice).mean().to(res[key].device)

        return res


    