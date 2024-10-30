from typing import Any

import torch
from torchmetrics import Metric


class AbstractMetric(Metric):
    def __init__(self, **kwargs: Any) -> None:
        """Abstract class for metrics."""
        super().__init__(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> None:
        raise NotImplementedError("update method is not implemented")

    def tensor2np(self, tensor: torch.Tensor):
        """Converts both cpu and gpu tensors to list."""
        return tensor.cpu().detach().numpy() if tensor.is_cuda else tensor.detach().numpy()

    def compute(self) -> Any:
        """Compute should return a tensor of a dict where each of the value is a tensor."""
        raise NotImplementedError("compute method is not implemented")
