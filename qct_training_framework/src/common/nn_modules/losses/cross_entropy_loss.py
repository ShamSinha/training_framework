import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss

from lightning.pytorch.trainer import Trainer
from loguru import logger
import os

logger.add(os.path.join("/home/users/shubham.kumar/projects/qct_training_framework", "loss.log"), level="DEBUG")

class CrossEntropyLossFunc(_Loss):
    def __init__(
        self,
        ignore_index: int = -100,
        weights: torch.Tensor = None,
        kwargs = {}
    ) -> None:
        
        """
        The CombinedLossDiffInput class is a subclass of the PyTorch nn.Module class that 
        computes the combined loss of two different inputs.
        """
        super(CrossEntropyLossFunc, self).__init__()
        self.weights = weights
        self.ignore_index = ignore_index
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        device = input.device

        # If weights are provided, move them to the same device and use them for the loss calculation
        if self.weights is not None:
            weights = self.weights.to(device)
            loss_fn = CrossEntropyLoss(ignore_index= self.ignore_index, weight= weights)
        else:
            loss_fn = CrossEntropyLoss(ignore_index= self.ignore_index)

        loss = loss_fn(input, target)
        if torch.isnan(loss) == True : 
            loss = 0
        return loss

        # compute_hausdorff_monai(new_pred_boundary, new_target_boundary, 100)

