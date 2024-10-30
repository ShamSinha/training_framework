from typing import Dict, List
import torch
from torch.nn.modules.loss import _Loss
from skimage import morphology
from omegaconf import DictConfig
from torch import nn
import numpy as np

class CombinedLoss(_Loss):
    def __init__(
        self,
        criterions: List[_Loss],
        weights: List[float],
        activations: List[nn.Module] = [nn.Identity()],
        kwargs = {}
    ) -> None:
        
        """
        The CombinedLoss class is used to calculate the combined loss of multiple loss functions. 
        It takes in a list of loss functions and their corresponding weights as input arguments. 
        The forward method of this class calculates the combined loss by summing the weighted losses of all the input loss functions.
        Args:
            The CombinedLoss class takes in the following input arguments:
            loss_functions(list): A list of loss functions to be combined.
            weights(list): A list of weights corresponding to each loss function. 
            The length of this list should be equal to the length of loss_functions
.
        Returns:
            The forward method of the CombinedLoss class returns the combined loss value as a scalar tensor.
        """
        # Make reduction none
        super().__init__(**kwargs)
        self.criterions = criterions
        sum_weights = np.sum(weights)
        self.weights = [weight/sum_weights for weight in weights]
        self.activations = activations

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        final_loss = 0
        for criterion, weight, activation in zip(self.criterions, self.weights, self.activations):
            final_loss += weight * criterion(activation(input), target)
        return final_loss
    

