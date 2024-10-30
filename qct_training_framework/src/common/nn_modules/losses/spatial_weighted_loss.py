from typing import Dict, List, Union
import torch
from torch.nn.modules.loss import _Loss
from .combined_loss import CombinedLoss
from loguru import logger
import time

class SpatialWeightedLoss(_Loss):
    def __init__(
        self,
        loss: Union[CombinedLoss, _Loss],
        weights : List[float], 
        roi_mask_keys :List[str] , 
        kwargs = {}
    ) -> None:
        
        """
        The CombinedLossDiffInput class is a subclass of the PyTorch nn.Module class that 
        computes the combined loss of two different inputs.
        """
        super().__init__(kwargs)
        self.loss = loss
        self.weights = weights
        self.roi_mask_keys = roi_mask_keys
        assert len(self.weights) == len(self.roi_mask_keys), "Number of weights and ROI masks must be the same."

    
    def forward(self, input: torch.Tensor, target: torch.Tensor, additional_data: Dict) -> torch.Tensor:
        base_loss = self.loss(input, target)
        spatial_losses = []

        for i, roi_mask_key in enumerate(self.roi_mask_keys):
            try:
                roi_mask = additional_data[roi_mask_key] > 0.5
                spatial_weights = torch.ones_like(target)
                spatial_weights = torch.where(roi_mask, self.weights[i] * torch.ones_like(target), spatial_weights)
                spatial_loss = base_loss * spatial_weights
                spatial_losses.append(spatial_loss)
            except KeyError:
                # Handle the case when the ROI mask is not found in additional_data
                continue
        
        if len(spatial_losses) > 0:
            # Calculate the mean loss over all pixels
            loss = torch.mean(torch.stack(spatial_losses, dim=0))
        else:
            # If no spatial losses were computed, use the base loss
            loss = base_loss
        return loss

