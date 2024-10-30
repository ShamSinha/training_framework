from typing import Dict, List, Union
import torch
from torch.nn.modules.loss import _Loss
from .combined_loss import CombinedLoss 
from loguru import logger

class CombinedLossDiffInput(_Loss):
    def __init__(
        self,
        losses: Dict[str, Union[CombinedLoss , _Loss]],  
        weights: Dict[str , float],
        loss_inputs : Dict[str , Union[str , List[str]]],
        activations: Dict[str , torch.nn.Module] = None,
        kwargs = {}
    ) -> None:
        
        """
        The CombinedLossDiffInput class is a subclass of the PyTorch nn.Module class that 
        computes the combined loss of two different inputs.
        """
        super().__init__(losses, weights, kwargs)
        self.losses = losses
        self.weights = weights
        self.loss_inputs = loss_inputs
        self.activations = activations
    
    def forward(self, input: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        if "slice_label" in target.keys() : 
            target["slice_label"] = torch.flatten(target["slice_label"])
        
        final_loss = 0
        
        for loss_key in self.loss_inputs.keys():
            keys = self.loss_inputs[loss_key]
            if isinstance(keys , str) : 
                if target[keys].ndim > 3 :  # (B,C,H,W) or (B,C,D,H,W)
                    indices_to_keep = []
                    for idx, im in enumerate(target[keys]):
                        if torch.sum(im) >= 0:
                            indices_to_keep.append(idx)
                    input[keys] = input[keys][indices_to_keep]
                    target[keys] = target[keys][indices_to_keep]

                if target[keys].shape[0] == 0 :
                    final_loss+= 0
                else : 
                    loss_input = self.activations[keys](input[keys])
                    loss_target = target[keys]
                    loss_func = self.losses[keys]
                    final_loss += self.weights[loss_key]*loss_func(loss_input, loss_target)
            else :
                loss_input = [self.activations[key](input[key]) for key in keys]
                loss_target = [target[key] for key in keys]
                loss_func = self.losses[loss_key]
                final_loss += (self.weights[loss_key])*loss_func(loss_input, loss_target)
 
        return final_loss
    

    