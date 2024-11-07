from typing import Dict, List, Union
import torch
from torch.nn.modules.loss import _Loss
from .combined_loss import CombinedLoss 
from loguru import logger
import torch.nn.functional as F
import time

class KD(_Loss):
    def __init__(
        self,
        losses: Union[CombinedLoss , _Loss],  
        alpha: float,
        temperature: float, 
        kwargs = {}
    ) -> None:
        
        """
        The CombinedLossDiffInput class is a subclass of the PyTorch nn.Module class that 
        computes the combined loss of two different inputs.
        """
        super().__init__(losses, kwargs)
        self.losses = losses
        self.alpha = alpha
        self.temperature = temperature
    
    def forward(self, input: Dict[str, torch.Tensor], target: Union[torch.Tensor,  Dict[str, torch.Tensor]], additional_data: Dict) -> torch.Tensor:

        teacher_dict = {}
        student_dict = {}

        for key, value in input.items():
            if key.startswith("teacher"):
                teacher_dict[key] = value
            elif key.startswith("student"):
                student_dict[key] = value

        if isinstance(target , dict) :
            pass
        else :
            assert len(student_dict) == 1 
            student_logits = next(iter(student_dict.values()))
            teacher_logits = next(iter(teacher_dict.values()))

            p = F.log_softmax(student_logits/self.temperature, dim=1)
            q = F.softmax(teacher_logits/self.temperature, dim=1)
            l_kl = F.kl_div(p, q, size_average=False) * (self.temperature**2)/student_logits.shape[0]

            if additional_data  == {} :
                loss = self.losses(student_logits, target)
            else :
                loss = self.losses(student_logits, target, additional_data)
        
        return l_kl * self.alpha + loss * (1. - self.alpha)


            



