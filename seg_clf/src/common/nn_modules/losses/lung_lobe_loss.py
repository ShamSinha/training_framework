from typing import Dict, List, Union
import torch
from torch.nn.modules.loss import _Loss
from .combined_loss import CombinedLoss
from monai.networks import one_hot
from torchmetrics import Dice
from skimage import morphology
from torchmetrics import MeanSquaredError
from .hausdroff_loss import compute_hausdorff_monai
from loguru import logger

class CombinedLungLobeLoss(_Loss):
    def __init__(
        self,
        activation: torch.nn.Module = None,
        kwargs = {}
    ) -> None:
        
        """
        The CombinedLossDiffInput class is a subclass of the PyTorch nn.Module class that 
        computes the combined loss of two different inputs.
        """
        super().__init__(kwargs)
        self.activation = activation
    
    def forward(self, input: List[torch.Tensor], target: List[torch.Tensor]) -> torch.Tensor:

        preds = [self.activation(logits) for logits in input]
        # logger.debug(f"preds shape : {preds[0].shape} ,  {preds[1].shape}")
        preds = [torch.argmax(pred, dim=1) for pred in preds]
        # logger.debug(f"preds shape : {preds[0].shape} ,  {preds[1].shape}")

        # logger.debug(f"target shape : {target[0].shape} ,  {target[1].shape}")

        for i in range(len(preds)):
            if len(torch.unique(target[i])) == 6 :
                preds[i][preds[i] == 2] = 2
                preds[i][preds[i] == 1] = 2
                preds[i][preds[i] == 5] = 1
                preds[i][preds[i] == 4] = 1
                preds[i][preds[i] == 3] = 1

                new_pred = preds[i]

            if len(torch.unique(target[i])) == 3 :
                new_target = preds[i].to(torch.uint8)

        dice = Dice(average='macro' , multiclass=True , num_classes=3).to(input[0].device)
        dice_loss = 1 - dice(new_pred, new_target)

        # mse = MeanSquaredError()
        # new_pred_boundary = get_boundary(new_pred)
        # new_target_boundary = get_boundary(new_target)
        # mse_loss = mse(new_target_boundary, new_pred_boundary)

        return dice_loss

        # compute_hausdorff_monai(new_pred_boundary, new_target_boundary, 100)


def get_boundary(mask: torch.Tensor) :

    kernel_t = torch.ones((3,3,3)).unsqueeze(0).unsqueeze(0)
    dilated_mask  = torch.clamp(torch.nn.functional.conv3d(mask, kernel_t, padding=(1,1,1)), 0, 1)
    eroded_mask = 1 - torch.clamp(torch.nn.functional.conv3d(1 - mask, kernel_t, padding=(1,1,1)), 0, 1)
    boundary_mask = dilated_mask - eroded_mask

    return boundary_mask


def lung_lobe_custom_loss(input: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) : 
    activation = torch.nn.Softmax(dim=1)

    lobe_output = activation(input["lobe_label"]) # (B,6,160,224,224)
    lung_output = activation(input["lung_label"]) # (B,3,160,224,224)
    new_lung_output = torch.zeros_like(lung_output)  # (B,3,160,224,224)

    new_lung_output[:,0,:,:,:] = lobe_output[:,0,:,:,:] 
    new_lung_output[:,1,:,:,:] = lobe_output[:,3,:,:,:]+ lobe_output[:,4,:,:,:] + lobe_output[:,5,:,:,:]
    new_lung_output[:,2,:,:,:]  = lobe_output[:,1,:,:,:] + lobe_output[:,2,:,:,:]

    









        











    
        
            


        


        










        








        
    

