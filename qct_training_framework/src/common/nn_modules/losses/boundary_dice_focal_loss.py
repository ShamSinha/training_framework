from typing import Dict
import torch
from monai.losses import DiceFocalLoss
from skimage import morphology


class BoundaryDiceFocalLoss(DiceFocalLoss):
    def __init__(
        self,
        factor_voxels_inside_boundary: float = 1.0,
        factor_voxels_on_boundary: float = 1.0,
        factor_voxels_outside_boundary: float = 0.0,
        kwargs: Dict = {},
    ) -> None:
        """Modified loss function to factor pixels inside, outside, on boundarys
           eroded & dilation masks are calculated using `skimage.morphology`
           voxels on boundary are calculated using `erosion xor mask `
                  inside boundary are `erosion and mask`
                  outside boundary are `dilation xor mask`
           default values is same as DiceFocalLoss
        Args:
            factor_voxels_inside_boundary_factor (float, optional): Factor for all pixels inside  boundary. Defaults to 1.0.
            factor_voxels_on_boundary (_type_, optional): Factor for all pixels on  boundary. Defaults to 1.0,
            factor_voxels_outside_boundary(float, optional): Factor for all pixels just outside  boundary:float=0.0.
            kwargs (Dict, optional): kwargs for `DiceFocalLoss`. Defaults to {}.
        Returns:
            _type_: _description_
        """
        # Make reduction none
        kwargs["reduction"] = "none"
        super().__init__(**kwargs)
        self.factor_voxels_inside_boundary = factor_voxels_inside_boundary
        self.factor_voxels_on_boundary = factor_voxels_on_boundary
        self.factor_voxels_outside_boundary = factor_voxels_outside_boundary

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        target = target.to(torch.int)
        bsize = target.shape[0]
        pixel_masks_on_boundary = []
        pixel_masks_outside_boundary = []
        pixel_masks_inside_boundary = []
        pixel_masks_background = []

        for i in range(bsize):
            mask = target[i][0]
            eroded = (
                torch.Tensor(morphology.binary_erosion(mask.cpu().numpy()))
                .to(torch.int)
                .to(target.device)
            )
            dilated = (
                torch.Tensor(morphology.binary_dilation(mask.cpu().numpy()))
                .to(torch.int)
                .to(target.device)
            )
            mask_inside_boundary = torch.logical_and(mask, eroded).to(torch.int)
            pixel_masks_inside_boundary.append(mask_inside_boundary.unsqueeze_(0))

            mask_on_boundary = torch.logical_xor(mask, eroded).to(torch.int)
            pixel_masks_on_boundary.append(mask_on_boundary.unsqueeze_(0))

            mask_outside_boundary = torch.logical_xor(mask, dilated).to(torch.int)
            pixel_masks_outside_boundary.append(mask_outside_boundary.unsqueeze_(0))

            # This is needed as we include background in focal loss. 
            mask_background = torch.logical_not(dilated).to(torch.int)
            pixel_masks_background.append(mask_background.unsqueeze_(0))
        pixel_masks_inside_boundary = torch.stack(pixel_masks_inside_boundary, dim=0)
        pixel_masks_on_boundary = torch.stack(pixel_masks_on_boundary, dim=0)
        pixel_masks_outside_boundary = torch.stack(pixel_masks_outside_boundary, dim=0)
        pixel_masks_background = torch.stack(pixel_masks_background, dim=0)

        # Calculate on `mask logical_or dilated_mask`
        dilated_target = (
            pixel_masks_inside_boundary
            + pixel_masks_on_boundary
            + pixel_masks_outside_boundary
        )
        total_loss = super().forward(input, dilated_target)

        weighted_target = (
            pixel_masks_inside_boundary * self.factor_voxels_inside_boundary
            + pixel_masks_on_boundary * self.factor_voxels_on_boundary
            + pixel_masks_outside_boundary * self.factor_voxels_outside_boundary
            + pixel_masks_background
        )

        final_loss = (total_loss * weighted_target).mean()
        # dfl_loss = total_loss.mean().item()
        # print(f"{dfl_loss:.2f}, {final_loss.item():.2f}")
        return final_loss
