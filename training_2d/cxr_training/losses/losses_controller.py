import torch
from cxr_training.losses.classification_loss import (
    vanilla_classification_loss,
)
from cxr_training.losses.segmentation_losses import (
    vanilla_segmentation_loss,
    dice_bce_segmentation_loss,
    pixel_loss_dynamicalpha_scan,
    dice_bce_pixel_loss,
)

from cxr_training.losses.misc_losses import mse_loss


def losses_framework(args, cls_loss_fn, seg_loss_fn, age_loss_fn):
    # model_out, target_dict,
    """[summary]

    Args:
        model_out ([dict]): output of model being trained
        target_dict ([dict]): gts
        class_losses_and_wts ([dict]): class tags for which loss type (cls, seg, seg)
        is to be carried about and user input loss wts for each of them.
        user input loss wts alter loss by the ratio.

    Returns:
        [dict]: a dict with cls seg side losses based on input
    """
    cls_loss_fn = cls_loss_fn(args)
    seg_loss_fn = seg_loss_fn(args)
    # age_loss_fn = age_loss_fn(args)

    def criterion(model_out, target_dict):
        losses_dict = {}

        if "classification_target" in target_dict.keys():
            cls_loss = cls_loss_fn(
                model_out["classification_out"], target_dict["classification_target"]
            )
            losses_dict.update(cls_loss)

        if "segmentation_target" in target_dict.keys():
            seg_loss = seg_loss_fn(
                model_out["segmentation_out"], target_dict["segmentation_target"]
            )
            losses_dict.update(seg_loss)

        # if "age_target" in target_dict.keys():
        #     age_mse_loss = age_loss_fn(model_out["age_out"], target_dict["age_target"])
        #     losses_dict.update(age_mse_loss)

        # if it batch consists of only -100 then loss becomes nan due to which the sum of it also becomes nan
        # ex : torch.tensor(0.7009) + float('nan')
        losses_dict = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses_dict.items()}

        return losses_dict

    return criterion


def default_losses(args):
    return losses_framework(
        args, vanilla_classification_loss, vanilla_segmentation_loss, mse_loss
    )


def dice_losses(args):
    return losses_framework(
        args, vanilla_classification_loss, dice_bce_segmentation_loss, mse_loss
    )


def sq_dice_losses(args):
    return losses_framework(
        args, vanilla_classification_loss, dice_bce_segmentation_loss, mse_loss
    )

def dice_bce_pixel_losses(args):
    return losses_framework(
        args, vanilla_classification_loss, dice_bce_pixel_loss, mse_loss
    )

def pix_losses(args):
    return losses_framework(
        args, vanilla_classification_loss, pixel_loss_dynamicalpha_scan, mse_loss
    )


loss_type_dict = {
    "default": default_losses,
    "dice-bce": dice_losses,
    "dice-bce_pix_seg": dice_bce_pixel_losses,
    "dice-bce_pix": pix_losses,
    "sq_dice-bce": sq_dice_losses,
}


def losses_controller(args):
    return loss_type_dict[args.params.loss_type](args)
