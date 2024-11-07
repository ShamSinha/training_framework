import torch.nn as nn
import torch
import torch.nn.functional as F
from cxr_training.losses.segment_utils import (
    DiceBCELoss,
    SquaredDiceBCELoss,
    DiceBCELossPix,
    DiceBCELossPix_seg,
)


def vanilla_segmentation_loss(args):
    seg_losses_heads = args.seg.heads
    seg_loss_wts = args.seg.loss_wts
    seg_alpha = args.seg.alpha

    loss_list = [
        nn.CrossEntropyLoss(torch.FloatTensor([1, seg_loss_wts[tag]]), reduction="mean")
        for tag in seg_losses_heads
    ]

    def seg_loss_fn(model_seg_out, seg_target):
        loss = {}
        for i, tag in enumerate(seg_losses_heads):
            model_out = model_seg_out[seg_losses_heads[i]]
            cls_target_out = seg_target[seg_losses_heads[i]]
            cross_entropy_loss = loss_list[i].to(model_out.device)
            loss[f"{tag}_seg"] = (
                cross_entropy_loss(model_out, cls_target_out) * seg_alpha
            )
        return loss

    return seg_loss_fn


def dice_bce_segmentation_loss(args):
    seg_losses_heads = args.seg.heads
    seg_loss_wts = args.seg.loss_wts
    seg_alpha = args.seg.alpha
    seg_dice_threshold = args.seg.dice_threshold
    loss_list = [DiceBCELoss(ce_weight=seg_loss_wts[tag]) for tag in seg_losses_heads]

    def seg_loss_fn(model_seg_out, seg_target):
        loss = {}
        for i, tag in enumerate(seg_losses_heads):
            model_out = model_seg_out[seg_losses_heads[i]]
            cls_target_out = seg_target[seg_losses_heads[i]]
            dice_bce_loss = loss_list[i].to(model_out.device)
            # bce_loss = loss_list[i]['bce_loss'].to(model_out.device)
            # dice_loss = loss_list[i]['dice_loss'].to(model_out.device)

            dice_bce_loss_values = dice_bce_loss(model_out, cls_target_out, seg_dice_threshold[i])
            loss[f"{tag}_seg"] = (
                dice_bce_loss_values['dice_bce']
                * seg_alpha
            )
            loss[f"{tag}_bce"] = dice_bce_loss_values['bce_loss']
            loss[f"{tag}_dice"] = dice_bce_loss_values['dice_loss']
        return loss

    return seg_loss_fn


def squared_dice_bce_segmentation_loss(args):
    seg_losses_heads = args.seg.heads
    seg_loss_wts = args.seg.loss_wts
    seg_alpha = args.seg.alpha
    seg_dice_threshold = args.seg.dice_threshold
    loss_list = [
        SquaredDiceBCELoss(ce_weight=seg_loss_wts[tag]) for tag in seg_losses_heads
    ]

    def seg_loss_fn(model_seg_out, seg_target):
        loss = {}
        for i, tag in enumerate(seg_losses_heads):
            model_out = model_seg_out[seg_losses_heads[i]]
            cls_target_out = seg_target[seg_losses_heads[i]]
            sq_dice_bce_loss = loss_list[i].to(model_out.device)
            loss[f"{tag}_seg"] = (
                sq_dice_bce_loss(model_out, cls_target_out, seg_dice_threshold[i])
                * seg_alpha
            )
        return loss

    return seg_loss_fn

def dice_bce_pixel_loss(args):
    seg_losses_heads = args.seg.heads
    seg_loss_wts = args.seg.loss_wts
    seg_alpha = args.seg.alpha
    seg_dice_threshold = args.seg.dice_threshold
    loss_list = [DiceBCELossPix_seg(ce_weight=seg_loss_wts[tag], reduction='mean') for tag in seg_losses_heads]

    def seg_loss_fn(model_seg_out, seg_target):
        loss = {}
        for i, tag in enumerate(seg_losses_heads):
            model_out = model_seg_out[seg_losses_heads[i]]
            cls_target_out = seg_target[seg_losses_heads[i]]
            dice_bce_loss = loss_list[i].to(model_out.device)
            # bce_loss = loss_list[i]['bce_loss'].to(model_out.device)
            # dice_loss = loss_list[i]['dice_loss'].to(model_out.device)

            dice_bce_loss_values = dice_bce_loss(model_out, cls_target_out, seg_dice_threshold[i])
            loss[f"{tag}_seg"] = (
                dice_bce_loss_values['dice_bce']
                * seg_alpha
            )
            loss[f"{tag}_bce"] = dice_bce_loss_values['bce_loss']
            loss[f"{tag}_dice"] = dice_bce_loss_values['dice_loss']
        return loss

    return seg_loss_fn

def pixel_loss_dynamicalpha_scan(args):
    seg_losses_heads = args.seg.heads
    seg_loss_wts = args.seg.loss_wts
    seg_alpha = args.seg.alpha
    seg_dice_threshold = args.seg.dice_threshold
    exponent = args.seg.exponent
    loss = DiceBCELossPix

    loss_list = [
        loss(ce_weight=seg_loss_wts[tag], reduction="none") for tag in seg_losses_heads
    ]

    def px_loss(model_seg_out, seg_target):
        loss = {}
        for i, tag in enumerate(seg_losses_heads):
            model_out = model_seg_out[seg_losses_heads[i]]
            cls_target_out = seg_target[seg_losses_heads[i]]

            dice_bce_loss = loss_list[i].to(model_out.device)
            loss_val = dice_bce_loss(model_out, cls_target_out, seg_dice_threshold[i])

            pixel_sum = cls_target_out.sum([1, 2])
            alpha_tensor = cls_target_out[0].numel() / (pixel_sum + 1)
            alpha_tensor = (
                (((alpha_tensor > 0) * alpha_tensor) ** args.seg.exponent[i]) * (pixel_sum > 0)
                + (pixel_sum == 0) * 1
                + (pixel_sum < 0) * 0.0
            )
            alpha_tensor = alpha_tensor/alpha_tensor.sum()

            loss[f"{tag}_seg"] = (loss_val['dice_bce'] * alpha_tensor).mean() * seg_alpha
            loss[f"{tag}_bce"] = (loss_val['bce_loss']).mean()
            loss[f"{tag}_dice"] = (loss_val['dice_loss']).mean()
        return loss

    return px_loss