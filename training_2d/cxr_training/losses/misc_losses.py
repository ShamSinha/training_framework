import torch.nn as nn
import torch


def mse_loss(args):
    age_alpha = args.params.age_alpha
    loss_fn = torch.nn.MSELoss()

    def mse_loss_fn(model_cls_out, cls_target):
        loss = {}
        # Filter out targets with -100
        mask = cls_target != -100
        filtered_outputs = model_cls_out[mask]
        filtered_targets = cls_target[mask].float()
        loss["age_mse_loss"] = loss_fn(filtered_outputs, filtered_targets) * age_alpha
        # Calculate MSE only for the valid targets
        return loss

    return mse_loss_fn
