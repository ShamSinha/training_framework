import torch.nn as nn
import torch


def vanilla_classification_loss(args):
    cls_loss_wts = args.cls.loss_wts
    cls_losses_heads = args.cls.heads
    cls_alpha = args.cls.alpha
    loss_list = [
        nn.CrossEntropyLoss(torch.FloatTensor([1, cls_loss_wts[tag]]))
        for tag in cls_losses_heads
    ]

    def cls_loss_fn(model_cls_out, cls_target):
        loss = {}
        for i, tag in enumerate(cls_losses_heads):
            """(1, class_wt) to account for pred for normal, class_tag."""
            model_out = model_cls_out[cls_losses_heads[i]]
            try:
                cls_target_out = cls_target[cls_losses_heads[i]].long()
                cross_entropy_loss = loss_list[i].to(model_out.device)
                loss[f"{tag}_cls"] = (
                    cross_entropy_loss(model_out, cls_target_out) * cls_alpha
                )
            except RuntimeError as e:
                if (
                    str(e)
                    != "Expected floating point type for target with class probabilities"
                ):
                    raise ValueError("check the target values")
                """due to Expected floating point type for target
                with class probabilities, got Long"""

                cls_target_out = cls_target[cls_losses_heads[i]]
                cross_entropy_loss = loss_list[i].to(model_out.device)
                loss[f"{tag}_cls"] = (
                    cross_entropy_loss(model_out, cls_target_out) * cls_alpha
                )
            cross_entropy_loss = cross_entropy_loss.cpu()
        return loss

    return cls_loss_fn
