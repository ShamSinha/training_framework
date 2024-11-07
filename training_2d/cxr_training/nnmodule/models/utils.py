from torch.nn import Softmax as softmax
import torch
import importlib


def get_preds_from_logits(model_out, args):
    preds = {"classification_out": {}, "segmentation_out": {}}
    recipe = args.trainer.recipe
    cls_heads = args.cls.heads
    seg_heads = args.seg.heads

    if "cls" in recipe:
        for tag in cls_heads:
            preds["classification_out"][tag] = apply_softmax(
                model_out["classification_out"][tag]
            )

    if "seg" in recipe:
        for tag in seg_heads:
            preds["segmentation_out"][tag] = apply_softmax(
                model_out["segmentation_out"][tag], mode="seg"
            )

    if "age" in recipe:
        preds["age_out"] = model_out["age_out"]

    return preds


def apply_softmax(model_out, mode: str = "cls"):
    if mode == "seg":
        return softmax(dim=1)(model_out)[:, 1, ...]
    else:
        return softmax(dim=1)(model_out)[..., 1]


def get_class_from_str(class_str):
    """
    Dynamically imports a class from a string.

    Args:
    class_str (str): A string representation of the class, e.g., 'torch.optim.SGD'.

    Returns:
    class: The Python class the string refers to.
    """
    module_name, class_name = class_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def filter_negative_targets(preds, targets):
    """
    To check it
    ```
    import torch
    targets = torch.ones(4, 1)
    targets[0] = -100
    preds = torch.ones(4, 1, 512, 512)
    filter_negative_targets(preds, targets)
    ```
    Returns
    >>> preds.shape
    torch.Size([3, 1, 512, 512])
    >>> print(targets)
    tensor([[1.],
    [1.],
    [1.]])
    """
    # print(targets)
    raw_pos_inds = (targets >= 0).nonzero(as_tuple=True)[0].tolist()
    pos_inds = torch.Tensor(list(set(raw_pos_inds))).long().to(preds.device)
    preds = torch.index_select(preds, 0, pos_inds)
    targets = torch.index_select(targets, 0, pos_inds)
    # pos_inds = pos_inds.cpu()
    return preds, targets
