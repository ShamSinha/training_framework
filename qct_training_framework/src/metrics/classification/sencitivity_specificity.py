from typing import Dict, List, Optional, Union

import numpy as np
from sklearn import metrics
from torch import Tensor


def compute_sencitivity_specificity(
    class_names: List[str],
    gt_class_ids: Union[np.ndarray, Tensor],
    pred_class_ids: Optional[Union[np.ndarray, Tensor]] = None,
):
    """Given a list of `gt_class_ids` and `pred_class_ids` values compute sencitivity, specificity
    and accuracy for `class_names` classes indivisually and overall macro average."""
    assert pred_class_ids is not None and len(pred_class_ids.shape) == 1
    assert len(gt_class_ids.shape) == 1

    def _compute_metric(
        y_true: np.ndarray, y_pred: np.ndarray, pos_label: Union[int, bool], round_off: int = 2
    ):
        try:
            result = round(
                metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label), round_off
            )
        except Exception as e:
            result = str(e)
        return result

    result = {}
    if len(class_names) > 2:
        sensitivity_l = []
        specificity_l = []
        for index, class_name in enumerate(class_names):
            y_true = gt_class_ids == index
            y_pred = pred_class_ids == index
            sensitivity = _compute_metric(
                y_true=y_true, y_pred=y_pred, pos_label=True, round_off=2
            )
            specificity = _compute_metric(
                y_true=y_true, y_pred=y_pred, pos_label=False, round_off=2
            )
            result[f"sensitivity_{class_name}"] = sensitivity
            result[f"specificity_{class_name}"] = specificity
            specificity_l.append(specificity)
            sensitivity_l.append(sensitivity)
        try:
            result["specificity"] = np.mean(specificity_l)
            result["sensitivity"] = np.mean(sensitivity_l)
        except:  # noqa
            result["specificity"] = None if "specificity" not in result else result["specificity"]
            result["sensitivity"] = None if "sensitivity" not in result else result["sensitivity"]

    else:
        sensitivity = _compute_metric(
            y_true=gt_class_ids, y_pred=pred_class_ids, pos_label=1, round_off=2
        )
        specificity = _compute_metric(
            y_true=gt_class_ids, y_pred=pred_class_ids, pos_label=0, round_off=2
        )
        result["sensitivity"] = sensitivity
        result["specificity"] = specificity

    return result


def get_sensitivity_specificity_list(
    num_classes: int,
    gt_class_idx_l: np.ndarray,
    pred_class_prob_l: np.ndarray,
    conf_l=np.arange(0.1, 1, 0.05),
):
    result = {}
    classes = np.arange(num_classes).tolist()
    if num_classes > 2:
        for conf in conf_l:
            conf = round(conf, 2)
            result[conf] = {}
            sensitivity_macro = []
            specificity_macro = []
            for i in range(num_classes):
                y_true = gt_class_idx_l == i
                y_pred = pred_class_prob_l[:, i] > conf
                sensitivity = round(
                    metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label=True), 2
                )
                specificity = round(
                    metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label=False), 2
                )
                result[conf].update(
                    {f"sensitivity_{i}": sensitivity, f"specificity_{i}": specificity}
                )
                sensitivity_macro.append(sensitivity)
                specificity_macro.append(specificity)
            result[conf].update(
                dict(
                    sensitivity_macro=round(np.mean(sensitivity_macro), 2),
                    specificity_macro=round(np.mean(specificity_macro), 2),
                )
            )
    else:
        for conf in conf_l:
            y_pred = np.array(pred_class_prob_l[:, 1] > conf, dtype=np.int8)
            sensitivity = metrics.recall_score(y_true=gt_class_idx_l, y_pred=y_pred, pos_label=1)
            specificity = metrics.recall_score(y_true=gt_class_idx_l, y_pred=y_pred, pos_label=0)
            acc = np.sum(np.array(gt_class_idx_l) == np.array(y_pred, dtype=np.int8)) / len(
                gt_class_idx_l
            )
            result[conf] = {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": acc,
            }
    return result
