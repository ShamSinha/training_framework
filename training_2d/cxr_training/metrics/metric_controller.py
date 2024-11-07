from torchmetrics import Accuracy, AUROC, R2Score
from torchmetrics import MetricCollection
from cxr_training.metrics.iou import qIOU, qIOU_corrected
from torchmetrics.classification import BinaryAveragePrecision
from torchmetrics.classification import BinaryAUROC, BinarySpecificity, BinaryRecall


def default_metric(model_type: str, prefix: str):
    if model_type == "cls":
        return MetricCollection(
            [
                # Accuracy(dist_sync_on_step=True),
                # MyAccuracy(dist_sync_on_step=True),
                BinaryAUROC(
                    thresholds=None,
                    dist_sync_on_step=True,
                    compute_on_step=False,
                    ignore_index=-100,
                ),
                BinarySpecificity(threshold=0.5, ignore_index=-100),
                BinaryRecall(threshold=0.5, ignore_index=-100),
                # BinaryAveragePrecision(thresholds=None),
                # Recall(dist_sync_on_step=True),
                # Specificity(dist_sync_on_step=True),
            ],
            prefix=prefix,
        )
    elif model_type == "seg":
        return MetricCollection(
            [
                qIOU_corrected(num_classes=2, dist_sync_on_step=False),
                # qIOU(num_classes=2, dist_sync_on_step=False),
                # IOU(num_classes=2, dist_sync_on_step=True, ignore_index=-100),
            ],
            prefix=prefix,
        )

    elif model_type == "age":
        return MetricCollection(
            [
                R2Score(num_classes=2, dist_sync_on_step=False, compute_on_step=False),
                # IOU(num_classes=2, dist_sync_on_step=True, ignore_index=-100),
            ],
            prefix=prefix,
        )


metric_type_dict = {"default": default_metric}


def metrics_controller(model_type: str, prefix: str, metric_type: str = "default"):
    """
    Obtain an instance of the metric based on the model type and metric_type.

    Args:
        model_type (str): Specifies the type of model. Can be:
            - "cls": For classification.
            - "seg": For segmentation.
            Example: model_type="cls"

        prefix (str): Prefix for logging the metric. Typically includes the training mode ("train", "val", or "test")
                      followed by the class name.
            Example: prefix="train_opacity_"

        metric_type (str, optional): Specifies the metric configuration from the metric_type_dict.
                                     Defaults to "default".

    Returns:
        MetricCollection: A collection of metrics suitable for the specified model type. MetricCollection is a PyTorch
                          container that bundles multiple metrics into a single module. This aids in more organized
                          logging and easier batch-wise computation.

    Usage:
        >>> metrics = metrics_controller("seg", "val_example_")
        >>> print(metrics)
        MetricCollection([
            qIOU(num_classes=2, dist_sync_on_step=False, compute_on_step=False)
        ])
    """
    return metric_type_dict[metric_type](model_type, prefix)
