import os
import time
from pytorch_lightning.callbacks import ModelCheckpoint


def generate_tag_checkpoints(args):
    """logging checkpoint got best auroc, iou and least loss for each tag.

    Returns:
        [list]: list of ModelCheckpoint
    """

    unique_id = int(time.time() % 2000)
    tag_checkpoints = []
    dirpath = os.path.join(
        args.path.checkpoint_dir, args.trainer.description, args.trainer.model_file
    )
    os.makedirs(dirpath, exist_ok=True)

    qfilename = f"model_{unique_id}-" + "{epoch:02d}-{" + "val_loss" + ":.6f}"
    tag_checkpoints.append(
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=dirpath,
            filename=qfilename,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=True,
            save_weights_only=False,
            save_top_k=5,
            save_last=False,
        )
    )

    qfilename = f"model_{unique_id}-" + "last"
    tag_checkpoints.append(
        ModelCheckpoint(
            monitor="epoch",
            mode="max",
            dirpath=dirpath,
            filename=qfilename,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=True,
            save_weights_only=False,
            save_top_k=10,
        )
    )
    """
    all_clases = set(args.cls.heads + args.seg.heads)
    save_last = True
    for each_tag in all_clases:
        metric_types = ["BinaryAUROC", "qIOU"]
        if each_tag not in args.seg.heads or "seg" not in args.trainer.recipe:
            metric_types.remove("qIOU")
        if each_tag not in args.cls.heads or "cls" not in args.trainer.recipe:
            metric_types.remove("BinaryAUROC")

        for each_metric_type in metric_types:
            qfilename = (
                f"model_{unique_id}-"
                + "{epoch:02d}-{"
                + f"val_{each_tag}_default_{each_metric_type}"
                + ":.3f}"
            )
            tag_checkpoints.append(
                ModelCheckpoint(
                    monitor=f"val_{each_tag}_default_{each_metric_type}",
                    mode="max",
                    dirpath=dirpath,
                    filename=qfilename,
                    every_n_epochs=2,
                    save_on_train_epoch_end=False,
                    auto_insert_metric_name=True,
                    save_weights_only=False,
                    save_top_k=3,
                    save_last=save_last,
                )
            )
            save_last = False
    """

    return tag_checkpoints
