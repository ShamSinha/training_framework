from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from cxr_training.callbacks.model_checkpointing_callback import (
    generate_tag_checkpoints,
)


def configure_callbacks(args):
    callbacks_list = []

    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
    progress = TQDMProgressBar(refresh_rate=1)
    checkpoint_callback = generate_tag_checkpoints(args)

    callbacks_list.append(lr_monitor)
    callbacks_list.extend(checkpoint_callback)
    callbacks_list.append(progress)

    return callbacks_list
