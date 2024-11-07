import numpy as np
import torch
import torchvision
from lightning.pytorch import LightningModule, Trainer

from .clearml_callback import ClearMLCallBack


class LogSegResults(ClearMLCallBack):
    def __init__(
        self,
        num_samples: int,
        epoch_interval: int,
        log_gt_label: bool = False,
        images_key: str = "images",
        outputs_key: str = "outputs",
        labels_key: str = "labels",
    ) -> None:
        """Saves Predictions to ClearML across epochs.

        Note: This only works on tensorboard.
        Args:
        num_samples: maximum number of sample to log to clearml.
        epoch_interval: Epoch interval after which log the images.
        log_gt_label: log the GT label also.
        """
        super().__init__()
        self.num_samples = num_samples
        self.epoch_interval = epoch_interval
        self.log_gt_label = log_gt_label
        self.images_key = images_key
        self.outputs_key = outputs_key
        self.labels_key = labels_key
        self.logger = None

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.logger is None:
            self._setup_clearml_task(trainer=trainer)

        if trainer.current_epoch % self.epoch_interval == 0:
            validation_outputs = trainer.model.validation_outputs
            images = validation_outputs[0][self.images_key].squeeze(1)
            labels = validation_outputs[0][self.labels_key].squeeze(1)
            outputs = torch.cat(
                [output[1].unsqueeze(0) for output in validation_outputs[0][self.outputs_key]]
            )

            labels_indices = labels.sum(axis=[2, 3])
            pos_indices = labels_indices

            for i in range(min([pos_indices.shape[0], self.num_samples])):
                indices = torch.nonzero(pos_indices[i])
                for index in indices:
                    grid = torchvision.utils.make_grid(
                        [
                            images[i][index].as_tensor(),
                            labels[i][index].as_tensor(),
                            outputs[i][index].as_tensor(),
                        ]
                    )
                    # TODO: check why clearml upload are not working
                    # self.logger.report_image("Validation", "DebugImages", iteration=trainer.current_epoch, image=grid.cpu().numpy().astype(np.uint8)[0] if grid.device.type=="cuda" else grid.numpy(), delete_after_upload=True)
                    trainer.logger.experiment.add_image(
                        f"Epoch-{trainer.current_epoch}-SampleNo-{i}-CropSliceIndex{index.item()}-image-label-output",
                        grid,
                        global_step=trainer.global_step,
                    )
