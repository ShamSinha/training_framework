import torch
import lightning.pytorch as pl
from loguru import logger


class MAE3DModule(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler):

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def shared_step(self, batch, phase: str) :
        
        image = batch["image"].to(dtype=torch.float16)
        image = image.permute(0, 1, 3, 4, 2)
        batch_size = image.shape[0]

        loss = self.model(image, apply_mask = phase == "train")

        self.log(name=f"{phase}_loss", value=loss, prog_bar=True, batch_size=batch_size, on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")
    
    def shared_epoch_end(self, phase):
        pass
       
    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("val")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def configure_optimizers(self):
        optimizer = self.optimizer(params=[x for x in self.parameters() if x.requires_grad])
        # try:
        scheduler = self.scheduler(optimizer=optimizer, total_steps=self.trainer.estimated_stepping_batches)
        #      # for one cycle lr we need to pass total_steps
        # except:  # noqa
        #     scheduler = self.scheduler(optimizer=optimizer)
        # return [optimizer]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
        
    
