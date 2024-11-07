import torch
import lightning.pytorch as pl
from loguru import logger

from qct_utils.schema.ct import ChestCTMaster

from transformers import VideoMAEConfig, VideoMAEForPreTraining, ViTMAEConfig, ViTMAEForPreTraining


class VideoMAEModule(pl.LightningModule):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 image_size : int ,
                 tubelet_size: int,
                 num_channels: int ,
                 num_frames: int,
                 patch_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 
                 mask_percentage: float):

        super().__init__()
        # self.save_hyperparameters()
        self.model_config = VideoMAEConfig(image_size=image_size, tubelet_size=tubelet_size, num_channels=num_channels, num_frames=num_frames, patch_size= patch_size,
                                           num_hidden_layers = num_hidden_layers, num_attention_heads=num_attention_heads)

        self.model = VideoMAEForPreTraining(config=self.model_config)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_frames = num_frames

        num_patches_per_frame = (self.model.config.image_size // self.model.config.patch_size) ** 2
        self.seq_length = (self.num_frames // self.model.config.tubelet_size) * num_patches_per_frame

        num_true = int(mask_percentage*self.seq_length)
        
        # Create a tensor with zeros (False) and ones (True)
        self.bool_tensor = torch.cat([torch.ones(num_true, dtype=torch.bool),
                             torch.zeros(self.seq_length - num_true, dtype=torch.bool)])
        
    
    def shared_step(self, batch, phase: str) :
        image = batch["image"].swapaxes(1,2).to(dtype=torch.float16)
        batch_size = image.shape[0]
        
        # Shuffle the tensor to distribute the True values randomly
        bool_masked_pos = []
        for i in range(batch_size) :
            bool_masked_pos.append(self.bool_tensor[torch.randperm(self.seq_length)])

        out = self.model(image , bool_masked_pos = torch.vstack(bool_masked_pos))
        loss = out.loss

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
        
    
