import torch
import lightning.pytorch as pl
from loguru import logger
import torch.nn.functional as F
import os

class SAM3DModule(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler):

        super().__init__()
        # self.save_hyperparameters()

        checkpoint_path = "/home/users/utkarsh.singh/qct/medsam/SAM-Med3D/work_dir/lidc_finetune/sam_model_dice_best.pth"

        self.model = model

        self.init_checkpoint(checkpoint_path)

        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def shared_step(self, batch, phase: str) :

        image = batch["image"]
        gt = batch["mask"]
        points = batch["points"]

        image_embedding = self.model.image_encoder(image)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )

        prev_masks = F.interpolate(low_res_masks, size=gt.shape[-3:], mode='trilinear', align_corners=False)

        image = batch["image"].to(dtype=torch.float16)
        image = image.permute(0, 1, 3, 4, 2)
        batch_size = image.shape[0]

        loss = self.model(image)

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
    
    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            last_ckpt = torch.load(ckp_path, map_location=self.args.device)
        if last_ckpt:
            self.model.load_state_dict(last_ckpt['model_state_dict'])
            # print(f"SAM-Med3D size: {sum(p.numel() for p in self.model.parameters())}")
            if not self.args.resume:
                self.start_epoch = 0 
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                # self.losses = last_ckpt['losses']
                # self.dices = last_ckpt['dices']
                # self.best_loss = last_ckpt['best_loss']
                # self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")
