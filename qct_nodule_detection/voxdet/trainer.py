import torch
import lightning.pytorch as pl
import fastcore.all as fc 
import numpy as np 
from omegaconf import OmegaConf, DictConfig

from loguru import logger

from voxdet.networks.monai_retina3d import retina_detector
from voxdet.utils import clean_state_dict
from voxdet.metrics.det_metrics import DetMetrics
from voxdet.bbox_func.nms import monai_nms
from voxdet.infer import subset_cfg_for_infer
from time import time

class DetTrainer(pl.LightningModule):
    def __init__(self, 
                net: DictConfig,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                inference: DictConfig):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.net_cfg  = OmegaConf.merge(net, inference)

        self.store_cfg = subset_cfg_for_infer(self.net_cfg)
        self.model = retina_detector(self.net_cfg)  
        self.metrics = [DetMetrics(iou_thr=j, conf_thr=0.05) for j in np.arange(1, 11, 1)/20]
        self.to_np = lambda x: x.detach().cpu().numpy()# if x is not None else x
        self.to_cpu = lambda x: x.detach().cpu()
        self.to_tfloat = lambda x: torch.Tensor([x]).type(torch.float32)

    def forward(self, images, targets=None, use_inferer=False):
        output = self.model(images, targets, use_inferer) 
        return output 
    
    def training_step(self, batch, batch_idx):
        images, gt_box, gt_labels = batch[:3]

        targets = [{"boxes": box, "labels": labels.long()} for box, labels in zip(gt_box, gt_labels)]
        
        output = self(images, targets)        
        logs = {}

        # weighted_ratio = getattr(self, "weighted_loss_ratio", 1.0)
        # logs["ratio"] = weighted_ratio.to(self.device)

        loss = output["classification"] + output["box_regression"]
        logs["loss"] = loss

        logs["cls_loss"] = output["classification"]
        logs["reg_loss"] = output["box_regression"]
        for k, v in logs.items(): 
            self.log(name=f"train/{k}", value=v.to(self.device), prog_bar=True, batch_size=images.shape[0], on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": loss, "metrics": logs}

    def on_validation_epoch_start(self) -> None:
        logger.debug("validation starts")
        self.validation_start_time = time()
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        images, gt_box, _ = batch[:3]

        logits = self(images, None, use_inferer=True)

        if self.net_cfg.infer_thr.nms_thr is not None: 
            logits = [monai_nms(i, self.net_cfg.infer_thr.nms_thr, self.net_cfg.infer_thr.conf_thr, False) for i in logits]   

        for pred, gtb in zip(logits, gt_box):
            for meter in self.metrics:
                meter.to(self.device)
                meter.update(pred["boxes"] , pred["scores"], gtb)

    def on_validation_epoch_end(self):
        logger.debug(time() - self.validation_start_time)

        # Compute final metrics based on updates made during each validation step
        mAP, mAR = [], []
        for meter in self.metrics:
            metric_results = meter.compute()
            mAP.append(metric_results["AP"])
            mAR.append(metric_results["recall"])
            if metric_results["conf"] == 0.9 and metric_results["iou"] == 0.1 : 
                for k, v in metric_results.items():
                    if k not in ["AP_interp", "FROC_interp", "FROC_thresholds"]:
                        # Log individual metric results, e.g., AP at specific IoU thresholds
                        self.log(f"val/{k}", self.to_tfloat(v).to(self.device), prog_bar=True, sync_dist=True)
            
        # Calculate and log mean AP and mean AR across all IoU thresholds
        mean_mAP = sum(mAP) / len(mAP)
        mean_mAR = sum(mAR) / len(mAR)
        self.log("val_mAP", self.to_tfloat(mean_mAP).to(self.device), prog_bar=True, sync_dist=True)
        self.log("val_mAR", self.to_tfloat(mean_mAR).to(self.device), prog_bar=True, sync_dist=True)
      
        # # Optionally, reset metrics after logging final values to prepare for the next epoch
        for meter in self.metrics:
            meter.reset()        

    def configure_optimizers(self):
        optimizer = self.optimizer([x for x in self.model.parameters() if x.requires_grad])

        try:
            logger.debug(self.trainer.estimated_stepping_batches)
            scheduler = self.scheduler(
                optimizer=optimizer , total_steps = self.trainer.estimated_stepping_batches
            )  # for one cycle lr we need to pass total_steps
            logger.debug(scheduler)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        except:  # noqa
            # scheduler = self.scheduler(optimizer=optimizer, mode = 'max')
            # logger.debug(scheduler)
            return {
                "optimizer": optimizer,
                # "lr_scheduler": {
                #     "scheduler": scheduler,
                #     "monitor": "val_mAP",
                # },
            }


    # def lr_scheduler_step(self, scheduler, metric):
        # scheduler.step(epoch=self.current_epoch)  # timm's scheduler needs the epoch value
    
    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["state_dict"] = clean_state_dict(checkpoint["state_dict"])
        checkpoint["cfg"] = self.store_cfg