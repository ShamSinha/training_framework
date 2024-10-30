import torch
import lightning.pytorch as pl
import fastcore.all as fc 
import numpy as np 

from mmengine.config import Config 
from loguru import logger

from voxdet.networks.monai_retina3d import retina_detector
from voxdet.utils import locate_cls, import_module, clean_state_dict
from voxdet.metrics.det_metrics import DetMetrics
from voxdet.bbox_func.nms import monai_nms
from voxdet.infer import subset_cfg_for_infer

from medct.convnextv2mim import mask_patches

from lightning.pytorch.callbacks import ModelCheckpoint

def _intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

class DetTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model_cfg = locate_cls(self.cfg.model_cfg)
        self.model = locate_cls(self.cfg.model, return_partial=True)(config=self.model_cfg)
        #self.learning_rate = self.cfg.learning_rate
        self.store_cfg = subset_cfg_for_infer(self.cfg)
        self.to_np = lambda x: x.detach().cpu().numpy()# if x is not None else x
        self.to_cpu = lambda x: x.detach().cpu()
        self.to_tfloat = lambda x: torch.Tensor([x]).type(torch.float32)
        self.val_step_outputs = []

    def forward(self, images, bool_masked_pos):
        output = self.model(images, bool_masked_pos) 
        return output 
    
    def training_step(self, batch, batch_idx):
        #try:
        bool_masked_pos = mask_patches(self.model.num_patches, self.cfg.mask_ratio).to(self.device)
        output = self(batch[0], bool_masked_pos)
        logs = {}
        logs["rloss"] = self.to_cpu(output.loss)
        for k, v in logs.items(): 
            self.log(name=f"train/{k}", value=v, prog_bar=True, batch_size=batch[0].shape[0], on_step=True, on_epoch=True, sync_dist=self.cfg.distributed)
        return {"loss": output.loss}

    def validation_step(self, batch, batch_idx):
        bool_masked_pos = mask_patches(self.model.num_patches, self.cfg.mask_ratio).to(self.device)
        images = torch.cat([i.unsqueeze(0) for i in batch[0]])
        with torch.no_grad(): out = self(images, bool_masked_pos)
        self.val_step_outputs.append(out.loss)
        return out.loss
        
    def on_validation_epoch_end(self):
        metric = torch.stack(self.val_step_outputs).mean().detach().cpu()
        self.log("val/rloss", metric, prog_bar=True, sync_dist=self.cfg.distributed)
        self.val_step_outputs.clear()
        # self.metrics.reset()

    def configure_optimizers(self):
        optimizer = import_module(
            self.cfg.optimizer,
            params=[x for x in self.model.parameters() if x.requires_grad],
            #lr=self.learning_rate#remove
        )
        self.optimizer = optimizer

        if "scheduler" not in self.cfg.keys():
            return optimizer
        
        scheduler = import_module(self.cfg.scheduler, optimizer=optimizer)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "frequency": 1}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler needs the epoch value
    
    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["state_dict"] = clean_state_dict(checkpoint["state_dict"])
        checkpoint["cfg"] = self.store_cfg


def main(cfg_path: str):
    cfg = Config.fromfile(cfg_path)
    model = DetTrainer(cfg)
    dl = import_module(cfg.dataloader)

    ckpt_path = cfg.resume_from if hasattr(cfg, "resume_from") & isinstance(cfg.resume_from, str) else None
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)['state_dict']
        for key in list(checkpoint.keys()):
            if 'network.' in key:
                new_key = key.replace('network.', 'model.network.')
                # print(key, new_key)
                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]

        for key in list(checkpoint.keys()):
            if key in cfg.exclude_keys:
                del checkpoint[key]

        model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Resuming training from checkpoint : {cfg.resume_from}")

    if cfg.log_using_clearml:
        try:
            from clearml import Task
            task = Task.init(project_name=cfg.clearml_project_name, task_name=cfg.task_name)
            task.connect_configuration(name="cfg", configuration=dict(cfg))
            task.add_tags(cfg.get("tags", ["no_tag"]))
            logger.info(f"Saving exp to ClearML project: {cfg.clearml_project_name}")
        except:
            logger.warning("ClearML not installed; Logging with Tensorboard.")
    else:
        logger.warning("ClearML logs not used. Tracking with Tensorboard.")

    # Load checkpoints
    if hasattr(cfg, "load_from") & (cfg.load_from is not None):
        logger.info("[Loading model from %s]" % cfg.load_from)
        weights = torch.load(cfg.load_from)["state_dict"]
        csd = _intersect_dicts(weights, model.model.state_dict())
        model.model.load_state_dict(csd, strict=True)
    
    trainer = pl.Trainer(
        # resume_from_checkpoint=ckpt_path,
        devices=cfg.devices if cfg.accelerator != "cpu" else None,
        accelerator=cfg.accelerator, 
        strategy="ddp" if cfg.distributed else "auto",
        max_epochs=cfg.epochs,
        log_every_n_steps=10,
        check_val_every_n_epoch=5, 
        callbacks=[locate_cls(s) for s in cfg.callbacks],
        logger = locate_cls(cfg.logger)
        #logger=#pl.loggers.TensorBoardLogger(cfg.lightling_dir, name=cfg.task_name),
        #precision=16,
        #track_grad_norm=2,
        # profiler="simple",
    )
    # tune to find lr #remove
    trainer.fit(model, train_dataloaders=dl.train_dl(), val_dataloaders=dl.val_dl()) 

if __name__ == "__main__":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    import argparse
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("--cfg_path", type=str, required=True)
    args = parser.parse_args()
    main(args.cfg_path)
    # cfg_loc = "configs/135/v13_134v6_random_erase.py"
    # cfg = Config.fromfile(cfg_loc)
    # dl = import_module(cfg.dataloader)
    #x = dl.tds[0]
    # import time 
    # start = time.time()
    # for st in dl.train_dl():
    #    print(st[0].shape, [i.shape[0] for i in st[1]])
    # print(time.time()-start)
    #main(cfg_loc, "nodules")

