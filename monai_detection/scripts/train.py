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

def _intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

class WeightedLoss:
    #TODO: move this to callbacks 
    """type is None, compute will always return None"""
    def __init__(self, type="auto"): fc.store_attr(); self.lt = [];
    def update(self, x): 
        if self.type == None: pass 
        x = torch.Tensor([x["reg_loss"], x["cls_loss"]])
        if len(self.lt) == 0: self.lt = x
        else: self.lt = torch.vstack([self.lt, x])
        if self.lt.shape[0] >100: self.lt[:, -100:]
            
    def compute(self): 
        if self.type==None: return torch.Tensor([1.])
        if len(self.lt) <=100: return torch.Tensor([1.])
        reg, cls = self.lt[:, -100:].median(axis=0).values
        if reg >= cls: return torch.max(torch.Tensor([reg/cls, 1/10]))
        return torch.main(torch.Tensor([reg/cls, 10]))
    
    def reset(self): self.lt=[]

class DetTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        #self.learning_rate = self.cfg.learning_rate
        self.store_cfg = subset_cfg_for_infer(self.cfg)
        self.model = retina_detector(cfg)
        self.metrics = [DetMetrics(iou_thr=j, conf_thr=0.05)for j in np.arange(1, 11, 1)/20]
        self.to_np = lambda x: x.detach().cpu().numpy()# if x is not None else x
        self.to_cpu = lambda x: x.detach().cpu()
        self.to_tfloat = lambda x: torch.Tensor([x]).type(torch.float32)
        self.wl = WeightedLoss(self.cfg.loss_weight)
        self.wl.reset()
        self.val_step_outputs = []

    def forward(self, images, targets=None, use_inferer=False):
        output = self.model(images, targets, use_inferer) 
        return output 
    
    def training_step(self, batch, batch_idx):
        #try:
        images, gt_box, gt_labels = batch[:3]
        targets = [{"boxes": box, "labels": labels.long()} for box, labels in zip(gt_box, gt_labels)]
        output = self(images, targets)
        logs = {}
        logs["ratio"] = self.wl.compute()
        loss = (logs["ratio"].to(self.device)*output["classification"]) + output["box_regression"]
        # loss = (output["classification"] + output["box_regression"])/8
        logs["loss"] = self.to_cpu(loss)
        logs["cls_loss"] = self.to_cpu(output["classification"])
        logs["reg_loss"] = self.to_cpu(output["box_regression"])
        if self.wl.type is not None: self.wl.update(logs)
        for k, v in logs.items(): 
            self.log(name=f"train/{k}", value=v, prog_bar=True, batch_size=images.shape[0], on_step=True, on_epoch=True, sync_dist=self.cfg.distributed)
        return {"loss": loss, "metrics": logs}
        # except Exception as e:
        #     logger.info(f"failed {batch_idx}: {e}")
        #     return None 

    def validation_step(self, batch, batch_idx):
        try:
            images, gt_box, gt_labels = batch[:3]
            with torch.no_grad(): logits = self(images, None, use_inferer=True)
            if self.cfg.infer_thr.nms_thr is not None: logits = [monai_nms(i, self.cfg.infer_thr.nms_thr, self.cfg.infer_thr.conf_thr) for i in logits]
            outs = [{"preds": pred, "gt": {"boxes": self.to_np(gtb), "labels": self.to_np(gtl)}} for pred, gtb, gtl in zip(logits, gt_box, gt_labels)]
            self.val_step_outputs.append(outs)
            return outs 
        except Exception as e:
            logger.info(f"failed {batch_idx}: {e}")
            return None 

    def on_validation_epoch_end(self):
        [i.reset() for i in self.metrics]
        # self.metrics.reset()
        validation_step_outputs = [i for i in self.val_step_outputs if i is not None]
        vso = [j for i in validation_step_outputs for j in i]
        for so in vso:
            preds, gt = so["preds"], so["gt"]
            for meter in self.metrics: meter.update(preds["boxes"], preds["scores"], gt["boxes"])
        m = [i.compute() for i in self.metrics]
        # m = self.metrics.compute()
        for k, v in m[0].items(): # log all other metrics at iou_thr == 0.1
            if k not in ["AP_interp", "FROC_interp", "FROC_thresholds"]: self.log("val/"+k, self.to_tfloat(v), prog_bar=True, sync_dist=self.cfg.distributed)
        ap, ar = [], []
        for metric in m:
            ap.append(metric["AP"])
            ar.append(metric["recall"])
        mAP, mAR = sum(ap)/len(ap), sum(ar)/len(ar)
        # mAP = sum([metric["AP"] for metric in m])/len(m)
        # mAR = sum([metric["recall"] for metric in m])/len(m)

        self.log("val/mAP", self.to_tfloat(mAP), prog_bar=True, sync_dist=self.cfg.distributed)
        self.log("val/mAR", self.to_tfloat(mAR), prog_bar=True, sync_dist=self.cfg.distributed)
        [i.reset() for i in self.metrics]
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
    
    if hasattr(cfg, "load_selfsup_from") & (cfg.load_selfsup_from is not None):
        logger.info(f"[Weights loading from selfsup model: {cfg.load_selfsup_from}]")
        from voxdet.networks.selfsup_utils import load_from_selfup_retina
        csd = load_from_selfup_retina(cfg.load_selfsup_from, model.model)
        model.model.network.feature_extractor.body.load_state_dict(csd, strict=True)

    
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