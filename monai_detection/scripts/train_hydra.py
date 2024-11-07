import torch

torch.multiprocessing.set_sharing_strategy("file_system")

from typing import List, Optional, Tuple

import hydra
import lightning as pl
from loguru import logger
from omegaconf import DictConfig
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

from voxdet import hydra_logging_utils

log = hydra_logging_utils.get_pylogger(__name__)

try:
    from clearml import Task
except:  # noqa
    logger.warning(
        "ClearML is not installed using tensorboard logger. `pip install clearml` to install clearML."
    )
    Task = None


def is_model_in_eval_mode(model):
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm2d)):
            if module.training:
                return False
    return True


@hydra_logging_utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    clearml_task_id = None
    if Task and not cfg.get("skip_clearml", True):
        task = Task.init(
            project_name=cfg.clearml_project, task_name=cfg.clearml_task_name
        )
        task.connect_configuration(name="cfg", configuration=dict(cfg))
        task.add_tags(cfg.get("tags", ["no_tag"]))
        clearml_task_id = task.task_id  # need it later to get the logger from callbacks

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data.dataloader._target_}>")
    data: LightningDataModule = hydra.utils.instantiate(cfg.data.dataloader)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # log.info(model)

    log.info(f"Instantiating ckpt_path <{cfg.ckpt_path}>")
    ckpt_path = cfg.ckpt_path

    # Adding this in-case if it is useful. We are saving config inside checkpoint helps us to be consistent.
    model.cfg = cfg

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = hydra_logging_utils.instantiate_callbacks(
        cfg.get("callbacks")
    )

    log.info("Instantiating loggers...")
    logger: List[Logger] = hydra_logging_utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "data": data,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "clearml_task_id": clearml_task_id,
    }

    if logger:
        log.info("Logging hyperparameters!")
        hydra_logging_utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train", False):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            train_dataloaders=data.train_dl(),
            val_dataloaders=data.val_dl(),
            ckpt_path=ckpt_path,
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test", False):
        log.info("Starting testing!")
        ckpt_path = (
            trainer.checkpoint_callback.best_model_path
            if cfg.get("train")
            else cfg.get("ckpt_path")
        )
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.2",
    config_path="/home/users/shubham.kumar/projects/qct_nodule_detection/hydra_configs",
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = hydra_logging_utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
