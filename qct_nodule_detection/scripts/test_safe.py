from typing import Any, Dict, List, Tuple
import hydra
from omegaconf import DictConfig
from voxdet.metrics.det_metrics import DetMetrics
from voxdet.metrics.sub_level_analysis import convert2df
from tqdm.auto import tqdm
from report import ReportGenerator
from lightning import LightningModule, LightningDataModule

from loguru import logger

def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
  

    logger.info(f"Instantiating datamodule <{cfg.data.dataloader._target_}>")
    data: LightningDataModule = hydra.utils.instantiate(cfg.data.dataloader)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    froc_thresholds = hydra.utils.instantiate(cfg.froc_thresholds)

    meters = [
        DetMetrics(iou_thr=j, conf_thr=i, froc_thresholds= froc_thresholds.tolist())
        for j in cfg.iou_thr
        for i in cfg.conf_thr
    ]        
    for img in tqdm(data.tds) :
        out = model(img)
        logger.debug(out["boxes"])
        logger.debug(out["scores"])
        logger.debug(img["boxes"])
        for meter in meters: meter.update(out["boxes"], out["scores"], img["boxes"])

    metrics = [i.compute() for i in meters]
    df = convert2df(metrics)

    df.to_csv(cfg.save_dir + cfg.filename, index=False)

    for meter in meters:
        rg = ReportGenerator(
            file_name=f"{cfg.save_dir}/concise_report_{meter.iou_thr}",
            title="Report",
            root=f"{cfg.save_dir}",
            dirs=[cfg.datasets],
            meters=[meter],
            subgroup_analysis=True,
            source_analysis=False,
        )
        rg.generate_report()


@hydra.main(version_base="1.3", config_path="/home/users/shubham.kumar/projects/qct_nodule_detection/hydra_configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    evaluate(cfg)


if __name__ == "__main__":
    main()