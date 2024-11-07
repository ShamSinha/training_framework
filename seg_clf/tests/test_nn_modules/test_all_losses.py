
import hydra
import omegaconf
import pyrootutils

from loguru import logger

root = pyrootutils.setup_root(__file__, pythonpath=True)
def test_losses():
    all_loss_cfgs = (root / "configs" / "model" / "nn_modules" / "losses").rglob("*.yaml")
    for cfg_path in all_loss_cfgs:
        cfg= omegaconf.OmegaConf.load(cfg_path)
        if len(cfg) > 0:
            logger.debug(f"Instatiating on {cfg_path}")
            _ = hydra.utils.instantiate(cfg)
            logger.info(f"Success on {cfg_path}")
        else:
            logger.warning(f"Empty config for {cfg_path}")

if __name__ == "__main__":
    test_losses()
