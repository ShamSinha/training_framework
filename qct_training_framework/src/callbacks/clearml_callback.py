from clearml import Task
from loguru import logger
from lightning import Callback, Trainer


class ClearMLCallBack(Callback):
    def _setup_clearml_task(self, trainer: Trainer):
        """ClearML does not directly integrate with lightning.

        We have to get the logger object using the task id. We are saving the taskid as hyperparams
        in trainer.
        """
        task_id = trainer.logger.hparams["clearml_task_id"]
        if task_id:
            task = Task.get_task(task_id=task_id)
            self.logger = task.get_logger()
        else:
            logger.warning(
                "`clearml_task_id` is not logged to logger hparams. Did you set `skip_clearml` to `False`?"
            )
