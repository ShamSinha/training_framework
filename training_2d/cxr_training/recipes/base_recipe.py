from pytorch_lightning import Trainer
from cxr_training.callbacks.callback_controller import configure_callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from cxr_training.nnmodule.models.utils import get_class_from_str
import torch


class BaseRecipe:
    """Class to define the model and the dataloaders and then send it directly to the Trainer module in pylightning"""

    def __init__(self, args):
        self._setup_from_args(args)
        self._configure_recipe()

    def _setup_from_args(self, args):
        """Initialize attributes from arguments."""
        self.args = args
        self.recipe = args.trainer.recipe
        self.gpus = args.trainer.gpus
        self.fast_dev_run = args.trainer.fast_dev_run
        self.checkpoint_dir = args.path.checkpoint_dir
        self.description = args.trainer.description
        self.model_folder = args.trainer.model_file
        self.checkpoint_path = args.path.checkpoint_path
        self.epochs = args.trainer.total_epochs
        self.strategy = args.trainer.strategy
        self.accumulate_grad_batches = args.trainer.accumulate_grad_batches
        self.precision = args.trainer.precision
        self.log_directory = args.path.log_directory
        self.model_loader = get_class_from_str(args.model.__target__)
        self.data_module = get_class_from_str(args.params.data_loader)

    def _configure_recipe(self):
        """Configure dataloaders, callbacks, model, and trainer."""

        self.setup_model()
        self.setup_dataloaders()
        self.setup_callbacks()
        self.setup_trainer()

    def setup_model(self):
        """Load pre-trained model if finetune is enabled, else set up a new model."""

        self.model = self.model_loader(self.args)

    def setup_dataloaders(self):
        self.dataloader = self.data_module(self.args)

    def setup_callbacks(self):
        self.callbacks = configure_callbacks(self.args)

    def setup_trainer(self):
        """Refer link: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags for trainer
        flags.

        ddp_find_unused_parameters_false if but did not find any unused parameters in the forward pass. This flag
        results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance.
        If your model indeed never has any unused parameters in the forward pass, consider turning this flag off.

        If you have any unused parameters in the model set stratefy as ddp itself , we can check if there are any
        unused parameters by seeing if the following code gives any output after loss_dict.backward() and before
        optimizer.step

        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)

        If you are sure you have set param.requires_grad = False in some of the model parameters then set strategy as
        ddp itself as then find_unused_parameters will be True by default
        """
        self.trainer = Trainer(
            enable_model_summary=True,
            num_sanity_val_steps=10,
            fast_dev_run=self.fast_dev_run,
            max_epochs=self.epochs,
            detect_anomaly=False,
            devices=-1 if self.gpus == [-1] else self.gpus,
            strategy=self.strategy,
            accelerator="auto",
            accumulate_grad_batches=self.accumulate_grad_batches,
            callbacks=self.callbacks,
            sync_batchnorm=True,
            benchmark=True,
            precision=self.precision,
            logger=TensorBoardLogger(
                save_dir=self.log_directory,
                name=self.description,
                version=self.model_folder,
            ),
            log_every_n_steps=1,
            check_val_every_n_epoch=4,
        )

    def run(self):
        if self.checkpoint_path != "":
            print("using checkpoint path")
            self.trainer.fit(
                self.model, self.dataloader, ckpt_path=self.checkpoint_path
            )
        else:
            self.trainer.fit(self.model, self.dataloader)
