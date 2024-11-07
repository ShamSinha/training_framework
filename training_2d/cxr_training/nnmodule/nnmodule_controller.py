import torch
from pytorch_lightning import LightningModule
from cxr_training.losses.losses_controller import losses_controller
from cxr_training.nnmodule.models.model_loader import get_model, get_main_model_string
from cxr_training.metrics.metric_controller import metrics_controller as MC
from cxr_training.nnmodule.models.utils import (
    get_preds_from_logits,
    filter_negative_targets,
    get_class_from_str,
)
from torchmetrics.aggregation import MeanMetric
from torchmetrics import MetricCollection
from pytorch_lightning.core.saving import save_hparams_to_yaml
import math
import omegaconf
import gc


class LitModel(LightningModule):
    """
    LitModel is a PyTorch Lightning model for training and evaluating a
    neural network for both classification and segmentation tasks.

    Attributes:
        args: Configuration arguments.
        ... (other attributes) which are explained the class files (look at configs/class_files)


    Note: The 'args' attribute is expected to have a specific structure,
    supplying values for the other attributes and determining the model's behavior.
    """

    def __init__(self, args):
        super().__init__()

        # Configuration setup
        self.args = args
        self._setup_from_args(args)
        self.model_str = self._get_model_str()
        self._print_initial_config()

        # Model and metrics configuration
        self.configure_metrics()
        self.configure_model()
        self.set_criterion()

    def _setup_from_args(self, args):
        """Setup class attributes based on configuration arguments."""
        # print("----------------args-------------", args)
        self.batch_size = args.trainer.batch_size
        self.checkpoint_path = self.args.path.checkpoint_path
        self.loaded_epoch = -1
        self.log_directory = args.path.log_directory
        self.description = args.trainer.description
        self.model_folder = args.trainer.model_file
        self.recipe = args.trainer.recipe
        self.gpus = args.trainer.gpus
        self.check_gradients = args.trainer.check_gradients
        self.num_samples = args.trainer.train_samples + args.trainer.validation_samples
        self.cls_heads = args.cls.heads
        self.seg_heads = args.seg.heads
        self.accumulate_grad_batches = args.trainer.accumulate_grad_batches

        self.metric_type = args.params.metric_type

        self.optim_class = get_class_from_str(args.model.optimizer._target_)
        self.optim_params = args.model.optimizer.params
        self.schd_params = omegaconf.OmegaConf.to_container(args.model.scheduler.params)
        self.optimizer_lr = args.model.optimizer.params.lr
        self.schd_type = args.model.scheduler._target_
        print("self.checkpoint_path : " , self.checkpoint_path)
        print(
            f"the optimizer used is {self.optim_class} and its parameters are {args.model.optimizer.params}"
        )
        self.schd_class = get_class_from_str(self.schd_type)
        print(
            f"the scheduler used is {self.schd_class} and its parameters are {args.model.scheduler.params}"
        )
        print("ckpt_path:" ,self.checkpoint_path)
        if self.checkpoint_path != "":
            print("ckpt_path:" ,self.checkpoint_path)
            loaded_checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            # Retrieve the epoch information
            self.loaded_epoch = loaded_checkpoint["epoch"]
            del loaded_checkpoint
            gc.collect()

    def _print_initial_config(self):
        """Print the initial configuration for debugging and verification."""
        global_batch_size = (
            torch.cuda.device_count() * self.batch_size * self.accumulate_grad_batches
            if self.gpus == [-1]
            else len(self.gpus) * self.batch_size * self.accumulate_grad_batches
        )
        # self.max_lr = math.sqrt(((global_batch_size) / 1024)) * self.max_lr
        print(
            f"total batch size used is {global_batch_size} and max learning rate with batch size is {self.optimizer_lr}"
        )

    def set_criterion(self):
        """Setup the loss criterion."""
        self.criterion = losses_controller(self.args)

    def configure_optimizers(self):
        """
            param_groups_list is a list of dictionaries containing the parameters and their respective optimizer parameters
            example:
            param_groups_list:
            - contain_key: ['encoder']
                params:
                lr: 0.01
                weight_decay: 0.0001
            - contain_key: ['decoder']
                params:
                lr: 0.1
                weight_decay: 0.0001
        """
        param_groups_list = getattr(self.args.model.optimizer, "param_groups_list", None)
        if param_groups_list is not None:
            ## convert the param_groups_list to the format required by the optimizer, also put remaining parameters in the last group (last group uses default optimizer parameters)
            model_key_names = [n for n,_ in self.model.named_parameters()]
            param_groups_list = [dict(p) for p in param_groups_list]
            print(param_groups_list)
            param_group_list_unrolled = [{"keys": [n for n in model_key_names if p_group["contain_key"] in n], "params": p_group["params"]} for p_group in param_groups_list]
            
            all_keys = set(sum([p_group["keys"] for p_group in param_group_list_unrolled], []))
            remaining_params = {"keys": [n for n in model_key_names if n not in all_keys], "params": {}}
            if len(remaining_params["keys"]) > 0:
                param_group_list_unrolled.append(remaining_params)
            param_groups = [{'params': [par for _,par in filter(lambda p : p[1].requires_grad and p[0] in p_group["keys"], self.model.named_parameters())], **p_group["params"]} for p_group in param_group_list_unrolled]
            self.schd_params["max_lr"] = list([p_group.get("lr", self.optim_params.get('lr', 1e-3)) for p_group in param_groups])
            ## print only for rank 0
            if self.trainer.global_rank == 0:
                print("PARAM_GROUPS".center(50, "-"))
                print("Total number of parameters", len(model_key_names))
                print("param_groups",[(len(p_group["params"]), {k:v for k,v in p_group.items() if k!="params"}) for p_group in param_groups])
                print("scheduler_lr",self.schd_params["max_lr"], type(self.schd_params["max_lr"]))
                print(self.schd_params)
                print("-"*50)
        else:
            param_groups = filter(lambda p: p.requires_grad, self.model.parameters())
        print("total number of steps", self.trainer.estimated_stepping_batches)
        self.opt = self.optim_class(
            # filter(lambda p: p.requires_grad, self.model.parameters()),
            param_groups,
            **self.optim_params,
        )

        if "OneCycleLR" in self.schd_type:
            self.sched = self.schd_class(
                self.opt,
                total_steps=self.trainer.estimated_stepping_batches,
                **self.schd_params,
            )
        else:
            self.sched = self.schd_class(
                self.opt,
                **self.schd_params,
            )

        self.sched = {
            "scheduler": self.sched,
            "interval": "step",
            "frequency": 1,
        }

        return [self.opt], [self.sched]

    def configure_model(self):
        """Configure the neural network model."""
        self.model = get_model(self.args)
        if getattr(self.args.model, "load_prt", False):
            st_dct = torch.load(self.args.model.pretrained_ckpt)["state_dict"]
            # model_ckpt_sd = st_dct
            model_ckpt_sd = {}
            for k, v in st_dct.items():
                model_ckpt_sd[k[6:]] = v
            filter_list = getattr(self.args.model, "filter_list", [])
            if len(filter_list) > 0:
                def filter_check(x,fl):
                    for i in fl:
                        if i in x:
                            return True
                    return False
                model_ckpt_sd = {k: v for k, v in model_ckpt_sd.items() if filter_check(k,filter_list)}
            # print("Loading encoder-decoder weights")
            print(f"Found keys: {len(model_ckpt_sd.keys())}", list(model_ckpt_sd.keys())[:3], list(model_ckpt_sd.keys())[-3:])
            missing_keys, unexpected_keys = self.model.load_state_dict(model_ckpt_sd, strict=getattr(self.args.model, "load_strict", True))
            if len(missing_keys) > 0:
                print(f"Missing keys: {len(missing_keys)}", missing_keys[:3], missing_keys[-3:])
            else:
                print("No missing keys found")
            if len(unexpected_keys) > 0:
                print(f"Unexpected keys: {len(unexpected_keys)}", unexpected_keys[:3], unexpected_keys[-3:])
            else:
                print("No unexpected keys found")
            if getattr(self.args.model, "freeze_old_weights", False):
                for name, param in self.model.named_parameters():
                    if name not in missing_keys:
                        param.requires_grad = False
        assert not (getattr(self.args.model, "freeze_encoder", False) and getattr(self.args.model, "unfreeze_encoder", False)), "Both freeze and unfreeze encoder flags are set"
        assert not (getattr(self.args.model, "freeze_decoder", False) and getattr(self.args.model, "unfreeze_decoder", False)), "Both freeze and unfreeze decoder flags are set"
        if getattr(self.args.model, "freeze_encoder", False):
            for param in self.model.main_arch.encoder.parameters():
                param.requires_grad = False
        if getattr(self.args.model, "freeze_decoder", False):
            for param in self.model.main_arch.decoder.parameters():
                param.requires_grad = False
        if getattr(self.args.model, "unfreeze_encoder", False):
            for param in self.model.main_arch.encoder.parameters():
                param.requires_grad = True
        if getattr(self.args.model, "unfreeze_decoder", False):
            for param in self.model.main_arch.decoder.parameters():
                param.requires_grad = True

    def configure_metrics(self):
        """Configure metrics for training and validation."""
        self.metrics = torch.nn.ModuleDict()
        for mode in ["train", "val"]:
            self._add_mode_metrics(mode)

        self.metrics["val_loss"] = MetricCollection(
            {"loss": MeanMetric()}, prefix="val_"
        )

        self.metrics["train_loss"] = MetricCollection(
            {"loss": MeanMetric()}, prefix="train_"
        )

    def _add_mode_metrics(self, mode):
        """Add metrics for a specific mode (train/val)."""
        if "cls" in self.recipe:
            for tag in self.cls_heads:
                self.metrics[f"{mode}_{tag}_cls_metrics"] = MC(
                    "cls", f"{mode}_{tag}_default_", self.metric_type
                )
                # self.metrics[f"{mode}_{tag}_cls_real_metrics"] = MC(
                #     "cls", f"{mode}_{tag}_real_", self.metric_type
                # )
                # self.metrics[f"{mode}_{tag}_cls_fake_metrics"] = MC(
                #     "cls", f"{mode}_{tag}_fake_", self.metric_type
                # )

        if "seg" in self.recipe:
            for tag in self.seg_heads:
                self.metrics[f"{mode}_{tag}_seg_metrics"] = MC(
                    "seg", f"{mode}_{tag}_default_", self.metric_type
                )

        if "age" in self.recipe:
            self.metrics["train_age_metrics"] = MC(
                "age", f"{mode}_default_", self.metric_type
            )
            self.metrics["val_age_metrics"] = MC(
                "age", f"{mode}_default_", self.metric_type
            )

    def forward(self, input_image):
        """input_image to be passed in func to get inference results or to just sanity checking."""
        logits = self.model(input_image)
        return logits

    def on_after_backward(self) -> None:
        if self.check_gradients:
            print("Gradients:")
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is None:
                        print(f"{name}: no gradient but is trainable")
                else:
                    print(f"{name}: not trainable")
        else:
            return super().on_after_backward()

    # partition input recursively based on boolean tensor, input can be a tensor, list of tensor
    # or a dict
    def _partition_input(self, input, real_label):
        if isinstance(input, torch.Tensor):
            return input[real_label]
        elif isinstance(input, list):
            return [self._partition_input(i, real_label) for i in input]
        elif isinstance(input, dict):
            return {k: self._partition_input(v, real_label) for k, v in input.items()}
        else:
            raise ValueError("input should be a tensor, list of tensor or a dict")  


    def training_step(self, batch, batch_idx):
        indices, input, targets, real_label = map(batch.get, ("idx", "input", "target", "real_label"))
        logits = self.model(input)

        real_label = real_label.bool()
        
        real_fake_partition_dict = {}
        if True in real_label:
            real_fake_partition_dict["target_real"] = self._partition_input(targets, real_label)
            real_fake_partition_dict["logits_real"] = self._partition_input(logits, real_label)
            real_loss_dict = self.criterion(real_fake_partition_dict["logits_real"], real_fake_partition_dict["target_real"])
        if False in real_label:
            real_fake_partition_dict["target_fake"] = self._partition_input(targets, ~real_label)
            real_fake_partition_dict["logits_fake"] = self._partition_input(logits, ~real_label)
            fake_loss_dict = self.criterion(real_fake_partition_dict["logits_fake"], real_fake_partition_dict["target_fake"])

        loss_dict = self.criterion(logits, targets)
        real_label = real_label.bool()

        loss = sum(value for key, value in loss_dict.items() if 'bce' not in key and 'dice' not in key)
        self.metrics["train_loss"].update(loss)
        # self.log_metric("train_loss", loss)

        
        if "seg" in self.recipe:
            loss_seg = sum(value for key, value in loss_dict.items() if 'seg' in key)
            self.log_metric("train_seg_loss", loss_seg)
            for tag in self.seg_heads:
                self.log_metric(f"train_{tag}_seg_loss", loss_dict[f"{tag}_seg"])
                self.log_metric(f"train_{tag}_bce_loss", loss_dict[f"{tag}_bce"])
                self.log_metric(f"train_{tag}_dice_loss", loss_dict[f"{tag}_dice"])
                if True in real_label:
                    self.log_metric(f"train_{tag}_seg_loss_real", real_loss_dict[f"{tag}_seg"])
                    # self.log_metric(f"train_{tag}_bce_loss_real", real_loss_dict[f"{tag}_bce"])
                    # self.log_metric(f"train_{tag}_dice_loss_real", real_loss_dict[f"{tag}_dice"])
                if False in real_label:
                    self.log_metric(f"train_{tag}_seg_loss_fake", fake_loss_dict[f"{tag}_seg"])
                    # self.log_metric(f"train_{tag}_bce_loss_fake", fake_loss_dict[f"{tag}_bce"])
                    # self.log_metric(f"train_{tag}_dice_loss_fake", fake_loss_dict[f"{tag}_dice"])
                

        if "cls" in self.recipe:
            loss_cls = sum(value for key, value in loss_dict.items() if 'cls' in key)
            self.log_metric("train_cls_loss", loss_cls)
            for tag in self.cls_heads:
                self.log_metric(f"train_{tag}_cls_loss", loss_dict[f"{tag}_cls"])
                if True in real_label:
                    self.log_metric(f"train_{tag}_cls_loss_real", real_loss_dict[f"{tag}_cls"])
                if False in real_label:
                    self.log_metric(f"train_{tag}_cls_loss_fake", fake_loss_dict[f"{tag}_cls"])


        if "age" in self.recipe:
            self.log_metric("train_age_rmse", math.sqrt(loss_dict["age_mse_loss"]))

        preds = get_preds_from_logits(logits, self.args)

        outputs = {
            "preds": preds,
            "target": targets,
        }

        self.training_log_metric(outputs)

        return loss

    def test_step(self, batch, batch_idx):

        indices, input, targets = map(batch.get, ("idx", "input", "target"))
        logits = self.model(input)

        loss_dict = self.criterion(logits, targets)
        loss = sum(value for key, value in loss_dict.items() if 'bce' not in key and 'dice' not in key)


        self.metrics["val_loss"].update(loss)

        if "seg" in self.recipe:
            for tag in self.seg_heads:
                self.log_metric(f"test_{tag}_seg_loss", loss_dict[f"{tag}_seg"])
                self.log_metric(f"test_{tag}_bce_loss", loss_dict[f"{tag}_bce"])
                self.log_metric(f"test_{tag}_dice_loss", loss_dict[f"{tag}_dice"])

        if "cls" in self.recipe:
            for tag in self.cls_heads:
                self.log_metric(f"test_{tag}_cls_loss", loss_dict[f"{tag}_cls"])

        if "age" in self.recipe:
            self.log_metric("test_age_rmse", math.sqrt(loss_dict["age_mse_loss"]))

        preds = get_preds_from_logits(logits, self.args)

        outputs = {
            "preds": preds,
            "target": targets,
        }

        self.validation_log_metric(outputs)
    
    def load_dummy_chkpoint_values(self, preds, targets):
        if "cls" in self.recipe:
            device_preds = preds["classification_out"][self.cls_heads[0]].device
            device_target = targets["classification_target"][self.cls_heads[0]].device
            for tag in self.cls_heads:
                preds["classification_out"][tag] = torch.cat(
                    [
                        preds["classification_out"][tag],
                        torch.tensor([0]).to(device_preds),
                    ],
                    dim=0,
                )
                targets["classification_target"][tag] = torch.cat(
                    [
                        targets["classification_target"][tag],
                        torch.tensor([0]).to(device_target),
                    ],
                    dim=0,
                )
        return preds, targets

    def validation_step(self, batch, batch_idx):

        indices, input, targets, real_label = map(batch.get, ("idx", "input", "target", "real_label"))
        logits = self.model(input)

        real_label = real_label.bool()
        
        real_fake_partition_dict = {}
        if True in real_label:
            real_fake_partition_dict["target_real"] = self._partition_input(targets, real_label)
            real_fake_partition_dict["logits_real"] = self._partition_input(logits, real_label)
            real_loss_dict = self.criterion(real_fake_partition_dict["logits_real"], real_fake_partition_dict["target_real"])
        if False in real_label:
            real_fake_partition_dict["target_fake"] = self._partition_input(targets, ~real_label)
            real_fake_partition_dict["logits_fake"] = self._partition_input(logits, ~real_label)
            fake_loss_dict = self.criterion(real_fake_partition_dict["logits_fake"], real_fake_partition_dict["target_fake"])

        loss_dict = self.criterion(logits, targets)
        real_label = real_label.bool()


        loss_dict = self.criterion(logits, targets)
        loss = sum(value for key, value in loss_dict.items() if 'bce' not in key and 'dice' not in key)
        # self.log_metric("val_loss", loss)


        self.metrics["val_loss"].update(loss)

        if "seg" in self.recipe:
            loss_seg = sum(value for key, value in loss_dict.items() if 'seg' in key)
            self.log_metric("val_seg_loss", loss_seg)
            for tag in self.seg_heads:
                self.log_metric(f"val_{tag}_seg_loss", loss_dict[f"{tag}_seg"])
                self.log_metric(f"val_{tag}_bce_loss", loss_dict[f"{tag}_bce"])
                self.log_metric(f"val_{tag}_dice_loss", loss_dict[f"{tag}_dice"])
                if True in real_label:
                    self.log_metric(f"val_{tag}_seg_loss_real", real_loss_dict[f"{tag}_seg"])
                    # self.log_metric(f"val_{tag}_bce_loss_real", real_loss_dict[f"{tag}_bce"])
                    # self.log_metric(f"val_{tag}_dice_loss_real", real_loss_dict[f"{tag}_dice"])
                if False in real_label:
                    self.log_metric(f"val_{tag}_seg_loss_fake", fake_loss_dict[f"{tag}_seg"])
                    # self.log_metric(f"val_{tag}_bce_loss_fake", fake_loss_dict[f"{tag}_bce"])
                    # self.log_metric(f"val_{tag}_dice_loss_fake", fake_loss_dict[f"{tag}_dice"])

        if "cls" in self.recipe:
            loss_cls = sum(value for key, value in loss_dict.items() if 'cls' in key)
            self.log_metric("val_cls_loss", loss_cls)
            for tag in self.cls_heads:
                self.log_metric(f"val_{tag}_cls_loss", loss_dict[f"{tag}_cls"])
                if True in real_label:
                    self.log_metric(f"val_{tag}_cls_loss_real", real_loss_dict[f"{tag}_cls"])
                if False in real_label:
                    self.log_metric(f"val_{tag}_cls_loss_fake", fake_loss_dict[f"{tag}_cls"])

        if "age" in self.recipe:
            self.log_metric("val_age_rmse", math.sqrt(loss_dict["age_mse_loss"]))

        preds = get_preds_from_logits(logits, self.args)

        # preds_real, preds_fake, targets_real, targets_fake = None, None, real_fake_partition_dict.get("target_real", None), real_fake_partition_dict.get("target_fake", None)
        # if True in real_label:
        #     preds_real = get_preds_from_logits(real_fake_partition_dict["logits_real"], self.args)
        # if False in real_label:
        #     preds_fake = get_preds_from_logits(real_fake_partition_dict["logits_fake"], self.args)

        if self.loaded_epoch == self.current_epoch:
            # cases an issue otherwise with loading checkpoints
            preds, targets = self.load_dummy_chkpoint_values(preds, targets)
            # if preds_real is not None:
            #     preds_real, targets_real = self.load_dummy_chkpoint_values(preds_real, real_fake_partition_dict["target_real"])
            # if preds_fake is not None:
            #     preds_fake, targets_fake = self.load_dummy_chkpoint_values(preds_fake, real_fake_partition_dict["target_fake"])

        outputs = {
            "preds": preds,
            "target": targets,
            # "preds_real": preds_real,
            # "target_real": targets_real,
            # "preds_fake": preds_fake,
            # "target_fake": targets_fake
        }

        self.validation_log_metric(outputs)

    def extract_callback_metrics_items(self, mode, tag):
        """
        Extract specific metric items from `self.combined_dicts` based on the provided mode and tag.

        Args:
            mode (str): The mode to filter on, e.g., "train" or "val".
            tag (str): The metric tag to filter on, e.g., "default_BinaryAUROC".

        Returns:
            dict: A dictionary containing the metrics that match the provided mode and tag.

        Example:
            Assuming `self.combined_dicts` is:
            {
                "train_default_BinaryAUROC": tensor(0.95),
                "val_default_BinaryAUROC": tensor(0.90),
                ...
            }
            >>> self.extract_callback_metrics_items("train", "default_BinaryAUROC")
            {
                "train_default_BinaryAUROC": 0.95
            }
        """

        return {
            k: v.item()
            for k, v in self.combined_dicts.items()
            if k.startswith(f"{mode}") and tag in k
        }

    def on_train_epoch_end(self) -> None:
        """
        At the end of each training epoch, logs specific metric values to the logger's experiment.

        This method:
        1. Retrieves the callback metrics from the trainer.
        2. Defines a list of metric tags to consider.
        3. Iterates over modes ("val", "train") and the defined metric tags to log aggregated metric values.
        4. Prints a separator for clarity in console outputs.

        Note:
            This function is typically used in conjunction with PyTorch Lightning's training loop
            and requires the class to have a logger and trainer attribute.
        """

        self.combined_dicts = self.trainer.callback_metrics
        tags_to_consider = [
            "default_BinaryAUROC",
            "default_BinarySpecificity",
            "default_BinaryRecall",
            "default_qIOU",
            "cls_loss",
            "seg_loss",
            "bce_loss",
            "dice_loss"
        ]

        for mode in ["val", "train"]:
            for tag in tags_to_consider:
                self.logger.experiment.add_scalars(
                    f"combined_{mode}_{tag}",
                    self.extract_callback_metrics_items(mode, tag),
                    global_step=self.current_epoch,
                )

        print("-" * 50)
        print("\n")

    def on_validation_epoch_end(self) -> None:
        """Print separator at the end of validation epoch."""
        print("-" * 50)
        print("\n")

    def on_test_epoch_end(self) -> None:
        """Print separator at the end of test epoch."""
        self.combined_dicts = self.trainer.callback_metrics
        tags_to_consider = [
            "default_BinaryAUROC",
            "default_BinarySpecificity",
            "default_BinaryRecall",
            "default_qIOU",
            "cls_loss",
            # "seg_loss",
            # "bce_loss",
            # "dice_loss"
        ]
        for tag in tags_to_consider:
                self.logger.experiment.add_scalars(
                    f"combined_test_{tag}",
                    self.extract_callback_metrics_items("test", tag),
                    global_step=self.current_epoch,
                )

        print("\n")



    # def training_log_metric(self, outputs):
        
    #     if "cls" in self.recipe:
    #         for tag in self.cls_heads:
    #             preds = outputs["preds"]["classification_out"][tag]
    #             targets = outputs["target"]["classification_target"][tag].long()
    #             preds, targets = filter_negative_targets(preds, targets)
    #             if len(targets) > 0:
    #                 self.metrics[f"train_{tag}_cls_metrics"](preds, targets)

    #     if "seg" in self.recipe:
    #         for tag in self.seg_heads:
    #             preds = outputs["preds"]["segmentation_out"][tag]
    #             targets = outputs["target"]["segmentation_target"][tag].long()
    #             if len(targets) > 0:
    #                 self.metrics[f"train_{tag}_seg_metrics"](preds, targets)

        
    #     if "age" in self.recipe:
    #         preds = outputs["preds"]["age_out"]
    #         targets = outputs["target"]["age_target"].long()
    #         mask = targets != -100
    #         preds = preds[mask]
    #         preds = preds.view(-1)  # since it comes like this [[0.], [0.], [0.], [0.]]
    #         targets = targets[mask].float()

    #     print("\n")



    # def training_log_metric(self, outputs):
    #     if "age" in self.recipe:
    #         preds = outputs["preds"]["age_out"]
    #         targets = outputs["target"]["age_target"].long()
    #         mask = targets != -100
    #         preds = preds[mask]
    #         preds = preds.view(-1)  # since it comes like this [[0.], [0.], [0.], [0.]]
    #         targets = targets[mask].float()

    #         if (
    #             len(targets) > 1
    #         ):  # ValueError: Needs at least two samples to calculate r2 score.
    #             self.metrics["train_age_metrics"](preds, targets)

    #     self.log_values(self.metrics, mode="train")

    def validation_log_metric(self, outputs):
        """Log metrics for the validation phase.

        Args:
            outputs (dict): Dictionary containing predictions and targets.
        """
        if "cls" in self.recipe:
            for tag in self.cls_heads:
                preds = outputs["preds"]["classification_out"][tag]
                targets = outputs["target"]["classification_target"][tag].long()
                preds, targets = filter_negative_targets(preds, targets)
                if len(targets) > 0:
                    self.metrics[f"val_{tag}_cls_metrics"](preds, targets)
                # if outputs["preds_real"] is not None:
                #     preds_real = outputs["preds_real"]["classification_out"][tag]
                #     targets_real = outputs["target_real"]["classification_target"][tag].long()
                #     preds_real, targets_real = filter_negative_targets(preds_real, targets_real)
                #     if len(targets_real) > 0:
                #         self.metrics[f"val_{tag}_cls_real_metrics"](preds_real, targets_real)
                # if outputs["preds_fake"] is not None:
                #     preds_fake = outputs["preds_fake"]["classification_out"][tag]
                #     targets_fake = outputs["target_fake"]["classification_target"][tag].long()
                #     preds_fake, targets_fake = filter_negative_targets(preds_fake, targets_fake)
                #     if len(targets_fake) > 0:
                #         self.metrics[f"val_{tag}_cls_fake_metrics"](preds_fake, targets_fake   

        if "seg" in self.recipe:
            for tag in self.seg_heads:
                preds = outputs["preds"]["segmentation_out"][tag]
                targets = outputs["target"]["segmentation_target"][tag].long()
                if len(targets) > 0:
                    self.metrics[f"val_{tag}_seg_metrics"](preds, targets)

        if "age" in self.recipe:
            preds = outputs["preds"]["age_out"]
            targets = outputs["target"]["age_target"].long()
            mask = targets != -100
            preds = preds.view(-1)
            preds = preds[mask]
            # since it comes like this [[0.], [0.], [0.], [0.]]
            targets = targets[mask].float()

            if (
                len(targets) > 1 & len(preds) > 1
            ):  # ValueError: Needs at least two samples to calculate r2 score.
                self.metrics["val_age_metrics"](preds, targets)
            else:
                # so that 0 gets added into the calculation
                self.metrics["val_age_metrics"](
                    torch.tensor([1.0, 1.0]), (torch.tensor([1.0, 0.0]))
                )

        self.log_values(self.metrics, mode="val")

    def training_log_metric(self, outputs):
        """Log metrics for the training phase.

        Args:
            outputs (dict): Dictionary containing predictions and targets.
        """
        if "cls" in self.recipe:
            for tag in self.cls_heads:
                preds = outputs["preds"]["classification_out"][tag]
                targets = outputs["target"]["classification_target"][tag].long()
                preds, targets = filter_negative_targets(preds, targets)
                if len(targets) > 0:
                    self.metrics[f"train_{tag}_cls_metrics"](preds, targets)

        if "seg" in self.recipe:
            for tag in self.seg_heads:
                preds = outputs["preds"]["segmentation_out"][tag]
                targets = outputs["target"]["segmentation_target"][tag].long()
                if len(targets) > 0:
                    self.metrics[f"train_{tag}_seg_metrics"](preds, targets)

        if "age" in self.recipe:
            preds = outputs["preds"]["age_out"]
            targets = outputs["target"]["age_target"].long()
            mask = targets != -100
            preds = preds.view(-1)
            preds = preds[mask]
            # since it comes like this [[0.], [0.], [0.], [0.]]
            targets = targets[mask].float()

            if (
                len(targets) > 1 & len(preds) > 1
            ):  # ValueError: Needs at least two samples to calculate r2 score.
                self.metrics["train_age_metrics"](preds, targets)
            else:
                # so that 0 gets added into the calculation
                self.metrics["val_age_metrics"](
                    torch.tensor([1.0, 1.0]), (torch.tensor([1.0, 0.0]))
                )

        self.log_values(self.metrics, mode="train")

    def log_metric(self, name, value):
        """
        sync_dist should be True , look at this issue https://github.com/Lightning-AI/pytorch-lightning/issues/8190
        for metrics its not needed , read here:
        https://pytorch-lightning.readthedocs.io/en/1.4.9/advanced/multi_gpu.html#synchronize-validation-and-test-logging
        """
        self.log(
            name=name,
            value=value,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )

    def log_values(self, input_dict: torch.nn.modules.container.ModuleDict, mode: str):
        """Takes ModuleDict consisting of cls and seg metrics and logs them.
        sync_dist parameter syncs metrics across GPUs and adds lot of computations
        overhead.Turn it to false in train to boost training speeds.
        Args:
            input_dict : ModuleDict of cls metrics.
            mode (str): mode can either be train, val or test. Based on mode
            filter is applied to input_dict to pick mode's metrics.
        """

        for model_out_type in input_dict.keys():
            if model_out_type.startswith(mode):
                self.log_dict(
                    input_dict[model_out_type],
                    prog_bar=True,
                    batch_size=self.batch_size,
                    on_step=False,
                    on_epoch=True,
                    rank_zero_only=True,
                )

    def _get_model_str(self):
        """Get the combined string representation of the main model and its loader."""
        multihead_str = get_main_model_string()
        get_model_path = get_model.__code__.co_filename
        with open(get_model_path, "r") as file:
            model_str = file.read()
        combined_model_str = multihead_str + "\n\n" + model_str
        return combined_model_str

    def on_save_checkpoint(self, checkpoint) -> None:
        """Actions to perform when saving a checkpoint."""
        args_string = omegaconf.OmegaConf.create(self.args)
        args_final = omegaconf.OmegaConf.to_container(args_string, resolve=True)
        checkpoint["args"] = args_final
        checkpoint["model_str"] = self.model_str
        save_hparams_to_yaml(
            config_yaml=f"{self.log_directory}/{self.description}/{self.model_folder}/hparams.yaml",
            hparams=self.args,
        )