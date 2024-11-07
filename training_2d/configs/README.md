

## Training Dataclass Documentation
### Model

Configuration of the model architecture.

- `__target__`: The dynamic path of the model class to be loaded , ex: cxr_training.nnmodule.nnmodule_controller.LitModel
- `encoder`: Encoder architecture, for available options refer to SMP documentation.
- `decoder`: Decoder architecture, for available options refer to SMP documentation.
- `in_channels`: Number of input channels (1 for grayscale, 3 for color).
- `out_channels`: Number of output channels (typically 2 for binary classification in multi-label setups).
- `optimizer`: Optimizer configuration, including the optimizer type and settings.
- `scheduler`: Scheduler configuration, including the scheduler type and settings.

### Scheduler_config

- `target`: Name of the scheduelr to use, limited to options provided in the PyTorch documentation.
- `params` :
    Configuration for the learning rate scheduler.
    For each of them it has to been configured according to how it is there in the pytorch scheduler since we are calling the Dynamically imports a class from a string and passing the params, look at get_class_from_str function from cxr_training.nnmodule.models.utils
    For each of them it has been configured according to how it is there in the pytorch schedulers , for example for OneCycleLR you can refer to https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html

### Optimizer_config

Settings for the optimizer.

- `target`: Name of the optimizer to use, limited to options provided in the PyTorch documentation. , ex: if ur using adamw then it would be torch.optim.AdamW
- `params` :
    Configuration for the optimizers.
    For each of them it has to been configured according to how it is there in the pytorch optimizer since we are calling the Dynamically imports a class from a string and passing the params, look at get_class_from_str function from cxr_training.nnmodule.models.utils
    example for AdamW you can refer to https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

### Details

General training details.

- `description`: Project description for naming saved model weight folders.
- `project`: Project name for storing logs in ClearML.
- `model_file`: Model type used for training, leave empty if not applicable.
- `check_gradients`: To see the behavior of the grad for all the parameters , if ur not able to strategy==ddp , u need to use this to check if grad is True but not None then the issue arrises.
- `recipe_module`: The full import path for the recipe , ex: cxr_training.recipes.base_recipe.BaseRecipe
-  `recipe`: The recipe parameter follows strict naming conventions: use cls, cls_seg, or cls_seg_age for classification tasks in that order, and seg or seg_age for segmentation tasks. Do not do other combinations or orderings for consistency since there may be cases the code behaviour would not be the same
- `total_epochs`: Total number of epochs to run the model.
- `accumulate_grad_batches`: Number of small batches to accumulate gradients over before performing a backward pass.
- `strategy`: Strategy for distributed training (e.g., DDP).
- `batch_size`: Number of samples processed at one time.
- `precision`: Specifies the floating-point precision ex, 16, 32 used in the PyTorch Lightning Trainer for model training and inference.16 works good in most cases.
- `num_workers`: Number of processes generating batches in parallel.
- `train_samples`: Number of training samples for the weighted random sampler.
- `validation_samples`: Number of validation samples for the weighted random sampler.
- `gpus`: Number of GPUs to use for the model, give the list of gpu ids to use, ex:[0,1,2]
- `fast_dev_run`: Enable to test one epoch in the Lightning trainer.

### Files

Paths to various required files.

- `ground_truth_csv`: Path to the ground truth CSV file.
- `img_folder_path`: Path to the image folder.
- `annotation_path`: Path containing the annotations.

### Classification

Settings specific to classification tasks.

- `sampling_tags`: Tags to sample for the model.
- `heads`: Classes of concern for the model.
- `user_class_wts`: Class weights for the weighted random sampler.
- `loss_wts`: Class weights for the cross-entropy loss function.
- `alpha`: Multiplier for the loss value.

### Segmentation

Settings specific to segmentation tasks.

- `sampling_tags`: Tags to sample for segmentation.
- `heads`: Classes of concern for the model.
- `dice_threshold`: [0.2] , use for dicebse loss , the importance of dice loss : Dice_BCE = BCE + dice_loss * dice_threshold
- `user_class_wts`: Class weights for the weighted random sampler.
- `loss_wts`: Class weights for the cross-entropy loss function.
- `alpha`: Multiplier for the loss value, adjusting weight given to segmentation.

### Path

Checkpoint paths.

- `checkpoint_dir`: Folder containing paths to checkpoint files.
- `checkpoint_path`: Specific path to checkpoint file to use for continuing training.
- `log_directory` : Folder containing paths to logs

### Common_params

Common parameters across models.

- `data_loader`: Dynamically path of the dataloader , ex: cxr_training.data.dataloader.base_dataloader.DataModule
- `dataset_type`: Dynamically path of the dataset_type , ex: cxr_training.data.datasets.cls_seg_dataset.Base_dataset
- `metric_type`: Type of metric function needed.
- `im_size`: Resized value for images during training and validation.
- `age_alpha`: The alpha for the rmse age loss
- `sources`: Sources to be used for data.
- `loss_type`: Type of loss function (default to cross-entropy).
- `equal_source_sampling`: Whether sampling should respect different sources equally.
- `mask_threshold`: Threshold for converting mask values for dataset loaders.
- `sampler`: Sampler type , it can either with respect to train or val , for ex : (train_random_val_weighted).

### Trainer_config

Main configuration for the trainer.

- `trainer`: Contains parameters required for the trainer arguments.
- `model`: Model-specific parameters.
- `path`: CSV and image folder paths.
- `files`: Contains checkpoint paths and directories.
- `cls`: Classification task parameters.
- `seg`: Segmentation task parameters.
- `params`: Parameters affecting the entire model pipeline.
- `use_clearml`: Toggle for using ClearML for experiment tracking.
- `validate config`: Toggle for pydantic validation , if u want u can write ur own config validation code


## Testing Dataclass Documentation

This documentation provides detailed information on the `testing` dataclass structure used for model configuration, including file paths, training details, testing configurations, and common parameters for model inference.

### 1. Files

Holds paths to various input files needed for the model.

- `ground_truth_csv` (`str`): Path to the ground truth CSV file.
- `img_folder_path` (`str`): Path to the image folder.
- `annotation_path` (`str`): Path containing the annotations.

### 2. Details

Configuration details for model training and inference.

- `description` (`str`): Project description, used in naming folders for saved model weights.
- `project` (`str`): Project name for storing logs in ClearML.
- `model_file` (`str`): Specifies the type of model used for training. Leave as an empty string if not needed.
- `batch_size` (`int`): Number of samples processed at one time.
- `num_workers` (`int`): Number of processes generating batches in parallel.
- `gpus` (`List`): Specifies the GPUs to use for testing. Use `[-1]` for all GPUs.
- `checkpoint_dir` (`str`): Directory where checkpoints are saved.
- `model_list` (`List`): Specific checkpoints to load, or an empty list to load all.
- `recipe` (`str`): Available recipes.

### 3. Test_Files

Paths to files used for testing.

- `testing_csv` (`str`): Path to the testing data CSV file.
- `testing_images` (`str`): Path to testing images.
- `testing_annotation_path` (`str`): Path containing testing annotations.

### 4. Common_params

Common parameters for model training and inference.

- `im_size` (`int`): Resized value for images during training and validation.
- `inference_type` (`str`): Available types for inference (`cls`, `seg`, `cls_seg`).
- `metric_type` (`str`): Available types for metrics.
- `inference_run` (`bool`): Enable inference run.
- `f1_beta_value` (`int`): Beta parameter for F1 score calculation.
- `metric_run` (`bool`): Enable metric calculation.
- `inference_on_subset` (`bool`): Decide whether to use a subset or the entire test dataset.
- `inference_on_images` (`bool`): Enable inference on images.

### 5. Classification

Configuration for classification tasks.

- `heads` (`List`): Classes of concern for the model.

### 6. Segmentation

Configuration for segmentation tasks.

- `heads` (`List`): Classes of concern for the model.

### 7. Testing_config

Testing configuration that combines other configurations.

- `trainer` (`Details`): Training details.
- `cls` (`Classification`): Classification task parameters.
- `seg` (`Segmentation`): Segmentation task parameters.
- `files` (`Test_Files`): Checkpoint paths and directories.
- `params` (`Common_params`): Model-wide parameters.
- `use_clearml` (`bool`): Toggle for using ClearML.
