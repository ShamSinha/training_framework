XR Training Repo
===

## To know how to Config is structured , please.. read this

Please visit here for that [Link Text](configs/README.md)

## Updated Features

Updates
  - Sensitivity and Specificity at 0.5 metrics is added to get an estimate of the final youden index.
  - model is loaded like this __target__: cxr_training.nnmodule.nnmodule_controller.LitModel
  - recipe module is added and recipe controller is removed
  - data_loader: cxr_training.data.dataloader.base_dataloader.DataModule
  - dataset_type: cxr_training.data.datasets.cls_seg_dataset.Base_dataset
  - Note : Deeplabv3plus dosent work with swin
https://github.com/isaaccorley/torchseg/issues/54
  - This completed all the major updates from my side , if for any other updates do a pull request and be a active contributor , any change in config do make in the config read me too since it would be helpful and save time

## To set up this code
  ```
  #!/bin/bash
  conda create -n qtrain python=3.10
  eval "$(conda shell.bash hook)"
  conda activate qtrain
  if [ -d "/home/users/$USER/xr_training" ]; then
      # If it exists, change to the 'xr_training' directory
      cd /home/users/$USER/xr_training
  fi
  pip install -e .
  ```

  or u can run
  ```
  source setup.sh
  ```
## Use

  U can run model by running the bash script
  ```
  sh run.sh
  ```

- for train of basic models
  ```
  python3 main.py --config-name=training --config-path=configs/yaml_files/akshay
  ```

- for test of basic models
  ```
  python3 main.py --config-name=testing --config-path=configs/yaml_files/akshay
  ```

- To set up the same vscode settings ,please set up flake8 linter and black formatting in a .vscode file in the same repo
  ```
  {
    "python.formatting.provider": "black",
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": [
        "--max-line-length=120",
        "--ignore=E302,E305,F401,E203,W503,W291",
    ],
    "editor.formatOnSave": true,
  }
  ```

## Setting Up Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) hooks to ensure code quality and consistency with tools like `black` for code formatting and `flake8` for linting.

### Steps to follow

1) First, ensure you have the necessary tools installed. If you haven't already, install the development dependencies: from the requirements.txt
1) pre-commit install
1) pre-commit run --all-files To manually run all pre-commit hooks on all files
1) Now before u commit all the time the pre commit hooks are run to maintain consistency

# Bonus INFO
## Learning Rate Scheduler Configuration

When tuning the learning rate (LR) scheduler, it's crucial to accurately calculate the total number of steps, which is determined by multiplying the number of epochs by the steps per epoch. Incorrectly specifying this number can lead to improper scheduling of the learning rate adjustments.

### Steps Per Epoch Calculation

The formula for calculating the steps per epoch is as follows:

$$
\text{Steps per Epoch} = \frac{\text{Number of Training Samples}}{(\text{Number of GPUs} \times \text{Batch Size} \times \text{Gradient Accumulation Batches})}
$$

This calculation ensures that the learning rate scheduler is aligned with the actual number of optimizer steps taken during training.

### Example

Given the following training setup:

- **Number of GPUs**: 2
- **Training Samples**: 896,000
- **Batch Size**: 280
- **Gradient Accumulation Batches**: 16

The steps per epoch can be calculated as:
$$
\text{Steps per Epoch} = \frac{896,000}{(2 \times 280 \times 16)} = 100
$$

## Checkpoint loading
When loading a checkpoint make sure to load with the same batch size and the same number of gpus otherwise the behavior in the inital epoch is unpredictable


If your goal is to adjust the learning rate at every epoch instead of every step, you can change the `interval` parameter in your scheduler configuration from `"step"` to `"epoch"`. This adjustment ensures that the learning rate is updated at the end of each epoch, rather than after a fixed number of steps.

Here's how to adjust the scheduler interval in your training module:

```python
scheduler = {
    "scheduler": self.sched,
    "interval": "epoch",  # Changed from "step" to "epoch"
    "frequency": 1,
}
```

## Resources

To understand and how the model loading and debugging and the how the configs are arranged , u can go through this notebook.
[Model_debugging](https://github.com/qureai/xr_training/blob/master/notebooks/model_debugging.ipynb)

To understand and how to load the model during inference and understand its gradcam , u can go through this notebook.
[Model_inference_gradcam.](https://github.com/qureai/xr_training/blob/master/notebooks/model_inference_gradcam.ipynb)

## Points to Note

Any changes or improvments to the code are welcome and do note that any changes made have to be backward compatible and also after the changes are done it should also work with the [qxr_model repo](https://github.com/qureai/cxr_models) , this is the repo that produces the torchscript files that would be used during production , so if u are able to make changes here that break the model convertion to torchscript, you should be ready to also make sure to work on making the models ready during production too.If u are fine with them , do send a pull request.
