from lightning.pytorch.callbacks import Callback
import torch

class WeightedLossCallback(Callback):
    def __init__(self, window_size=100, loss_type="auto"):

        self.window_size = window_size
        self.loss_type = loss_type
        self.loss_tuples = []

    def on_train_start(self, trainer, pl_module):
        # Initialize or reset the loss history at the start of training
        self.loss_tuples = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.loss_type is None:
            # If loss_type is None, no operation is required
            return        
        # Extract the regression and classification losses from the outputs
        # Make sure your training_step returns these values in the outputs dictionary
        reg_loss = outputs["metrics"]["reg_loss"].item()
        cls_loss = outputs["metrics"]["cls_loss"].item()

        # Update the loss history with the latest losses
        self.loss_tuples.append((reg_loss, cls_loss))
        # Keep only the last `window_size` entries
        self.loss_tuples = self.loss_tuples[-self.window_size:]

        # Compute the weighted loss ratio
        ratio = self.compute_weighted_loss_ratio()

        # Store the ratio in the pl_module (model) for use in the next training_step
        setattr(pl_module, "weighted_loss_ratio", ratio)

    def compute_weighted_loss_ratio(self):
        if not self.loss_tuples or self.loss_type is None:
            return torch.tensor(1.0)

        # Convert the list of tuples to a tensor for easier processing
        losses = torch.tensor(self.loss_tuples)
        reg_losses = losses[:, 0]
        cls_losses = losses[:, 1]
        
        reg_median, cls_median = reg_losses.median(), cls_losses.median()
        if reg_median >= cls_median:
            return torch.max(torch.tensor([reg_median / cls_median, 1 / 10]))
        else:
            return torch.min(torch.tensor([reg_median / cls_median, 10]))

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset or initialize the weighted loss ratio at the start of each epoch if needed
        setattr(pl_module, "weighted_loss_ratio", torch.tensor(1.0))
