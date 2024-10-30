import torch
import torch.nn as nn

class TemperatureScaledCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing = 0.1, initial_temperature=1.0):
        super(TemperatureScaledCrossEntropyLoss, self).__init__()
        # Initialize temperature as a learnable parameter
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
        # self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing= label_smoothing)
        self.bceloss = nn.BCEWithLogitsLoss(reduction = "mean")
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # new_logits = torch.cat((-logits, logits), dim=1)
        # Scale the logits using the temperature parameter
        scaled_logits = logits / self.temperature
        smooth_targets = targets * (1 - self.label_smoothing) + self.label_smoothing

        # Compute the cross-entropy loss with scaled logits
        loss = self.bceloss(scaled_logits, smooth_targets)
        return loss
