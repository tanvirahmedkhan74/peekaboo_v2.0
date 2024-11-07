import torch.nn.functional as F
import torch.nn as nn
import torch

from dice_loss import DiceLoss
from focal_loss import FocalLoss

class EnhancedHybridDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0, dice_weight=0.3, focal_weight=0.2):
        super(EnhancedHybridDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, student_output, student_output_binary, teacher_output, ground_truth=None):
        # Knowledge distillation loss
        kd_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1),
            reduction="batchmean"
        ) * (self.alpha * self.temperature ** 2)

        total_loss = kd_loss
        if ground_truth is not None:
            supervised_loss = F.binary_cross_entropy_with_logits(student_output_binary, ground_truth)
            dice_loss = self.dice_loss(student_output_binary, ground_truth)
            focal_loss = self.focal_loss(student_output_binary, ground_truth)

            total_loss = (
                kd_loss * self.alpha
                + supervised_loss * (1 - self.alpha)
                + self.dice_weight * dice_loss
                + self.focal_weight * focal_loss
            )

        return total_loss