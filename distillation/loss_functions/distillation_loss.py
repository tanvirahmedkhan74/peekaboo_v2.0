import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, student_output, teacher_output, labels):
        # Adjusted dimension for single-channel data
        kd_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=0),
            F.softmax(teacher_output / self.temperature, dim=0),
            reduction="batchmean"
        ) * (self.alpha * self.temperature ** 2)

        # Supervised BCE loss
        bce_loss = self.bce_loss(student_output, labels) * (1 - self.alpha)

        return kd_loss + bce_loss
