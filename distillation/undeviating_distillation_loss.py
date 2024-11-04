import torch.nn as nn
import torch.nn.functional as F

class UndeviatingDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super(UndeviatingDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        # Knowledge distillation loss
        kd_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=0),
            F.softmax(teacher_output / self.temperature, dim=0),
            reduction="batchmean"
        ) * (self.alpha * self.temperature ** 2)

        return kd_loss
