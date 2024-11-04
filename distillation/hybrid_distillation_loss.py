import torch.nn.functional as F
import torch.nn as nn

class HybridDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0, beta=0.5):
        super(HybridDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.beta = beta  # weight for supervised loss with ground truth

    def forward(self, student_output, teacher_output, ground_truth=None):
        # Distillation loss
        kd_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1),
            reduction="batchmean"
        ) * (self.alpha * self.temperature ** 2)

        # Supervised loss with ground truth, if available
        if ground_truth is not None:
            supervised_loss = F.binary_cross_entropy_with_logits(student_output, ground_truth)
            total_loss = kd_loss + self.beta * supervised_loss
        else:
            total_loss = kd_loss  # Fallback to pure distillation loss if no ground truth

        return total_loss