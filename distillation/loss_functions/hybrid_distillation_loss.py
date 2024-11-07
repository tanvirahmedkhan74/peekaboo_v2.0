import torch.nn.functional as F
import torch.nn as nn

class HybridDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super(HybridDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_output, student_output_binary, teacher_output, ground_truth=None):
        # Distillation loss with teacher's predictions
        kd_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1),
            reduction="batchmean"
        ) * (self.alpha * self.temperature ** 2)

        # Supervised loss with ground truth using binarized student output
        if ground_truth is not None:
            supervised_loss = F.binary_cross_entropy_with_logits(student_output_binary, ground_truth)
            total_loss = kd_loss * self.alpha + supervised_loss * (1 - self.alpha)
        else:
            total_loss = kd_loss  # Fallback to pure distillation loss if no ground truth

        return total_loss