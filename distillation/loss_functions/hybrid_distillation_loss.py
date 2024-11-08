import torch.nn as nn
import torch.nn.functional as F

class HybridDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        """
        Initializes the hybrid distillation loss module with configurable alpha and temperature.

        Args:
            alpha (float): Weighting factor for balancing distillation and pixel-wise MSE loss.
            temperature (float): Temperature for softening logits during distillation.
        """
        super(HybridDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        self.criterion_mse = nn.MSELoss()

    def forward(self, student_output, student_output_binary, teacher_output, ground_truth=None):
        """
        Forward pass to compute the hybrid distillation loss and supervised loss.

        Args:
            student_output (torch.Tensor): Raw logits from the student model.
            student_output_binary (torch.Tensor): Binarized output from the student model (for supervised loss).
            teacher_output (torch.Tensor): Raw logits from the teacher model.
            ground_truth (torch.Tensor, optional): Ground truth labels for supervised loss (for binary cross-entropy).

        Returns:
            torch.Tensor: The total loss, combining distillation loss, MSE loss, and supervised loss.
        """
        # Softened teacher and student outputs for distillation
        soft_teacher = F.softmax(teacher_output / self.temperature, dim=1)
        soft_student = F.log_softmax(student_output / self.temperature, dim=1)

        # Distillation loss (KL Divergence)
        kd_loss = self.criterion_kl(soft_student, soft_teacher) * (self.temperature ** 2)

        # Pixel-wise MSE loss between teacher and student raw logits
        mse_loss = self.criterion_mse(student_output, teacher_output)

        # Combine the distillation loss and the MSE loss, weighted by alpha
        distillation_loss = self.alpha * kd_loss + (1 - self.alpha) * mse_loss

        # Supervised loss with ground truth using binarized student output (if ground_truth is provided)
        if ground_truth is not None:
            supervised_loss = F.binary_cross_entropy_with_logits(student_output_binary, ground_truth)
            total_loss = distillation_loss + supervised_loss
        else:
            total_loss = distillation_loss  # Only distillation loss if no ground truth is provided

        return total_loss
