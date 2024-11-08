import torch.nn as nn
import torch.nn.functional as F


class UndeviatingDistillationLoss(nn.Module):
    def __init__(self, alpha=1.0, temperature=5.0):
        """
        Initializes the loss module for knowledge distillation with configurable
        alpha and temperature.

        Args:
            alpha (float): Weighting factor for balancing distillation and pixel-wise MSE loss.
            temperature (float): Temperature for softening logits during distillation.
        """
        super(UndeviatingDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        self.criterion_mse = nn.MSELoss()

    def forward(self, student_logits, teacher_logits):
        """
        Forward pass to compute the combined distillation loss and pixel-wise MSE loss.

        Args:
            student_logits (torch.Tensor): Raw logits output from the student model.
            teacher_logits (torch.Tensor): Raw logits output from the teacher model.

        Returns:
            torch.Tensor: Combined loss (weighted sum of distillation loss and pixel-wise loss).
        """
        # Softened predictions for distillation
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)

        # Distillation loss
        loss_distillation = self.criterion_kl(soft_student, soft_teacher) * (self.temperature ** 2)

        # Pixel-wise MSE loss for raw logits
        loss_mse = self.criterion_mse(student_logits, teacher_logits)

        # Combined loss with balance factor
        loss = self.alpha * loss_distillation + (1 - self.alpha) * loss_mse
        return loss
