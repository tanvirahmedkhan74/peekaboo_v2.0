import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-5)
        return focal_loss.mean()
