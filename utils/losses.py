import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification with class imbalance.
    FL(p) = -alpha * (1-p)^gamma * log(p)        for positives
            -(1-alpha) * p^gamma * log(1-p)       for negatives
    alpha: weight for the positive class (set to ~negative rate to balance)
    gamma: focusing parameter (2 is standard)
    """
    def __init__(self, alpha=0.9, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        p_t     = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()
