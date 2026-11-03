"""
Road Extraction Loss Functions
Roads in satellite imagery are thin, long, and continuous, representing < 5% of tile area.
Standard Cross-Entropy produces fragmented results. We employ hybrid BCE + Dice + Focal / Soft IoU loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for segmenting thin linear structures (roads).
    Smooths gradients and handles severe foreground/background class imbalance.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss to focus learning on hard examples (narrow road edges, tree canopies, shadows).
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class SoftIoULoss(nn.Module):
    """
    Soft Jaccard / IoU Loss directly optimizing the primary road extraction metric.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        total = (probs + targets).sum()
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou


class CombinedRoadLoss(nn.Module):
    """
    Comprehensive Road Extraction Loss:
    L = w_bce * BCE + w_dice * Dice + w_iou * SoftIoU
    """
    def __init__(self, bce_weight: float = 0.4, dice_weight: float = 0.4, iou_weight: float = 0.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.iou = SoftIoULoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        iou_loss = self.iou(logits, targets)
        return (self.bce_weight * bce_loss +
                self.dice_weight * dice_loss +
                self.iou_weight * iou_loss)
