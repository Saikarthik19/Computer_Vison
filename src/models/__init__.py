"""
Model Factory for Satellite Road Extraction
"""

import torch
import torch.nn as nn
from .dlinknet import DLinkNet34
from .unet import AttentionUNet
from .loss import CombinedRoadLoss, DiceLoss, FocalLoss, SoftIoULoss


def build_model(architecture: str = "DLinkNet34", num_classes: int = 1, pretrained: bool = False) -> nn.Module:
    arch = architecture.lower()
    if "dlink" in arch:
        return DLinkNet34(num_classes=num_classes, pretrained=pretrained)
    elif "attention" in arch or "unet" in arch:
        return AttentionUNet(in_channels=3, num_classes=num_classes)
    else:
        # Default to DLinkNet34
        return DLinkNet34(num_classes=num_classes, pretrained=pretrained)


__all__ = [
    "DLinkNet34",
    "AttentionUNet",
    "CombinedRoadLoss",
    "DiceLoss",
    "FocalLoss",
    "SoftIoULoss",
    "build_model"
]
