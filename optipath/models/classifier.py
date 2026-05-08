"""
EfficientNetV2-based Classifier for Blood Cell Disease Detection
Supports: Normal, Leukemia, Malaria, Sickle Cell Anaemia
"""
import torch
import torch.nn as nn
import timm
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    MODEL_NAME, NUM_CLASSES, DROP_RATE, DEVICE, MODEL_DIR, PRETRAINED
)


class BloodCellClassifier(nn.Module):
    """
    EfficientNetV2-S backbone with custom classification head.
    Can be used for binary (one disease) or multi-class (all diseases).
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=PRETRAINED):
        super().__init__()
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=0  # remove default head
        )
        self.num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROP_RATE / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        """Freeze backbone for initial training (transfer learning)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED):
    """Create a new model instance."""
    model = BloodCellClassifier(num_classes=num_classes, pretrained=pretrained)
    model.to(DEVICE)
    return model


def load_model(checkpoint_path, num_classes=NUM_CLASSES):
    """Load a trained model from checkpoint."""
    model = BloodCellClassifier(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model
