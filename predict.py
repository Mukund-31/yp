"""
Leukemia Classifier - Prediction Script
Usage: python predict.py <image_path>
"""

import sys
from pathlib import Path

# Check and install dependencies
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    import timm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Installing required packages...")
    import os
    os.system("pip install torch torchvision timm pillow")
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    import timm

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["Normal", "Leukemia"]

# Auto-detect device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


class LeukemiaClassifier(nn.Module):
    """EfficientNetV2-based classifier for Leukemia detection."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=False,
            num_classes=0
        )
        self.num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def load_model():
    """Load the trained model."""
    model = LeukemiaClassifier()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image_path):
    """Predict on a single image."""
    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(1).item()

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": probs[pred_idx].item() * 100,
        "normal_prob": probs[0].item() * 100,
        "leukemia_prob": probs[1].item() * 100
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py test_images\\leukemia_sample.bmp")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    print("Loading model...")
    model = load_model()

    print(f"Analyzing: {image_path.name}")
    result = predict_image(model, image_path)

    print()
    print("=" * 40)
    print(f"Image: {image_path.name}")
    print("=" * 40)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Probabilities:")
    print(f"  Normal:   {result['normal_prob']:.2f}%")
    print(f"  Leukemia: {result['leukemia_prob']:.2f}%")


if __name__ == "__main__":
    main()
