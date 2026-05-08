"""
OptiPath Configuration
Hematological Disease Detection Pipeline
Diseases: Leukemia, Sickle Cell Anaemia, Malaria
"""
import os
from pathlib import Path
import torch

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Dataset settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 for macOS compatibility

# Classes for the combined classifier
CLASS_NAMES = ["Normal", "Leukemia", "Malaria", "SickleCell"]
NUM_CLASSES = len(CLASS_NAMES)

# For binary training (one disease at a time)
LEUKEMIA_CLASSES = ["Normal", "Leukemia"]
MALARIA_CLASSES = ["Normal", "Malaria"]
SICKLE_CELL_CLASSES = ["Normal", "SickleCell"]

# Training hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

# Model
MODEL_NAME = "tf_efficientnetv2_s"  # from timm library
PRETRAINED = True
DROP_RATE = 0.3

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Normalization (ImageNet stats)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# YOLO Detection settings
YOLO_MODEL = "yolov8s"  # ultralytics yolov8 small
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
CELL_CLASSES = ["RBC", "WBC", "Platelet"]
