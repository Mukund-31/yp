# OptiPath — Hematological Blood Disease Detection

AI-powered pipeline for detecting blood diseases from microscope blood smear images using deep learning.

**Targets:** Leukemia, Malaria, and Sickle Cell Anaemia

---

## Architecture

```
Full Blood Smear Image
        │
        ▼
┌───────────────────┐
│   YOLO Detector   │  Finds all cells, labels them RBC / WBC / Platelet
└───────────────────┘
        │  Cropped cells
        ▼
┌───────────────────┐
│  EfficientNetV2-S │  Classifies each cell for disease
└───────────────────┘
        │
        ▼
   Diagnosis Report (JSON)
```

**Smart routing:** WBCs → checked for Leukemia | RBCs → checked for Malaria & Sickle Cell

---

## Trained Models & Results

All 4 models are pre-trained and included in `optipath/models/`:

| Model | File | Test Accuracy | Test Samples | Dataset Size |
|-------|------|--------------|-------------|-------------|
| YOLO Cell Detector | `yolo_cell_detector.pt` (23 MB) | mAP50: 76.2% | — | 1,260 images |
| Leukemia Classifier | `best_leukemia_model.pth` (80 MB) | **90.44%** | 1,067 | 10,661 images |
| Malaria Classifier | `best_malaria_model.pth` (80 MB) | **97.50%** | 2,756 | 27,558 images |
| Sickle Cell Classifier | `best_sickle_cell_model.pth` (80 MB) | **92.00%** | 100 | 991 images |

> Accuracy is measured on held-out test sets the models never saw during training. Run `python optipath/evaluate.py` to reproduce these numbers.

---

## Project Structure

```
├── predict.py                     # Quick single-image prediction
├── optipath/
│   ├── pipeline.py                # Full inference pipeline (detect → classify → report)
│   ├── train.py                   # Training script for disease classifiers
│   ├── evaluate.py                # Evaluation script (generates accuracy proof)
│   ├── configs/
│   │   └── config.py              # All settings & hyperparameters
│   ├── models/
│   │   ├── classifier.py          # EfficientNetV2-S model definition
│   │   ├── detector.py            # YOLO cell detector wrapper
│   │   ├── best_leukemia_model.pth
│   │   ├── best_malaria_model.pth
│   │   ├── best_sickle_cell_model.pth
│   │   └── yolo_cell_detector.pt
│   ├── datasets/
│   │   └── dataset.py             # Data loaders for all 3 diseases
│   ├── evaluation_results/        # Confusion matrices & confidence charts
│   └── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Mukund-31/yp.git
cd yp
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r optipath/requirements.txt
```

---

## How to Run

### Classify a single cell image (no YOLO needed)

```bash
python optipath/pipeline.py --image /path/to/cell_image.jpg --skip_detection
```

**Example output:**
```
Results for: cell_image.jpg
----------------------------------------
  leukemia: Normal (100.0%)
  malaria: Malaria (100.0%)
  sickle_cell: Normal (87.2%)
```

### Analyze a full blood smear image

```bash
python optipath/pipeline.py --image /path/to/blood_smear.jpg
```

YOLO detects all cells → crops each one → classifies for disease → outputs a full report:

```
Analysis Report
========================================
Image: blood_smear.jpg
Cells detected: 47

Summary:
  Normal: 42
  Leukemia: 0
  Malaria: 3
  SickleCell: 2
```

### Quick predict (legacy single-disease script)

```bash
python predict.py /path/to/cell_image.bmp
```

---

## Evaluate Models (Reproduce Accuracy)

Run evaluation on the test sets to generate confusion matrices, classification reports, and confidence distribution charts:

```bash
cd optipath
python evaluate.py
```

This produces:
- Accuracy, Precision, Recall, F1-Score for each disease
- Confusion matrix images in `evaluation_results/`
- Confidence distribution charts

> Requires datasets to be present in `optipath/datasets/`. See [Training](#training-optional) section.

---

## Training (Optional)

### Download datasets

Download and place under `optipath/datasets/`:

| Disease | Dataset | Source |
|---------|---------|--------|
| Leukemia | C-NMC 2019 (10,661 images) | [kaggle.com/datasets/avk256/cnmc-leukemia](https://kaggle.com/datasets/avk256/cnmc-leukemia) |
| Malaria | NIH Cell Images (27,558 images) | [kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria](https://kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) |
| Sickle Cell | erythrocytesIDB (991 images) | [kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset](https://kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset) |
| Cell Detection | TXL-PBC (1,260 images) | [github.com/lugan113/TXL-PBC_Dataset](https://github.com/lugan113/TXL-PBC_Dataset) |

### Train disease classifiers

```bash
cd optipath

# Train leukemia classifier
python train.py --disease leukemia --data_dir ./datasets/leukemia

# Train malaria classifier
python train.py --disease malaria --data_dir ./datasets/malaria

# Train sickle cell classifier
python train.py --disease sickle_cell --data_dir ./datasets/sickle_cell
```

**Training strategy:**
- Epochs 1–4: Backbone frozen (transfer learning from ImageNet)
- Epochs 5+: Backbone unfrozen with 10x lower learning rate (fine-tuning)
- Early stopping with patience of 7 epochs
- Weighted loss function for class imbalance
- Data augmentation: flips, rotation, color jitter

### Train YOLO cell detector

```bash
cd optipath

# Using TXL-PBC dataset (recommended)
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='datasets/txl-pbc/TXL-PBC/data.yaml', epochs=50, imgsz=640, batch=16, patience=10, device='mps', workers=0)
"

# Or using BCCD dataset
python models/detector.py --action convert --bccd_dir ./datasets/bccd --output_dir ./datasets/bccd_yolo
python models/detector.py --action train --data_yaml ./datasets/bccd_yolo/data.yaml --epochs 100
```

---

## All Commands Reference

| Command | Description |
|---------|-------------|
| `python optipath/pipeline.py --image <img> --skip_detection` | Classify a single cell image |
| `python optipath/pipeline.py --image <img>` | Full smear analysis (YOLO + classify) |
| `python predict.py <img>` | Quick single-image prediction |
| `python optipath/evaluate.py` | Evaluate all models, generate proof |
| `python optipath/train.py --disease leukemia --data_dir ./datasets/leukemia` | Train leukemia model |
| `python optipath/train.py --disease malaria --data_dir ./datasets/malaria` | Train malaria model |
| `python optipath/train.py --disease sickle_cell --data_dir ./datasets/sickle_cell` | Train sickle cell model |

### Pipeline CLI options

```
python optipath/pipeline.py [OPTIONS]

Required:
  --image <path>              Input image path

Optional:
  --skip_detection            Skip YOLO (for pre-cropped cell images)
  --detector <path>           Custom YOLO model path (default: models/yolo_cell_detector.pt)
  --leukemia_model <path>     Custom leukemia model path
  --malaria_model <path>      Custom malaria model path
  --sickle_cell_model <path>  Custom sickle cell model path
```

---

## Technical Details

### Model Architecture
- **Backbone:** EfficientNetV2-S (pretrained on ImageNet, 1280 features)
- **Head:** Dropout(0.3) → Linear(1280→512) → ReLU → Dropout(0.15) → Linear(512→2)
- **Detector:** YOLOv8-nano (640×640 input)

### Hyperparameters
- Image size: 224×224
- Batch size: 32
- Learning rate: 1e-4 (head), 1e-5 (backbone after unfreeze)
- Optimizer: AdamW with weight decay 1e-4
- Scheduler: Cosine Annealing
- Early stopping patience: 7 epochs

### Hardware
- Trained on Apple M4 Pro (MPS acceleration)
- Total training time: ~2.5 hours

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See [optipath/requirements.txt](optipath/requirements.txt) for full list

```
torch, torchvision, timm, ultralytics, pillow, numpy, tqdm, scikit-learn, matplotlib
```
