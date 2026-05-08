# OptiPath — Hematological Blood Disease Detection

AI-powered pipeline for detecting blood diseases from microscope blood smear images using deep learning.

**Targets:** Sickle Cell Anaemia, Malaria, and Leukemia

## Architecture

```
Full Blood Smear Image
        │
        ▼
   YOLO Detector  →  Detects & crops individual cells (RBC / WBC / Platelet)
        │
        ▼
  EfficientNetV2  →  Classifies each cell: Normal / Leukemia / Malaria / Sickle Cell
        │
        ▼
   JSON Report
```

## Project Structure

```
├── predict.py                 # Quick single-image prediction script
├── optipath/
│   ├── pipeline.py            # Full inference pipeline (detection + classification)
│   ├── train.py               # Training script for all disease classifiers
│   ├── configs/
│   │   └── config.py          # All settings & hyperparameters
│   ├── models/
│   │   ├── classifier.py      # EfficientNetV2 disease classifier
│   │   ├── detector.py        # YOLO cell detector + BCCD converter
│   │   ├── best_leukemia_model.pth
│   │   ├── best_malaria_model.pth
│   │   ├── best_sickle_cell_model.pth
│   │   └── yolo_cell_detector.pt
│   ├── datasets/
│   │   └── dataset.py         # Data loading for all 3 diseases
│   └── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Mukund-31/yp.git
cd yp
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r optipath/requirements.txt
```

## How to Run

### Predict on a single cell image

```bash
python predict.py <path_to_cell_image>
```

### Run the full pipeline on a blood smear image

```bash
# Full blood smear → YOLO detects cells → classifier predicts disease
python optipath/pipeline.py --image /path/to/blood_smear.jpg

# If the image is already a cropped cell (skip YOLO detection)
python optipath/pipeline.py --image /path/to/cell_crop.jpg --skip_detection
```

## Training (Optional)

Download datasets and place them under `optipath/datasets/`:

| Disease     | Dataset              | Source                                              |
|-------------|----------------------|-----------------------------------------------------|
| Leukemia    | C-NMC 2019           | kaggle.com/datasets/avk256/cnmc-leukemia            |
| Malaria     | NIH Cell Images      | kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria |
| Sickle Cell | erythrocytesIDB      | Search "Sickle Cell blood smear dataset"            |
| Cell Detection | BCCD              | github.com/Shenggan/BCCD_Dataset                    |

### Train classifiers

```bash
cd optipath

python train.py --disease leukemia --data_dir ./datasets/leukemia
python train.py --disease malaria --data_dir ./datasets/malaria
python train.py --disease sickle_cell --data_dir ./datasets/sickle_cell
```

### Train YOLO cell detector

```bash
cd optipath

# Convert BCCD dataset to YOLO format
python models/detector.py --action convert --bccd_dir ./datasets/bccd --output_dir ./datasets/bccd_yolo

# Train
python models/detector.py --action train --data_yaml ./datasets/bccd_yolo/data.yaml --epochs 100
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See [optipath/requirements.txt](optipath/requirements.txt) for full list

## Pre-trained Models

Trained model weights are included in `optipath/models/`:
- `best_leukemia_model.pth` — Leukemia classifier
- `best_malaria_model.pth` — Malaria classifier
- `best_sickle_cell_model.pth` — Sickle Cell classifier
- `yolo_cell_detector.pt` — YOLO cell detection model
