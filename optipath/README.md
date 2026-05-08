# OptiPath - Hematological Blood Disease Detection

AI-powered pipeline for detecting blood diseases from microscope blood smear images.
Targets: **Sickle Cell Anaemia**, **Malaria**, and **Leukemia**

## Architecture

```
Full Blood Smear Image
        │
        ▼
   YOLO Detector  →  Detects & crops cells (RBC/WBC/Platelet)
        │
        ▼
  EfficientNetV2  →  Classifies: Normal / Leukemia / Malaria / SickleCell
        │
        ▼
   JSON Report
```

## Quick Start

### 1. Install Dependencies
```bash
cd optipath
pip install -r requirements.txt
```

### 2. Download Datasets

| Disease | Dataset | Link |
|---------|---------|------|
| Leukemia | C-NMC 2019 | kaggle.com/datasets/avk256/cnmc-leukemia |
| Malaria | NIH Cell Images | kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria |
| Sickle Cell | erythrocytesIDB | Search "Sickle Cell blood smear dataset" |
| Cell Detection | BCCD | github.com/Shenggan/BCCD_Dataset |

### 3. Organize Datasets
```
optipath/datasets/
├── leukemia/          ← C-NMC dataset (fold_0, fold_1, fold_2)
├── malaria/           ← NIH dataset (Parasitized/, Uninfected/)
├── sickle_cell/       ← (Normal/, SickleCell/)
└── bccd/              ← BCCD dataset for YOLO
```

### 4. Train Classifiers (one at a time)
```bash
# Train leukemia (already proven: 87%+ accuracy epoch 1)
python train.py --disease leukemia --data_dir ./datasets/leukemia

# Train malaria
python train.py --disease malaria --data_dir ./datasets/malaria

# Train sickle cell
python train.py --disease sickle_cell --data_dir ./datasets/sickle_cell
```

### 5. Train YOLO Cell Detector
```bash
# Convert BCCD to YOLO format
python models/detector.py --action convert --bccd_dir ./datasets/bccd --output_dir ./datasets/bccd_yolo

# Train YOLO
python models/detector.py --action train --data_yaml ./datasets/bccd_yolo/data.yaml --epochs 100
```

### 6. Run Full Pipeline
```bash
# On a pre-cropped cell image (no YOLO needed)
python pipeline.py --image /path/to/cell.png --skip_detection

# On a full blood smear (needs YOLO)
python pipeline.py --image /path/to/smear.jpg --detector ./models/yolo_best.pt
```

## Project Structure
```
optipath/
├── configs/
│   └── config.py          # All settings & hyperparameters
├── models/
│   ├── classifier.py      # EfficientNetV2 disease classifier
│   └── detector.py        # YOLO cell detector + BCCD conversion
├── datasets/
│   └── dataset.py         # Data loading for all 3 diseases
├── train.py               # Training script
├── pipeline.py            # Full inference pipeline
├── requirements.txt
└── README.md
```

## Current Status
- [x] Project structure
- [x] EfficientNetV2 classifier (supports all 3 diseases)
- [x] Dataset loaders (Leukemia, Malaria, Sickle Cell)
- [x] Training pipeline with class balancing
- [x] YOLO detection pipeline + BCCD converter
- [x] Full inference pipeline
- [ ] Train leukemia model (need dataset on this machine)
- [ ] Train malaria model (need dataset)
- [ ] Train sickle cell model (need dataset)
- [ ] Train YOLO detector (need BCCD dataset)
