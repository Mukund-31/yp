"""
Full Inference Pipeline for OptiPath
Takes a full blood smear image → detects cells → classifies disease

Usage:
    python pipeline.py --image /path/to/blood_smear.jpg
    python pipeline.py --image /path/to/cell_crop.jpg --skip_detection
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).parent))
from configs.config import (
    DEVICE, IMAGE_SIZE, MEAN, STD, MODEL_DIR,
    LEUKEMIA_CLASSES, MALARIA_CLASSES, SICKLE_CELL_CLASSES
)
from models.classifier import load_model
from models.detector import CellDetector


class OptiPathPipeline:
    """
    Full blood disease detection pipeline.
    Blood Smear → YOLO (detect cells) → EfficientNetV2 (classify) → Report
    """

    def __init__(self, detector_path: str = None,
                 leukemia_model: str = None,
                 malaria_model: str = None,
                 sickle_cell_model: str = None):
        """
        Initialize pipeline with model paths.
        Pass None for models not yet trained.
        """
        # Cell detector
        self.detector = None
        if detector_path and Path(detector_path).exists():
            self.detector = CellDetector(detector_path)

        # Disease classifiers
        self.classifiers = {}

        if leukemia_model and Path(leukemia_model).exists():
            self.classifiers["leukemia"] = {
                "model": load_model(leukemia_model, num_classes=2),
                "classes": LEUKEMIA_CLASSES
            }
            print(f"Loaded leukemia model")

        if malaria_model and Path(malaria_model).exists():
            self.classifiers["malaria"] = {
                "model": load_model(malaria_model, num_classes=2),
                "classes": MALARIA_CLASSES
            }
            print(f"Loaded malaria model")

        if sickle_cell_model and Path(sickle_cell_model).exists():
            self.classifiers["sickle_cell"] = {
                "model": load_model(sickle_cell_model, num_classes=2),
                "classes": SICKLE_CELL_CLASSES
            }
            print(f"Loaded sickle cell model")

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def classify_cell(self, cell_image: np.ndarray, cell_type: str = None) -> Dict:
        """
        Classify a single cell crop.
        If cell_type is known (from YOLO), only run relevant classifier:
          - WBC → check leukemia
          - RBC → check malaria, sickle cell
        """
        image = Image.fromarray(cell_image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(DEVICE)

        results = {}

        # Determine which classifiers to run
        if cell_type == "WBC":
            diseases_to_check = ["leukemia"]
        elif cell_type == "RBC":
            diseases_to_check = ["malaria", "sickle_cell"]
        else:
            # Unknown cell type, run all
            diseases_to_check = list(self.classifiers.keys())

        for disease in diseases_to_check:
            if disease not in self.classifiers:
                continue

            model = self.classifiers[disease]["model"]
            classes = self.classifiers[disease]["classes"]

            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred_idx = output.argmax(1).item()

            results[disease] = {
                "prediction": classes[pred_idx],
                "confidence": probs[pred_idx].item() * 100,
                "probabilities": {
                    classes[i]: probs[i].item() * 100
                    for i in range(len(classes))
                }
            }

        return results

    def analyze_smear(self, image_path: str) -> Dict:
        """
        Full pipeline: detect cells then classify each one.
        Returns comprehensive report.
        """
        report = {
            "image": str(image_path),
            "cells_detected": 0,
            "cell_results": [],
            "summary": {}
        }

        if self.detector is None:
            print("No YOLO detector loaded. Use --skip_detection for pre-cropped cells.")
            return report

        # Step 1: Detect cells
        detections = self.detector.detect_cells(image_path)
        report["cells_detected"] = len(detections)

        # Step 2: Crop and classify each cell
        crops = self.detector.crop_cells(image_path, detections)

        disease_counts = {"Normal": 0, "Leukemia": 0, "Malaria": 0, "SickleCell": 0}

        for cell_image, det_info in crops:
            cell_type = det_info["class_name"]  # RBC, WBC, Platelet
            classification = self.classify_cell(cell_image, cell_type)

            cell_result = {
                "cell_type": cell_type,
                "bbox": det_info["bbox"],
                "detection_confidence": det_info["confidence"],
                "classification": classification
            }
            report["cell_results"].append(cell_result)

            # Count disease predictions
            for disease, result in classification.items():
                pred = result["prediction"]
                if pred in disease_counts:
                    disease_counts[pred] += 1

        report["summary"] = disease_counts
        return report

    def classify_single_image(self, image_path: str) -> Dict:
        """
        Classify a single pre-cropped cell image (no detection needed).
        Runs all available classifiers.
        """
        image = Image.open(image_path).convert("RGB")
        cell_image = np.array(image)
        return self.classify_cell(cell_image)


def main():
    parser = argparse.ArgumentParser(description="OptiPath Blood Disease Detection")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--skip_detection", action="store_true",
                        help="Skip YOLO detection (for pre-cropped cell images)")
    parser.add_argument("--detector", type=str,
                        default=str(MODEL_DIR / "yolo_cell_detector.pt"),
                        help="Path to YOLO model")
    parser.add_argument("--leukemia_model", type=str,
                        default=str(MODEL_DIR / "best_leukemia_model.pth"))
    parser.add_argument("--malaria_model", type=str,
                        default=str(MODEL_DIR / "best_malaria_model.pth"))
    parser.add_argument("--sickle_cell_model", type=str,
                        default=str(MODEL_DIR / "best_sickle_cell_model.pth"))

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = OptiPathPipeline(
        detector_path=args.detector,
        leukemia_model=args.leukemia_model,
        malaria_model=args.malaria_model,
        sickle_cell_model=args.sickle_cell_model,
    )

    if not pipeline.classifiers:
        print("ERROR: No trained models found. Train at least one classifier first.")
        print("  python train.py --disease leukemia --data_dir ./datasets/leukemia")
        return

    if args.skip_detection:
        # Direct classification of a cell image
        results = pipeline.classify_single_image(args.image)
        print(f"\nResults for: {args.image}")
        print("-" * 40)
        for disease, result in results.items():
            print(f"  {disease}: {result['prediction']} ({result['confidence']:.1f}%)")
    else:
        # Full pipeline with detection
        report = pipeline.analyze_smear(args.image)
        print(f"\nAnalysis Report")
        print("=" * 40)
        print(f"Image: {report['image']}")
        print(f"Cells detected: {report['cells_detected']}")
        print(f"\nSummary:")
        for disease, count in report.get("summary", {}).items():
            print(f"  {disease}: {count}")
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
