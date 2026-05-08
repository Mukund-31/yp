"""
YOLO-based Blood Cell Detector
Detects and crops individual cells (RBC, WBC, Platelet) from full blood smear images.
Uses Ultralytics YOLOv8.

Recommended dataset for training:
  TXL-PBC Dataset (1,260 images, 18,143 annotations, already YOLO format)
  https://github.com/lugan113/TXL-PBC_Dataset
  Published: Nature Scientific Data 2025

Alternative: BCCD Dataset (364 images)
  https://github.com/Shenggan/BCCD_Dataset
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, CELL_CLASSES


class CellDetector:
    """
    YOLO-based cell detector for blood smear images.
    Detects RBC, WBC, and Platelets, crops them for classification.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize detector.
        Args:
            model_path: Path to trained YOLO model (.pt file)
                       If None, uses pretrained YOLOv8s (won't detect cells specifically)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model from: {model_path}")
        else:
            print("WARNING: No trained cell detection model found.")
            print("You need to train YOLO on BCCD dataset first.")
            print("See: train_yolo() function below.")
            self.model = None

    def detect_cells(self, image_path: str, conf: float = YOLO_CONF_THRESHOLD) -> List[Dict]:
        """
        Detect cells in a blood smear image.
        Returns list of detected cells with bounding boxes and types.
        """
        if self.model is None:
            raise RuntimeError("No YOLO model loaded. Train one first.")

        results = self.model(image_path, conf=conf, iou=YOLO_IOU_THRESHOLD)

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf_score = boxes.conf[i].cpu().item()
                cls_id = int(boxes.cls[i].cpu().item())
                cls_name = result.names[cls_id]

                detections.append({
                    "bbox": box.tolist(),  # [x1, y1, x2, y2]
                    "confidence": conf_score,
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

        return detections

    def crop_cells(self, image_path: str, detections: List[Dict],
                   padding: int = 10) -> List[Tuple[np.ndarray, Dict]]:
        """
        Crop detected cells from the image.
        Returns list of (cropped_image, detection_info) tuples.
        """
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        crops = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Add padding
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(img_w, int(x2) + padding)
            y2 = min(img_h, int(y2) + padding)

            crop = image.crop((x1, y1, x2, y2))
            crops.append((np.array(crop), det))

        return crops


def train_yolo(data_yaml: str, epochs: int = 100, model_size: str = "yolov8s"):
    """
    Train YOLO on BCCD dataset for cell detection.

    Before running, prepare your BCCD dataset in YOLO format:
    bccd_yolo/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml

    data.yaml should contain:
    ---
    train: ./train/images
    val: ./val/images
    nc: 3
    names: ['RBC', 'WBC', 'Platelet']
    ---

    Args:
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        model_size: YOLO model size (yolov8n, yolov8s, yolov8m)
    """
    from ultralytics import YOLO

    model = YOLO(f"{model_size}.pt")  # Load pretrained

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        name="blood_cell_detector",
        patience=20,
    )

    print(f"\nYOLO training complete!")
    print(f"Best model saved at: runs/detect/blood_cell_detector/weights/best.pt")
    return results


def convert_bccd_to_yolo(bccd_dir: str, output_dir: str):
    """
    Convert BCCD dataset (XML annotations) to YOLO format.
    Args:
        bccd_dir: Path to BCCD_Dataset directory
        output_dir: Where to save YOLO-formatted data
    """
    import xml.etree.ElementTree as ET
    import shutil

    bccd_path = Path(bccd_dir)
    output_path = Path(output_dir)

    # Class mapping
    class_map = {"RBC": 0, "WBC": 1, "Platelets": 2}

    annotations_dir = bccd_path / "BCCD" / "Annotations"
    images_dir = bccd_path / "BCCD" / "JPEGImages"

    if not annotations_dir.exists():
        print(f"ERROR: Annotations not found at {annotations_dir}")
        return

    # Get all annotation files
    xml_files = sorted(annotations_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} annotations")

    # Split 80/20 train/val
    np.random.seed(42)
    indices = np.random.permutation(len(xml_files))
    split = int(0.8 * len(xml_files))
    train_indices = indices[:split]
    val_indices = indices[split:]

    for split_name, split_indices in [("train", train_indices), ("val", val_indices)]:
        img_out = output_path / split_name / "images"
        lbl_out = output_path / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for idx in split_indices:
            xml_file = xml_files[idx]
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Image dimensions
            size = root.find("size")
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)

            # Copy image
            img_name = root.find("filename").text
            src_img = images_dir / img_name
            if src_img.exists():
                shutil.copy2(src_img, img_out / img_name)

            # Convert annotations to YOLO format
            label_lines = []
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in class_map:
                    continue
                cls_id = class_map[cls_name]

                bbox = obj.find("bndbox")
                x1 = int(bbox.find("xmin").text)
                y1 = int(bbox.find("ymin").text)
                x2 = int(bbox.find("xmax").text)
                y2 = int(bbox.find("ymax").text)

                # Convert to YOLO format (center_x, center_y, width, height) normalized
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Write label file
            label_file = lbl_out / (xml_file.stem + ".txt")
            with open(label_file, "w") as f:
                f.write("\n".join(label_lines))

    # Write data.yaml
    yaml_content = f"""train: {output_path / 'train' / 'images'}
val: {output_path / 'val' / 'images'}
nc: 3
names: ['RBC', 'WBC', 'Platelet']
"""
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"YOLO dataset created at: {output_path}")
    print(f"Train: {len(train_indices)} images, Val: {len(val_indices)} images")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["convert", "train", "detect"],
                        required=True)
    parser.add_argument("--bccd_dir", type=str, help="BCCD dataset directory")
    parser.add_argument("--output_dir", type=str, help="Output directory for YOLO data")
    parser.add_argument("--data_yaml", type=str, help="data.yaml for training")
    parser.add_argument("--model_path", type=str, help="Trained model for detection")
    parser.add_argument("--image", type=str, help="Image to detect cells in")
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    if args.action == "convert":
        convert_bccd_to_yolo(args.bccd_dir, args.output_dir)
    elif args.action == "train":
        train_yolo(args.data_yaml, args.epochs)
    elif args.action == "detect":
        detector = CellDetector(args.model_path)
        detections = detector.detect_cells(args.image)
        print(f"Found {len(detections)} cells")
        for d in detections:
            print(f"  {d['class_name']}: conf={d['confidence']:.2f} bbox={d['bbox']}")
