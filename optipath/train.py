"""
Training Script for Blood Cell Disease Classifier
Trains EfficientNetV2-S on Leukemia, Malaria, or Sickle Cell datasets
Usage:
    python train.py --disease leukemia --data_dir ./datasets/leukemia
    python train.py --disease malaria --data_dir ./datasets/malaria
    python train.py --disease sickle_cell --data_dir ./datasets/sickle_cell
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent))
from configs.config import (
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, MODEL_DIR
)
from models.classifier import create_model, BloodCellClassifier
from datasets.dataset import (
    load_leukemia_dataset, load_malaria_dataset,
    load_sickle_cell_dataset, create_dataloaders
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=loss.item(), acc=100. * correct / total)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train(disease: str, data_dir: str, epochs: int = EPOCHS):
    """Full training pipeline."""
    print(f"\n{'='*60}")
    print(f"  OptiPath - Training {disease.upper()} Classifier")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Load dataset based on disease type
    if disease == "leukemia":
        image_paths, labels = load_leukemia_dataset(data_dir)
        num_classes = 2
        class_names = ["Normal", "Leukemia"]
    elif disease == "malaria":
        image_paths, labels = load_malaria_dataset(data_dir)
        num_classes = 2
        class_names = ["Normal", "Malaria"]
    elif disease == "sickle_cell":
        image_paths, labels = load_sickle_cell_dataset(data_dir)
        num_classes = 2
        class_names = ["Normal", "SickleCell"]
    else:
        raise ValueError(f"Unknown disease: {disease}. Use: leukemia, malaria, sickle_cell")

    if len(image_paths) == 0:
        print("ERROR: No images found. Check your data_dir path.")
        return

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(image_paths, labels)

    # Create model
    model = create_model(num_classes=num_classes, pretrained=True)
    model.freeze_backbone()  # Start with frozen backbone
    print(f"\nModel: EfficientNetV2-S (backbone frozen)")
    print(f"Classes: {class_names}")

    # Loss with class weights
    class_counts = np.bincount(labels)
    weights = torch.FloatTensor(len(labels) / (len(class_counts) * class_counts)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer (only classifier params initially)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    unfreeze_epoch = 5  # Unfreeze backbone after 5 epochs

    save_path = MODEL_DIR / f"best_{disease}_model.pth"

    print(f"\nStarting training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after initial epochs
        if epoch == unfreeze_epoch:
            model.unfreeze_backbone()
            # Reset optimizer with lower LR for backbone
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(), "lr": LEARNING_RATE / 10},
                {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - epoch)
            print(f"\n>> Backbone UNFROZEN at epoch {epoch} (fine-tuning)\n")

        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - start

        print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%", end="")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "class_names": class_names,
                "num_classes": num_classes,
                "disease": disease,
            }, save_path)
            print(f" | SAVED (best)")
        else:
            patience_counter += 1
            print(f" | patience {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Final evaluation on test set
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved: {save_path}")
    print(f"{'='*60}")

    # Load best model and evaluate on test set
    model = BloodCellClassifier(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(save_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"\n  Test Accuracy: {test_acc:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train blood cell disease classifier")
    parser.add_argument("--disease", type=str, required=True,
                        choices=["leukemia", "malaria", "sickle_cell"],
                        help="Disease to train for")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of epochs (default: {EPOCHS})")

    args = parser.parse_args()
    train(args.disease, args.data_dir, args.epochs)
