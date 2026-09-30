"""Training pipeline for ChequeSense handwritten/printed digit recognition model."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.recognition.dataset import ChequeDigitDataset, NUM_CLASSES
from src.recognition.evaluate import RecognitionEvaluator, RecognitionResult
from src.recognition.model import ChequeDigitCNN


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for tensors, labels, _ in dataloader:
        tensors = tensors.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(tensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * tensors.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += tensors.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, RecognitionResult]:
    """Evaluates validation loss and complete multi-class metrics."""
    model.eval()
    running_loss = 0.0
    total = 0

    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []

    for tensors, labels, _ in dataloader:
        tensors = tensors.to(device)
        labels_dev = labels.to(device)

        outputs = model(tensors)
        loss = criterion(outputs, labels_dev)
        running_loss += loss.item() * tensors.size(0)
        total += tensors.size(0)

        preds, confs = model.predict_glyph(tensors)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        confidences.extend(confs.cpu().tolist())

    val_loss = running_loss / max(1, total)
    evaluator = RecognitionEvaluator()
    result = evaluator.evaluate(y_true, y_pred, confidences)
    return val_loss, result


def run_training(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    device_name: str = "auto",
    models_dir: str = "models/recognizer",
    artifacts_dir: str = "artifacts/recognition",
    seed: int = 42,
) -> Dict[str, Any]:
    """Full training pipeline with validation-based checkpointing."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu" if device_name == "auto" else device_name)
    print(f"Training ChequeDigitCNN on device: {device}")

    models_path = Path(models_dir)
    artifacts_path = Path(artifacts_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # 1. Datasets
    train_dataset = ChequeDigitDataset(manifest_path="data/manifests/train_manifest.json", augment=True)
    val_dataset = ChequeDigitDataset(manifest_path="data/manifests/val_manifest.json", augment=False)

    print(f"Loaded {len(train_dataset)} training glyphs and {len(val_dataset)} validation glyphs.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model & Optimization
    model = ChequeDigitCNN(num_classes=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 3. Training Loop
    best_val_f1 = 0.0
    best_metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_loss, val_result = evaluate_epoch(model, val_loader, criterion, device)
        epoch_dur = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_result.accuracy, 4),
            "val_f1": round(val_result.macro_f1, 4),
            "val_precision": round(val_result.macro_precision, 4),
            "val_recall": round(val_result.macro_recall, 4),
            "duration_sec": round(epoch_dur, 2),
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_dur:.1f}s] | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_result.accuracy*100:.1f}%, F1: {val_result.macro_f1:.4f})"
        )

        # Checkpoint best model based on validation F1-score
        if val_result.macro_f1 >= best_val_f1:
            best_val_f1 = val_result.macro_f1
            best_metrics = val_result.to_dict()
            best_model_path = models_path / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_result.accuracy,
                "val_f1": best_val_f1,
                "num_classes": NUM_CLASSES,
            }, best_model_path)

    total_time = time.time() - start_time

    # Save final model
    final_model_path = models_path / "final_model.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "final_f1": history[-1]["val_f1"],
    }, final_model_path)

    # Save history
    summary = {
        "model_architecture": "ChequeDigitCNN",
        "num_classes": NUM_CLASSES,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "best_val_f1": round(best_val_f1, 4),
        "total_training_time_sec": round(total_time, 2),
        "history": history,
        "best_validation_metrics": best_metrics,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
    }

    history_file = artifacts_path / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete in {total_time:.1f}s. Best Val F1: {best_val_f1:.4f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChequeSense digit recognizer.")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    run_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
