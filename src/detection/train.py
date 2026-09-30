"""Training script for ChequeSense field detector.

Trains a lightweight Faster R-CNN MobileNetV3-Large FPN model on the
deduplicated synthetic cheque dataset, using validation mAP@50 for model
selection.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models.detection as d
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.detection.dataset import ChequeDetectionDataset, NUM_CLASSES, get_detection_collate_fn
from src.detection.evaluate import DetectionEvaluator, EvaluationResult


def create_detector_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
) -> nn.Module:
    """Builds a Faster R-CNN with MobileNetV3-Large FPN backbone."""
    if pretrained:
        weights = d.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = d.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    else:
        model = d.fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    # Replace classifier head for the 6 cheque fields + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    loss_components: Dict[str, float] = {}

    for images, targets in dataloader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += losses.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v.item()

    n = max(1, len(dataloader))
    avg_loss = running_loss / n
    avg_components = {k: v / n for k, v in loss_components.items()}
    avg_components["total_loss"] = avg_loss
    return avg_components


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_thresh: float = 0.3,
) -> EvaluationResult:
    """Evaluates detector on validation/test dataloader."""
    model.eval()
    all_preds: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []

    for images, targets in dataloader:
        images = [im.to(device) for im in images]
        predictions = model(images)

        # Move to CPU for evaluation
        for p in predictions:
            all_preds.append({k: v.cpu() for k, v in p.items()})
        for t in targets:
            all_targets.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()})

    evaluator = DetectionEvaluator(score_thresh=score_thresh)
    return evaluator.evaluate(all_preds, all_targets)


def run_training(
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 0.0005,
    weight_decay: float = 0.0001,
    target_size: Tuple[int, int] = (1000, 460),
    device_name: str = "auto",
    models_dir: str = "models/field_detector",
    artifacts_dir: str = "artifacts/field_detection",
    seed: int = 42,
) -> Dict[str, Any]:
    """Full training pipeline with early checkpointing on validation mAP@50."""
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device selection
    if device_name == "auto":
        # Default to CPU for maximum stability across all PyTorch architectures
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Executing field detector training on device: {device}")

    models_path = Path(models_dir)
    artifacts_path = Path(artifacts_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # 1. Datasets and Loaders
    train_dataset = ChequeDetectionDataset(
        manifest_path="data/manifests/train_manifest.json",
        target_size=target_size,
        augment=True,
    )
    val_dataset = ChequeDetectionDataset(
        manifest_path="data/manifests/val_manifest.json",
        target_size=target_size,
        augment=False,
    )

    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    collate_fn = get_detection_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 2. Model initialization
    model = create_detector_model(num_classes=NUM_CLASSES, pretrained=True)
    model.to(device)

    # 3. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Training Loop
    best_map50 = 0.0
    history: List[Dict[str, Any]] = []
    best_metrics: Dict[str, Any] = {}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss_dict = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # Validation evaluation
        val_result = evaluate_model(model, val_loader, device, score_thresh=0.3)
        epoch_dur = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss_dict["total_loss"], 4),
            "loss_classifier": round(train_loss_dict.get("loss_classifier", 0.0), 4),
            "loss_box_reg": round(train_loss_dict.get("loss_box_reg", 0.0), 4),
            "val_precision": val_result.mean_precision_50,
            "val_recall": val_result.mean_recall_50,
            "val_f1": val_result.mean_f1_50,
            "val_map50": val_result.map_50,
            "val_map50_95": val_result.map_50_95,
            "duration_sec": round(epoch_dur, 2),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_dur:.1f}s] | "
            f"Loss: {train_loss_dict['total_loss']:.4f} | "
            f"Val mAP@50: {val_result.map_50:.4f} | "
            f"Val Recall: {val_result.mean_recall_50:.4f} | "
            f"Val Precision: {val_result.mean_precision_50:.4f}"
        )

        # Checkpoint best model based on validation mAP@50
        if val_result.map_50 >= best_map50:
            best_map50 = val_result.map_50
            best_metrics = val_result.to_dict()
            best_model_path = models_path / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_map50": best_map50,
                "target_size": target_size,
                "num_classes": NUM_CLASSES,
                "classes": NUM_CLASSES,
            }, best_model_path)

    total_training_time = time.time() - start_time

    # Save final model
    final_model_path = models_path / "final_model.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "final_map50": history[-1]["val_map50"],
        "target_size": target_size,
    }, final_model_path)

    # Save training history & metadata
    summary = {
        "model_architecture": "Faster R-CNN (MobileNetV3-Large FPN)",
        "num_classes": NUM_CLASSES,
        "classes": NUM_CLASSES,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "best_val_map50": round(best_map50, 4),
        "total_training_time_sec": round(total_training_time, 2),
        "history": history,
        "best_validation_metrics": best_metrics,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
    }

    history_path = artifacts_path / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete in {total_training_time:.1f}s. Best Val mAP@50: {best_map50:.4f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChequeSense field detector.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, mps, cuda)")
    args = parser.parse_args()

    run_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device_name=args.device)
