"""Unit tests for the ChequeSense field detection module."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.detection.dataset import (
    CLASS_TO_IDX,
    FIELD_CLASSES,
    IDX_TO_CLASS,
    NUM_CLASSES,
    ChequeDetectionDataset,
    get_detection_collate_fn,
)
from src.detection.evaluate import DetectionEvaluator, box_iou, compute_ap
from src.detection.inference import FieldBox, FieldPrediction
from src.detection.visualize import draw_field_detections


def test_field_classes_schema():
    assert NUM_CLASSES == 7
    assert FIELD_CLASSES[0] == "background"
    for name in ["date", "amount", "ifsc", "acno", "sign", "name"]:
        assert name in CLASS_TO_IDX
        assert IDX_TO_CLASS[CLASS_TO_IDX[name]] == name


def test_box_iou_computation():
    box1 = np.array([[10, 10, 50, 50]], dtype=np.float32)
    box2 = np.array([[10, 10, 50, 50]], dtype=np.float32)
    iou = box_iou(box1, box2)
    assert np.isclose(iou[0, 0], 1.0)

    # Disjoint boxes
    box3 = np.array([[100, 100, 150, 150]], dtype=np.float32)
    iou_disjoint = box_iou(box1, box3)
    assert iou_disjoint[0, 0] == 0.0

    # Half overlap: 40x40 area = 1600. Overlap 20x40 = 800. Union = 1600 + 1600 - 800 = 2400. IoU = 800/2400 = 1/3
    box4 = np.array([[30, 10, 70, 50]], dtype=np.float32)
    iou_half = box_iou(box1, box4)
    assert np.isclose(iou_half[0, 0], 800.0 / 2400.0, atol=1e-3)


def test_compute_ap():
    # Perfect precision across all recall points
    recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    precisions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    ap = compute_ap(recalls, precisions)
    assert np.isclose(ap, 1.0)


def test_detection_evaluator_synthetic():
    evaluator = DetectionEvaluator(iou_thresholds=[0.5])

    # Target
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 100.0, 50.0]]),
        "labels": torch.tensor([CLASS_TO_IDX["date"]]),
    }
    # Prediction: exact match
    pred = {
        "boxes": torch.tensor([[10.0, 10.0, 100.0, 50.0]]),
        "scores": torch.tensor([0.95]),
        "labels": torch.tensor([CLASS_TO_IDX["date"]]),
    }

    result = evaluator.evaluate([pred], [target])
    assert result.mean_precision_50 == 1.0
    assert result.mean_recall_50 == 1.0
    assert result.map_50 == 1.0
    assert "date" in result.per_class
    assert result.per_class["date"].true_positives_50 == 1


def test_detection_dataset_loading():
    dataset = ChequeDetectionDataset(
        manifest_path="data/manifests/val_manifest.json",
        target_size=(800, 360),
        augment=False,
    )
    assert len(dataset) > 0
    img, target = dataset[0]

    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 360, 800)  # [C, H, W]
    assert "boxes" in target
    assert "labels" in target
    assert target["boxes"].shape[0] == target["labels"].shape[0]

    # Verify boxes lie within resized image dimensions
    for b in target["boxes"]:
        assert b[0] >= 0
        assert b[1] >= 0
        assert b[2] <= 800.1
        assert b[3] <= 360.1
        assert b[2] > b[0]
        assert b[3] > b[1]


def test_detection_collate_fn():
    dataset = ChequeDetectionDataset(
        manifest_path="data/manifests/val_manifest.json",
        target_size=(400, 200),
    )
    collate_fn = get_detection_collate_fn()
    batch = [dataset[0], dataset[1]]
    images, targets = collate_fn(batch)
    assert len(images) == 2
    assert len(targets) == 2
    assert isinstance(images[0], torch.Tensor)
    assert isinstance(targets[0], dict)


def test_draw_field_detections():
    img = Image.new("RGB", (600, 300), color="white")
    box = FieldBox(
        field_name="date",
        confidence=0.92,
        xmin=400,
        ymin=20,
        xmax=580,
        ymax=80,
    )
    pred = FieldPrediction(
        image_size=(600, 300),
        fields={"date": box},
        all_detections=[box],
    )
    vis = draw_field_detections(img, pred)
    assert isinstance(vis, Image.Image)
    assert vis.size == (600, 300)
