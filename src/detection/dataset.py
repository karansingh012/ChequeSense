"""PyTorch Dataset and data loading utilities for ChequeSense field detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from src.data.dataset_loader import DatasetLoader
from src.data.image_preprocessor import ImagePreprocessor

# Canonical mapping of the 6 annotated cheque fields
FIELD_CLASSES = [
    "background",
    "date",
    "amount",
    "ifsc",
    "acno",
    "sign",
    "name",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(FIELD_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(FIELD_CLASSES)}
NUM_CLASSES = len(FIELD_CLASSES)  # 7 classes (1 background + 6 foreground)


class ChequeDetectionDataset(Dataset):
    """PyTorch Dataset yielding cheque images and bounding box targets."""

    def __init__(
        self,
        manifest_path: Union[str, Path] = "data/manifests/train_manifest.json",
        target_size: Optional[Tuple[int, int]] = (1000, 460),  # (width, height)
        augment: bool = False,
        base_dir: str = "Dataset",
    ):
        self.manifest_path = Path(manifest_path)
        self.target_size = target_size
        self.augment = augment
        self.loader = DatasetLoader(base_dir=base_dir)

        # Load samples from manifest
        with open(self.manifest_path) as f:
            all_records = json.load(f)

        # Filter strictly to synthetic dataset with bounding boxes
        self.records = [
            r for r in all_records
            if r["dataset_name"] == "synthetic" and r.get("annotation_type") == "bounding_box"
        ]

        # Build fast lookup map for synthetic sample images
        self._sample_cache = {s.sample_id: s for s in self.loader.load_synthetic(split=None)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec = self.records[idx]
        sample_id = rec["sample_id"]
        sample = self._sample_cache[sample_id]
        img = sample.get_image().convert("RGB")
        orig_w, orig_h = img.size

        # Extract bounding boxes
        raw_boxes = rec["annotations"]
        boxes: List[List[float]] = []
        labels: List[int] = []

        for field_name, coords in raw_boxes.items():
            if field_name not in CLASS_TO_IDX:
                continue

            xmin = float(coords["xmin"])
            ymin = float(coords["ymin"])
            xmax = float(coords["xmax"])
            ymax = float(coords["ymax"])

            # Ensure valid geometry
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(CLASS_TO_IDX[field_name])

        # Resize image & scale bounding boxes if target_size is set
        if self.target_size is not None:
            target_w, target_h = self.target_size
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

            scaled_boxes = []
            for b in boxes:
                scaled_boxes.append([
                    b[0] * scale_x,
                    b[1] * scale_y,
                    b[2] * scale_x,
                    b[3] * scale_y,
                ])
            boxes = scaled_boxes
            curr_w, curr_h = target_w, target_h
        else:
            curr_w, curr_h = orig_w, orig_h

        # Convert to Tensor
        img_tensor = F.to_tensor(img)  # [3, H, W] in [0.0, 1.0]

        # Apply photometrics augmentations if training
        if self.augment:
            # Color jitter (brightness, contrast)
            if torch.rand(1).item() > 0.5:
                factor = 0.8 + 0.4 * torch.rand(1).item()
                img_tensor = F.adjust_brightness(img_tensor, factor)
            if torch.rand(1).item() > 0.5:
                factor = 0.8 + 0.4 * torch.rand(1).item()
                img_tensor = F.adjust_contrast(img_tensor, factor)

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        if len(boxes) > 0:
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "sample_id": sample_id,
        }

        return img_tensor, target


def get_detection_collate_fn():
    """Collate function for PyTorch DataLoader handling variable number of boxes."""
    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)
    return collate_fn
