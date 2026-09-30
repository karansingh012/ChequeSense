"""Inference module for ChequeSense field detector.

Loads trained field detector checkpoint, processes full cheque images, and predicts
exact bounding box coordinates in original image space for the 6 cheque fields:
date, amount, ifsc, acno, sign, name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from src.detection.dataset import FIELD_CLASSES, IDX_TO_CLASS, NUM_CLASSES
from src.detection.train import create_detector_model


@dataclass
class FieldBox:
    """Predicted bounding box for a single cheque field."""

    field_name: str
    confidence: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "confidence": round(self.confidence, 4),
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class FieldPrediction:
    """Full prediction container for a cheque document."""

    image_size: Tuple[int, int]  # (orig_w, orig_h)
    fields: Dict[str, FieldBox]
    all_detections: List[FieldBox]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "detection_count": len(self.all_detections),
        }


class FieldDetector:
    """High-level wrapper for cheque field detection inference."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/field_detector/best_model.pt",
        device: str = "cpu",
        score_threshold: float = 0.35,
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.target_size = (1000, 460)

        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """Loads trained weights into Faster R-CNN architecture."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.target_size = checkpoint.get("target_size", (1000, 460))

        model = create_detector_model(num_classes=NUM_CLASSES, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        return_single_best_per_class: bool = True,
    ) -> FieldPrediction:
        """Predicts locations of cheque fields on a full cheque image.

        Args:
            image_input: Image path, PIL Image, or numpy array.
            return_single_best_per_class: If True, selects the highest-scoring box
                                         for each field class.

        Returns:
            FieldPrediction with mapped original coordinates.
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError("image_input must be a file path, PIL Image, or numpy array")

        orig_w, orig_h = pil_img.size

        # Resize for model input
        target_w, target_h = self.target_size
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        resized_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        img_tensor = F.to_tensor(resized_img).unsqueeze(0).to(self.device)

        # Forward pass
        predictions = self.model(img_tensor)[0]

        pred_boxes = predictions["boxes"].cpu().numpy()
        pred_scores = predictions["scores"].cpu().numpy()
        pred_labels = predictions["labels"].cpu().numpy()

        all_detections: List[FieldBox] = []
        best_per_class: Dict[str, FieldBox] = {}

        for b, score, label_id in zip(pred_boxes, pred_scores, pred_labels):
            if score < self.score_threshold or label_id not in IDX_TO_CLASS:
                continue

            field_name = IDX_TO_CLASS[int(label_id)]
            if field_name == "background":
                continue

            # Scale back to original image space
            orig_xmin = int(round(b[0] / scale_x))
            orig_ymin = int(round(b[1] / scale_y))
            orig_xmax = int(round(b[2] / scale_x))
            orig_ymax = int(round(b[3] / scale_y))

            # Clamp to image boundaries
            orig_xmin = max(0, min(orig_w - 1, orig_xmin))
            orig_ymin = max(0, min(orig_h - 1, orig_ymin))
            orig_xmax = max(orig_xmin + 1, min(orig_w, orig_xmax))
            orig_ymax = max(orig_ymin + 1, min(orig_h, orig_ymax))

            box = FieldBox(
                field_name=field_name,
                confidence=float(score),
                xmin=orig_xmin,
                ymin=orig_ymin,
                xmax=orig_xmax,
                ymax=orig_ymax,
            )
            all_detections.append(box)

            if return_single_best_per_class:
                if field_name not in best_per_class or score > best_per_class[field_name].confidence:
                    best_per_class[field_name] = box

        return FieldPrediction(
            image_size=(orig_w, orig_h),
            fields=best_per_class if return_single_best_per_class else {b.field_name: b for b in all_detections},
            all_detections=all_detections,
        )
