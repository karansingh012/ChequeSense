"""Field detection and region cropping stage for the ChequeSense inference pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from src.detection.inference import FieldBox, FieldDetector, FieldPrediction
from src.pipeline.schemas import BoundingBox, PipelineConfig, SignatureOutput

logger = logging.getLogger("chequesense.pipeline.detect")


@dataclass
class DetectedFieldCrop:
    """Cropped region of interest for an individual detected cheque field."""

    field_name: str
    crop_image: Image.Image
    confidence: float
    bounding_box: BoundingBox


class PipelineFieldDetector:
    """Manages object detection inference and field cropping."""

    # Mandatory financial fields
    CORE_FIELDS = ["date", "amount", "ifsc", "acno", "name"]

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detector = FieldDetector(
            model_path=config.detector_model_path,
            device=config.device,
            score_threshold=config.detection_threshold,
        )

    def detect_and_crop(
        self, original_image: Image.Image, padding: int = 4
    ) -> Tuple[Dict[str, DetectedFieldCrop], SignatureOutput, FieldPrediction]:
        """Detects cheque fields and crops their bounding regions."""
        orig_w, orig_h = original_image.size
        prediction = self.detector.predict(original_image)

        field_crops: Dict[str, DetectedFieldCrop] = {}
        signature_status = SignatureOutput(present=False, confidence=0.0)

        # 1. Process all detected fields
        for name, box in prediction.fields.items():
            if name == "sign":
                signature_status = SignatureOutput(
                    present=True,
                    confidence=round(box.confidence, 4),
                    bounding_box=BoundingBox(
                        xmin=box.xmin,
                        ymin=box.ymin,
                        xmax=box.xmax,
                        ymax=box.ymax,
                    ),
                )
                logger.debug("Signature detected with confidence %.3f at %s", box.confidence, (box.xmin, box.ymin, box.xmax, box.ymax))
                continue

            # Crop with safe padding clamped to image dimensions
            x1 = max(0, box.xmin - padding)
            y1 = max(0, box.ymin - padding)
            x2 = min(orig_w, box.xmax + padding)
            y2 = min(orig_h, box.ymax + padding)

            crop = original_image.crop((x1, y1, x2, y2))

            field_crops[name] = DetectedFieldCrop(
                field_name=name,
                crop_image=crop,
                confidence=box.confidence,
                bounding_box=BoundingBox(
                    xmin=box.xmin,
                    ymin=box.ymin,
                    xmax=box.xmax,
                    ymax=box.ymax,
                ),
            )
            logger.debug(
                "Field '%s' detected (confidence: %.3f, size: %dx%d)",
                name,
                box.confidence,
                crop.width,
                crop.height,
            )

        # 2. Check for missing core fields
        missing = [f for f in self.CORE_FIELDS if f not in field_crops]
        if missing:
            logger.warning("Core fields missing from initial high-confidence detection: %s", missing)

        return field_crops, signature_status, prediction
