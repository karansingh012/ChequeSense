"""Handwritten character and sequence recognition stage for ChequeSense."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

from src.pipeline.schemas import PipelineConfig
from src.recognition.predict import FieldPredictionResult, FieldRecognizer

logger = logging.getLogger("chequesense.pipeline.recognize")


@dataclass
class RecognitionOutput:
    """Output from the handwritten recognition module."""

    field_name: str
    recognized_text: str
    confidence: float
    character_count: int
    characters: List[Dict[str, Any]]
    success: bool


class PipelineRecognizer:
    """Dispatches cropped fields to the handwritten digit/character recognition module."""

    # Fields suitable for handwritten character recognition
    RECOGNITION_SUPPORTED_FIELDS = ["amount", "date"]

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.field_recognizer = FieldRecognizer(
            model_path=config.recognizer_model_path,
            device=config.device,
        )

    def recognize_field(self, crop_image: Image.Image, field_name: str) -> RecognitionOutput:
        """Transcribes handwritten field crops into text and per-character confidences."""
        try:
            res: FieldPredictionResult = self.field_recognizer.recognize_field(crop_image)
            return RecognitionOutput(
                field_name=field_name,
                recognized_text=res.predicted_text,
                confidence=round(res.aggregate_confidence, 4),
                character_count=len(res.characters),
                characters=res.characters,
                success=True,
            )
        except Exception as e:
            logger.error("Error recognizing field '%s': %s", field_name, e, exc_info=True)
            return RecognitionOutput(
                field_name=field_name,
                recognized_text="",
                confidence=0.0,
                character_count=0,
                characters=[],
                success=False,
            )
