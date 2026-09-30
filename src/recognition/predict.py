"""Inference prediction module for ChequeSense character and digit recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from src.recognition.dataset import DIGIT_CLASSES, IDX_TO_CLASS, NUM_CLASSES
from src.recognition.model import ChequeDigitCNN
from src.recognition.preprocessing import GlyphPreprocessor


@dataclass
class SingleCharPrediction:
    """Prediction output for an isolated character/digit glyph."""

    char: str
    class_id: int
    confidence: float
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "char": self.char,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


@dataclass
class FieldPredictionResult:
    """Prediction output for a multi-digit field (e.g. amount in figures or date)."""

    predicted_text: str
    aggregate_confidence: float
    characters: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_text": self.predicted_text,
            "aggregate_confidence": round(self.aggregate_confidence, 4),
            "character_count": len(self.characters),
            "characters": self.characters,
        }


class CharacterRecognizer:
    """Classifies single isolated character/digit glyphs."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/recognizer/best_model.pt",
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.preprocessor = GlyphPreprocessor(target_size=(32, 32))
        self.model = self._load_model()

    def _load_model(self) -> ChequeDigitCNN:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = ChequeDigitCNN(num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict_glyph(self, glyph_input: Union[Image.Image, np.ndarray, str, Path]) -> SingleCharPrediction:
        """Predicts class and confidence for an isolated character glyph."""
        if isinstance(glyph_input, (str, Path)):
            pil_img = Image.open(str(glyph_input))
            norm = self.preprocessor.normalize_glyph(pil_img)
        elif isinstance(glyph_input, (Image.Image, np.ndarray)):
            norm = self.preprocessor.normalize_glyph(glyph_input)
        else:
            raise TypeError("glyph_input must be an image, numpy array, or file path")

        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 32, 32]
        probs = self.model.predict_proba(tensor)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_char = IDX_TO_CLASS[pred_idx]

        prob_dict = {IDX_TO_CLASS[i]: float(probs[i]) for i in range(NUM_CLASSES)}

        return SingleCharPrediction(
            char=pred_char,
            class_id=pred_idx,
            confidence=confidence,
            probabilities=prob_dict,
        )


class FieldRecognizer:
    """Segments and recognizes multi-character field crops (courtesy amount, date)."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/recognizer/best_model.pt",
        device: str = "cpu",
    ):
        self.recognizer = CharacterRecognizer(model_path=model_path, device=device)
        self.preprocessor = self.recognizer.preprocessor

    def recognize_field(self, field_image: Union[Image.Image, np.ndarray, str, Path]) -> FieldPredictionResult:
        """Segments field into glyphs and transcribes complete numerical string."""
        if isinstance(field_image, (str, Path)):
            pil_img = Image.open(str(field_image))
        elif isinstance(field_image, np.ndarray):
            pil_img = Image.fromarray(field_image)
        elif isinstance(field_image, Image.Image):
            pil_img = field_image
        else:
            raise TypeError("field_image must be an Image, array, or file path")

        segmented = self.preprocessor.segment_field_into_glyphs(pil_img)

        if not segmented:
            return FieldPredictionResult(predicted_text="", aggregate_confidence=0.0, characters=[])

        chars = []
        confidences = []

        for norm_glyph, (x, y, w, h) in segmented:
            pred = self.recognizer.predict_glyph(norm_glyph)
            chars.append({
                "char": pred.char,
                "confidence": round(pred.confidence, 4),
                "box": {"x": x, "y": y, "w": w, "h": h},
            })
            confidences.append(pred.confidence)

        full_text = "".join(c["char"] for c in chars)
        agg_conf = float(np.mean(confidences)) if confidences else 0.0

        return FieldPredictionResult(
            predicted_text=full_text,
            aggregate_confidence=agg_conf,
            characters=chars,
        )
