"""ChequeSense OCR and Text-Processing Module."""

from src.ocr.confidence import ConfidenceAssessor, FieldConfidenceReport
from src.ocr.ocr_engine import OCREngine, OCRResult, OCRWordToken
from src.ocr.postprocess import FieldNormalizer, NormalizedFieldResult
from src.ocr.preprocessing import OCRPreprocessor

__all__ = [
    "OCREngine",
    "OCRResult",
    "OCRWordToken",
    "OCRPreprocessor",
    "FieldNormalizer",
    "NormalizedFieldResult",
    "ConfidenceAssessor",
    "FieldConfidenceReport",
]
