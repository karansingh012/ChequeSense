"""ChequeSense end-to-end inference pipeline module."""

from src.pipeline.detect_fields import DetectedFieldCrop, PipelineFieldDetector
from src.pipeline.ocr import PipelineOCR, PipelineOCROutput
from src.pipeline.pipeline import ChequeInferencePipeline
from src.pipeline.preprocess import PipelineImagePreprocessor
from src.pipeline.recognize import PipelineRecognizer, RecognitionOutput
from src.pipeline.reconstruct import FieldReconstructor
from src.pipeline.schemas import (
    BoundingBox,
    ChequePipelineResult,
    FieldOutput,
    PipelineConfig,
    PipelineDiagnostics,
    SignatureOutput,
)

__all__ = [
    "ChequeInferencePipeline",
    "PipelineImagePreprocessor",
    "PipelineFieldDetector",
    "PipelineRecognizer",
    "PipelineOCR",
    "FieldReconstructor",
    "PipelineConfig",
    "ChequePipelineResult",
    "FieldOutput",
    "SignatureOutput",
    "BoundingBox",
    "PipelineDiagnostics",
    "DetectedFieldCrop",
    "RecognitionOutput",
    "PipelineOCROutput",
]
