"""Pydantic schemas and data transfer objects for the ChequeSense inference pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Spatial bounding box coordinates on the cheque image."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int


class FieldOutput(BaseModel):
    """Standardized output representation for an individual extracted cheque field."""

    value: str = Field(description="Normalized, sanitized field value")
    raw_value: Optional[str] = Field(default="", description="Original raw extraction before normalization")
    confidence: float = Field(ge=0.0, le=1.0, description="Propagated composite confidence score")
    detection_confidence: float = Field(ge=0.0, le=1.0, description="Spatial detector bounding box confidence")
    extraction_confidence: float = Field(ge=0.0, le=1.0, description="Recognition/OCR engine confidence")
    method: str = Field(default="ocr", description="Extraction method used: 'ocr', 'recognizer', or 'hybrid'")
    confidence_tier: str = Field(default="MEDIUM", description="Reliability tier: 'HIGH', 'MEDIUM', or 'LOW'")
    bounding_box: Optional[BoundingBox] = Field(default=None, description="Coordinates of the detected field region")


class SignatureOutput(BaseModel):
    """Detection and verification status for signature fields."""

    present: bool = Field(default=False, description="Whether a valid signature stroke pattern was detected")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Detector confidence for signature presence")
    bounding_box: Optional[BoundingBox] = Field(default=None, description="Bounding box of signature region")


class PipelineDiagnostics(BaseModel):
    """Runtime telemetry and audit diagnostics."""

    processing_time_ms: float = Field(description="Total end-to-end pipeline latency in milliseconds")
    stage_latencies_ms: Dict[str, float] = Field(default_factory=dict, description="Per-stage latency breakdown")
    image_width: int
    image_height: int
    review_reasons: List[str] = Field(default_factory=list, description="Specific triggers requiring human teller review")


class ChequePipelineResult(BaseModel):
    """Complete structured JSON output of the ChequeSense inference pipeline."""

    cheque_id: str = Field(description="Unique identifier or filename of the processed cheque")
    status: str = Field(default="SUCCESS", description="Processing status: 'SUCCESS', 'PARTIAL', or 'ERROR'")
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Harmonic mean confidence across detected fields")
    review_required: bool = Field(default=False, description="Flag indicating whether human bank review is needed")
    fields: Dict[str, FieldOutput] = Field(default_factory=dict, description="Dictionary of extracted cheque fields")
    signatures: SignatureOutput = Field(default_factory=SignatureOutput, description="Signature detection status")
    diagnostics: Optional[PipelineDiagnostics] = Field(default=None, description="Audit telemetry")


class PipelineConfig(BaseModel):
    """Configuration settings and model checkpoint paths for the pipeline."""

    detector_model_path: str = "models/field_detector/best_model.pt"
    recognizer_model_path: str = "models/recognizer/best_model.pt"
    tesseract_cmd: Optional[str] = None
    tesseract_language: str = "eng"
    device: str = "cpu"
    detection_threshold: float = 0.50
    high_confidence_threshold: float = 0.80
    low_confidence_threshold: float = 0.60
    enable_clahe: bool = True
