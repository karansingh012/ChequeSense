"""Pydantic schemas and DTOs for ChequeSense FastAPI backend."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.analytics.metrics import (
    AmountMetrics,
    ConfidenceMetrics,
    ExecutiveSummary,
    FieldPerformanceMetrics,
    LatencyMetrics,
    ReviewMetrics,
    ValidationMetrics,
    VolumeMetrics,
)
from src.analytics.trends import AmountTrendPoint, VolumeTrendPoint


# ------------------------------------------------------------------------------
# Generic & Health Schemas
# ------------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """System health and operational status."""

    status: str = Field(description="Overall system health status")
    version: str = Field(description="API software version")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    database: str = Field(description="Database connectivity state")
    pipeline: str = Field(description="Inference pipeline availability state")

    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    """Standardized API error response payload."""

    error: str = Field(description="Error category or code")
    detail: str = Field(description="Detailed error description")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


# ------------------------------------------------------------------------------
# Cheque Schemas
# ------------------------------------------------------------------------------

class ChequeUploadResponse(BaseModel):
    """Metadata response returned immediately upon successful cheque image upload."""

    id: int = Field(description="Database primary key ID")
    cheque_identifier: str = Field(description="Unique cheque tracking reference identifier")
    image_path: str = Field(description="Storage URI / path to stored image file")
    image_width: int = Field(description="Image pixel width")
    image_height: int = Field(description="Image pixel height")
    image_hash: str = Field(description="Cryptographic SHA-256 digest of image bytes")
    status: str = Field(default="PROCESSED", description="Operational processing status")
    overall_confidence: float = Field(default=0.0, description="Overall extraction confidence")
    review_required: bool = Field(default=False, description="Flag indicating human review requirement")
    created_at: datetime.datetime = Field(description="Ingestion timestamp")

    model_config = ConfigDict(from_attributes=True)


class ExtractedFieldDetail(BaseModel):
    """Normalized field record ready for banking ledger posting."""

    id: int
    field_name: str
    raw_value: str
    normalized_value: str
    confidence: float
    detection_confidence: float
    extraction_confidence: float
    extraction_method: str
    confidence_tier: str
    bounding_box_json: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class ValidationResultDetail(BaseModel):
    """Validation rule outcome and compliance review result."""

    id: int
    field_name: Optional[str]
    check_type: str
    validation_status: str
    is_valid: bool
    validation_reason: str
    review_priority: str
    review_resolution: Optional[str]
    reviewed_at: Optional[datetime.datetime]

    model_config = ConfigDict(from_attributes=True)


class ProcessingRunDetail(BaseModel):
    """Audit log of pipeline execution pass."""

    id: int
    run_timestamp: datetime.datetime
    status: str
    detector_model_version: str
    recognizer_model_version: str
    ocr_engine_version: str
    total_latency_ms: float
    stage_latencies_json: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class ChequeDetailResponse(ChequeUploadResponse):
    """Comprehensive cheque detail record including associated fields, validation checks, and runs."""

    source_dataset: str = "production"
    updated_at: datetime.datetime
    extracted_fields: List[ExtractedFieldDetail] = Field(default_factory=list)
    validation_results: List[ValidationResultDetail] = Field(default_factory=list)
    processing_runs: List[ProcessingRunDetail] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ChequeListResponse(BaseModel):
    """Paginated list of cheques."""

    total: int = Field(description="Total count matching query filters")
    skip: int = Field(description="Offset of current page")
    limit: int = Field(description="Page size limit")
    items: List[ChequeDetailResponse] = Field(default_factory=list)


class ChequeProcessResponse(BaseModel):
    """Full execution output returned after processing a cheque through the ML pipeline."""

    cheque_id: int = Field(description="Database primary key ID")
    cheque_identifier: str = Field(description="Unique tracking identifier")
    status: str = Field(description="Final operational status (VERIFIED, REVIEW_REQUIRED, etc.)")
    overall_confidence: float = Field(description="Composite confidence across fields")
    review_required: bool = Field(description="Whether human teller review is triggered")
    fields: Dict[str, Any] = Field(description="Extracted field values and confidence scores")
    reasons_for_review: List[str] = Field(default_factory=list, description="List of review triggers")
    review_signals: List[Dict[str, Any]] = Field(
        default_factory=list, description="Statistical anomaly advisory signals (non-fraud)"
    )
    processing_time_ms: float = Field(description="Execution latency in milliseconds")
    processed_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


# ------------------------------------------------------------------------------
# Analytics Schemas
# ------------------------------------------------------------------------------

class AnalyticsTrendsResponse(BaseModel):
    """Time-series trends for volume, review rates, and monetary clearing."""

    interval: str = Field(description="Aggregation bucket interval: day, hour, or week")
    volume_trends: List[VolumeTrendPoint] = Field(default_factory=list)
    amount_trends: List[AmountTrendPoint] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
