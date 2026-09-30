"""Pydantic schemas for database DTOs and API serialization."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ------------------------------------------------------------------------------
# User Schemas
# ------------------------------------------------------------------------------

class UserBase(BaseModel):
    username: str
    email: str
    role: str = "TELLER"
    is_active: bool = True


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Extracted Field Schemas
# ------------------------------------------------------------------------------

class ExtractedFieldBase(BaseModel):
    field_name: str
    raw_value: str
    normalized_value: str
    confidence: float
    detection_confidence: float = 0.0
    extraction_confidence: float = 0.0
    extraction_method: str = "ocr"
    confidence_tier: str = "MEDIUM"
    bounding_box_json: Optional[Dict[str, Any]] = None


class ExtractedFieldCreate(ExtractedFieldBase):
    cheque_id: int
    processing_run_id: int


class ExtractedFieldResponse(ExtractedFieldBase):
    id: int
    cheque_id: int
    processing_run_id: int
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Prediction Schemas
# ------------------------------------------------------------------------------

class PredictionBase(BaseModel):
    stage: str
    model_name: str
    model_version: str
    field_name: str
    raw_output: str
    raw_confidence: float = 0.0
    bounding_box_json: Optional[Dict[str, Any]] = None


class PredictionCreate(PredictionBase):
    cheque_id: int
    processing_run_id: int


class PredictionResponse(PredictionBase):
    id: int
    cheque_id: int
    processing_run_id: int
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Validation Result Schemas
# ------------------------------------------------------------------------------

class ValidationResultBase(BaseModel):
    field_name: Optional[str] = None
    check_type: str
    validation_status: str
    is_valid: bool
    validation_reason: str
    review_priority: str = "MEDIUM"
    reviewed_by_user_id: Optional[int] = None
    review_resolution: Optional[str] = "PENDING"
    reviewed_at: Optional[datetime.datetime] = None


class ValidationResultCreate(ValidationResultBase):
    cheque_id: int
    processing_run_id: int


class ValidationResultResponse(ValidationResultBase):
    id: int
    cheque_id: int
    processing_run_id: int
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Processing Run Schemas
# ------------------------------------------------------------------------------

class ProcessingRunBase(BaseModel):
    status: str = "SUCCESS"
    detector_model_version: str
    recognizer_model_version: str
    ocr_engine_version: str
    total_latency_ms: float = 0.0
    stage_latencies_json: Optional[Dict[str, Any]] = None


class ProcessingRunCreate(ProcessingRunBase):
    cheque_id: int


class ProcessingRunResponse(ProcessingRunBase):
    id: int
    cheque_id: int
    run_timestamp: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Cheque Schemas
# ------------------------------------------------------------------------------

class ChequeBase(BaseModel):
    cheque_identifier: str
    image_path: str
    image_width: int
    image_height: int
    image_hash: str
    source_dataset: str = "production"
    status: str = "PROCESSED"
    overall_confidence: float = 0.0
    review_required: bool = False


class ChequeCreate(ChequeBase):
    created_by_user_id: Optional[int] = None


class ChequeResponse(ChequeBase):
    id: int
    created_by_user_id: Optional[int]
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class ChequeDetailResponse(ChequeResponse):
    """Rich detail view including associated fields, validations, and processing history."""

    extracted_fields: List[ExtractedFieldResponse] = Field(default_factory=list)
    validation_results: List[ValidationResultResponse] = Field(default_factory=list)
    processing_runs: List[ProcessingRunResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
