"""SQLAlchemy normalized ORM models for ChequeSense."""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from src.database.connection import Base

# Compatibility JSON type (falls back to generic JSON on non-PostgreSQL engines)
JSONType = JSON().with_variant(JSONB, "postgresql")


class User(Base):
    """Banking system users, compliance officers, tellers, and service accounts."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="TELLER", nullable=False)  # TELLER, AUDITOR, ADMIN
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    cheques: Mapped[List[Cheque]] = relationship("Cheque", back_populates="creator")
    reviewed_validations: Mapped[List[ValidationResult]] = relationship(
        "ValidationResult", back_populates="reviewer"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class Cheque(Base):
    """Core cheque document entity with metadata and storage reference.

    Note: Large image binaries are intentionally NOT stored directly in the database.
    Only image file paths or object-storage references are stored.
    """

    __tablename__ = "cheques"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cheque_identifier: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)  # Storage path/URI reference
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    image_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # SHA-256 for deduplication
    source_dataset: Mapped[str] = mapped_column(String(64), default="production", nullable=False)
    
    # Audit & Status
    status: Mapped[str] = mapped_column(
        String(32), default="PROCESSED", nullable=False, index=True
    )  # PROCESSED, VERIFIED, REVIEW_REQUIRED, INVALID
    overall_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    review_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    created_by_user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    creator: Mapped[Optional[User]] = relationship("User", back_populates="cheques")
    processing_runs: Mapped[List[ProcessingRun]] = relationship(
        "ProcessingRun", back_populates="cheque", cascade="all, delete-orphan"
    )
    extracted_fields: Mapped[List[ExtractedField]] = relationship(
        "ExtractedField", back_populates="cheque", cascade="all, delete-orphan"
    )
    predictions: Mapped[List[Prediction]] = relationship(
        "Prediction", back_populates="cheque", cascade="all, delete-orphan"
    )
    validation_results: Mapped[List[ValidationResult]] = relationship(
        "ValidationResult", back_populates="cheque", cascade="all, delete-orphan"
    )

    # Compound Indexes for frequent banking query patterns
    __table_args__ = (
        Index("idx_cheques_status_review", "status", "review_required"),
        Index("idx_cheques_created_at_status", "created_at", "status"),
    )

    def __repr__(self) -> str:
        return f"<Cheque(id={self.id}, identifier='{self.cheque_identifier}', status='{self.status}')>"


class ProcessingRun(Base):
    """Audit log of individual pipeline execution passes on a cheque."""

    __tablename__ = "processing_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cheque_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cheques.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="SUCCESS", nullable=False)
    
    # Preserved Model & Tool Versions
    detector_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    recognizer_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    ocr_engine_version: Mapped[str] = mapped_column(String(128), nullable=False)
    
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    stage_latencies_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)

    # Relationships
    cheque: Mapped[Cheque] = relationship("Cheque", back_populates="processing_runs")
    predictions: Mapped[List[Prediction]] = relationship(
        "Prediction", back_populates="processing_run", cascade="all, delete-orphan"
    )
    extracted_fields: Mapped[List[ExtractedField]] = relationship(
        "ExtractedField", back_populates="processing_run", cascade="all, delete-orphan"
    )
    validation_results: Mapped[List[ValidationResult]] = relationship(
        "ValidationResult", back_populates="processing_run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ProcessingRun(id={self.id}, cheque_id={self.cheque_id}, timestamp='{self.run_timestamp}')>"


class Prediction(Base):
    """Raw predictions output directly from models before post-processing and business normalization."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    processing_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("processing_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    cheque_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cheques.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # FIELD_DETECTION, OCR_EXTRACTION, HANDWRITTEN_RECOGNITION
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    field_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    raw_output: Mapped[str] = mapped_column(Text, nullable=False)  # Preserved raw prediction
    raw_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    bounding_box_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    processing_run: Mapped[ProcessingRun] = relationship("ProcessingRun", back_populates="predictions")
    cheque: Mapped[Cheque] = relationship("Cheque", back_populates="predictions")

    __table_args__ = (
        Index("idx_predictions_run_field", "processing_run_id", "field_name"),
    )

    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, field='{self.field_name}', raw='{self.raw_output[:20]}')>"


class ExtractedField(Base):
    """Sanitized, normalized field values ready for banking ledger posting."""

    __tablename__ = "extracted_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cheque_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cheques.id", ondelete="CASCADE"), nullable=False, index=True
    )
    processing_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("processing_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    field_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # Preserved values
    raw_value: Mapped[str] = mapped_column(Text, nullable=False)  # Preserved raw input
    normalized_value: Mapped[str] = mapped_column(Text, nullable=False)  # Preserved leading zeros & format
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # Propagated confidence
    detection_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    extraction_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    extraction_method: Mapped[str] = mapped_column(String(32), default="ocr", nullable=False)  # ocr, recognizer, hybrid
    confidence_tier: Mapped[str] = mapped_column(String(16), default="MEDIUM", nullable=False, index=True)
    bounding_box_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    cheque: Mapped[Cheque] = relationship("Cheque", back_populates="extracted_fields")
    processing_run: Mapped[ProcessingRun] = relationship("ProcessingRun", back_populates="extracted_fields")

    __table_args__ = (
        Index("idx_extracted_fields_cheque_name", "cheque_id", "field_name"),
    )

    def __repr__(self) -> str:
        return f"<ExtractedField(id={self.id}, field='{self.field_name}', value='{self.normalized_value}')>"


class ValidationResult(Base):
    """Validation outcomes, business rule checks, and teller review records."""

    __tablename__ = "validation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cheque_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cheques.id", ondelete="CASCADE"), nullable=False, index=True
    )
    processing_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("processing_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    field_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    check_type: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )  # FORMAT, DATE_VALIDITY, CONSISTENCY, REQUIRED_FIELD, CONFIDENCE_GATE
    
    validation_status: Mapped[str] = mapped_column(
        String(32), default="VALID", nullable=False, index=True
    )  # VALID, INVALID, WARNING
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    validation_reason: Mapped[str] = mapped_column(Text, nullable=False)  # Preserved reason for review or failure
    review_priority: Mapped[str] = mapped_column(String(16), default="MEDIUM", nullable=False)  # HIGH, MEDIUM, LOW
    
    # Human Teller Review Resolution
    reviewed_by_user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    review_resolution: Mapped[Optional[str]] = mapped_column(String(32), default="PENDING", nullable=True)  # PENDING, ACCEPTED, REJECTED, CORRECTED
    reviewed_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    cheque: Mapped[Cheque] = relationship("Cheque", back_populates="validation_results")
    processing_run: Mapped[ProcessingRun] = relationship("ProcessingRun", back_populates="validation_results")
    reviewer: Mapped[Optional[User]] = relationship("User", back_populates="reviewed_validations")

    __table_args__ = (
        Index("idx_validation_cheque_status", "cheque_id", "validation_status"),
    )

    def __repr__(self) -> str:
        return f"<ValidationResult(id={self.id}, check='{self.check_type}', status='{self.validation_status}')>"


class AuditLog(Base):
    """Immutable audit trail for compliance, security events, and operator actions."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    username: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )  # LOGIN, UPLOAD_CHEQUE, PROCESS_CHEQUE, MANUAL_CORRECTION, STATUS_CHANGE
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)  # user, cheque, validation_result, auth
    resource_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    details_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)

    # Relationships
    user: Mapped[Optional[User]] = relationship("User")

    __table_args__ = (
        Index("idx_audit_logs_action_timestamp", "action", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action='{self.action}', user='{self.username}', time='{self.timestamp}')>"

