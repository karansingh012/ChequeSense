"""CRUD (Create, Read, Update, Delete) repository layer for ChequeSense database."""

from __future__ import annotations

import datetime
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import desc, func, select, update
from sqlalchemy.orm import Session, selectinload

from src.database.models import (
    Cheque,
    ExtractedField,
    Prediction,
    ProcessingRun,
    User,
    ValidationResult,
)
from src.database.schemas import (
    ChequeCreate,
    ExtractedFieldCreate,
    PredictionCreate,
    ProcessingRunCreate,
    UserCreate,
    ValidationResultCreate,
)

logger = logging.getLogger("chequesense.database.crud")


# ------------------------------------------------------------------------------
# User CRUD
# ------------------------------------------------------------------------------

def create_user(db: Session, user_in: UserCreate) -> User:
    """Creates a new user record with hashed password."""
    # Simple SHA-256 for basic authentication demonstration
    hashed = hashlib.sha256(user_in.password.encode("utf-8")).hexdigest()
    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hashed,
        role=user_in.role,
        is_active=user_in.is_active,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.get(User, user_id)


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    stmt = select(User).where(User.username == username)
    return db.scalars(stmt).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    stmt = select(User).where(User.email == email)
    return db.scalars(stmt).first()


def list_users(db: Session, skip: int = 0, limit: int = 50) -> List[User]:
    stmt = select(User).offset(skip).limit(limit)
    return list(db.scalars(stmt).all())


# ------------------------------------------------------------------------------
# Cheque CRUD
# ------------------------------------------------------------------------------

def create_cheque(db: Session, cheque_in: ChequeCreate) -> Cheque:
    """Creates a new cheque document record."""
    cheque = Cheque(
        cheque_identifier=cheque_in.cheque_identifier,
        image_path=cheque_in.image_path,
        image_width=cheque_in.image_width,
        image_height=cheque_in.image_height,
        image_hash=cheque_in.image_hash,
        source_dataset=cheque_in.source_dataset,
        status=cheque_in.status,
        overall_confidence=cheque_in.overall_confidence,
        review_required=cheque_in.review_required,
        created_by_user_id=cheque_in.created_by_user_id,
    )
    db.add(cheque)
    db.commit()
    db.refresh(cheque)
    return cheque


def get_cheque_by_id(db: Session, cheque_id: int, include_relations: bool = False) -> Optional[Cheque]:
    """Retrieves a cheque by primary key ID, optionally eager-loading children."""
    if not include_relations:
        return db.get(Cheque, cheque_id)

    stmt = (
        select(Cheque)
        .where(Cheque.id == cheque_id)
        .options(
            selectinload(Cheque.extracted_fields),
            selectinload(Cheque.validation_results),
            selectinload(Cheque.processing_runs),
        )
    )
    return db.scalars(stmt).first()


def get_cheque_by_identifier(db: Session, identifier: str) -> Optional[Cheque]:
    stmt = select(Cheque).where(Cheque.cheque_identifier == identifier)
    return db.scalars(stmt).first()


def get_cheque_by_hash(db: Session, image_hash: str) -> Optional[Cheque]:
    stmt = select(Cheque).where(Cheque.image_hash == image_hash)
    return db.scalars(stmt).first()


def list_cheques(
    db: Session,
    status: Optional[str] = None,
    review_required: Optional[bool] = None,
    skip: int = 0,
    limit: int = 50,
) -> List[Cheque]:
    """Queries cheques with indexed filtering on status and review requirements."""
    stmt = select(Cheque)
    if status is not None:
        stmt = stmt.where(Cheque.status == status)
    if review_required is not None:
        stmt = stmt.where(Cheque.review_required == review_required)

    stmt = stmt.order_by(desc(Cheque.created_at)).offset(skip).limit(limit)
    return list(db.scalars(stmt).all())


def update_cheque_status(
    db: Session,
    cheque_id: int,
    status: str,
    overall_confidence: Optional[float] = None,
    review_required: Optional[bool] = None,
) -> Optional[Cheque]:
    """Updates the operational processing status of a cheque."""
    cheque = db.get(Cheque, cheque_id)
    if not cheque:
        return None

    cheque.status = status
    if overall_confidence is not None:
        cheque.overall_confidence = overall_confidence
    if review_required is not None:
        cheque.review_required = review_required

    cheque.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(cheque)
    return cheque


# ------------------------------------------------------------------------------
# Processing Run CRUD
# ------------------------------------------------------------------------------

def create_processing_run(db: Session, run_in: ProcessingRunCreate) -> ProcessingRun:
    """Logs a new pipeline execution run."""
    run = ProcessingRun(
        cheque_id=run_in.cheque_id,
        status=run_in.status,
        detector_model_version=run_in.detector_model_version,
        recognizer_model_version=run_in.recognizer_model_version,
        ocr_engine_version=run_in.ocr_engine_version,
        total_latency_ms=run_in.total_latency_ms,
        stage_latencies_json=run_in.stage_latencies_json,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_processing_run_by_id(db: Session, run_id: int) -> Optional[ProcessingRun]:
    return db.get(ProcessingRun, run_id)


def list_runs_for_cheque(db: Session, cheque_id: int) -> List[ProcessingRun]:
    stmt = (
        select(ProcessingRun)
        .where(ProcessingRun.cheque_id == cheque_id)
        .order_by(desc(ProcessingRun.run_timestamp))
    )
    return list(db.scalars(stmt).all())


# ------------------------------------------------------------------------------
# Prediction CRUD
# ------------------------------------------------------------------------------

def bulk_create_predictions(db: Session, predictions: List[PredictionCreate]) -> List[Prediction]:
    """Persists a batch of raw model outputs in a single transaction."""
    records = [
        Prediction(
            processing_run_id=p.processing_run_id,
            cheque_id=p.cheque_id,
            stage=p.stage,
            model_name=p.model_name,
            model_version=p.model_version,
            field_name=p.field_name,
            raw_output=p.raw_output,
            raw_confidence=p.raw_confidence,
            bounding_box_json=p.bounding_box_json,
        )
        for p in predictions
    ]
    db.add_all(records)
    db.commit()
    return records


# ------------------------------------------------------------------------------
# Extracted Field CRUD
# ------------------------------------------------------------------------------

def bulk_create_extracted_fields(db: Session, fields: List[ExtractedFieldCreate]) -> List[ExtractedField]:
    """Persists a batch of sanitized, normalized banking field records."""
    records = [
        ExtractedField(
            cheque_id=f.cheque_id,
            processing_run_id=f.processing_run_id,
            field_name=f.field_name,
            raw_value=f.raw_value,
            normalized_value=f.normalized_value,
            confidence=f.confidence,
            detection_confidence=f.detection_confidence,
            extraction_confidence=f.extraction_confidence,
            extraction_method=f.extraction_method,
            confidence_tier=f.confidence_tier,
            bounding_box_json=f.bounding_box_json,
        )
        for f in fields
    ]
    db.add_all(records)
    db.commit()
    return records


def get_fields_for_cheque(db: Session, cheque_id: int) -> List[ExtractedField]:
    stmt = (
        select(ExtractedField)
        .where(ExtractedField.cheque_id == cheque_id)
        .order_by(ExtractedField.field_name)
    )
    return list(db.scalars(stmt).all())


# ------------------------------------------------------------------------------
# Validation Result CRUD
# ------------------------------------------------------------------------------

def bulk_create_validation_results(db: Session, results: List[ValidationResultCreate]) -> List[ValidationResult]:
    """Persists a batch of validation and compliance check records."""
    records = [
        ValidationResult(
            cheque_id=r.cheque_id,
            processing_run_id=r.processing_run_id,
            field_name=r.field_name,
            check_type=r.check_type,
            validation_status=r.validation_status,
            is_valid=r.is_valid,
            validation_reason=r.validation_reason,
            review_priority=r.review_priority,
            reviewed_by_user_id=r.reviewed_by_user_id,
            review_resolution=r.review_resolution,
            reviewed_at=r.reviewed_at,
        )
        for r in results
    ]
    db.add_all(records)
    db.commit()
    return records


def update_review_resolution(
    db: Session,
    validation_id: int,
    reviewed_by_user_id: int,
    resolution: str,  # 'ACCEPTED', 'REJECTED', 'CORRECTED'
) -> Optional[ValidationResult]:
    """Records human teller review resolution on a validation check."""
    record = db.get(ValidationResult, validation_id)
    if not record:
        return None

    record.reviewed_by_user_id = reviewed_by_user_id
    record.review_resolution = resolution
    record.reviewed_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(record)
    return record


# ------------------------------------------------------------------------------
# Integrated Atomic Pipeline Ingestion
# ------------------------------------------------------------------------------

def save_pipeline_execution(
    db: Session,
    cheque_identifier: str,
    image_path: str,
    image_width: int,
    image_height: int,
    image_hash: str,
    pipeline_result: Any,
    review_item: Optional[Any] = None,
    detector_version: str = "fasterrcnn_mobilenet_v3_large_fpn:v1.0",
    recognizer_version: str = "cheque_digit_cnn:v1.0",
    ocr_version: str = "tesseract:5.5.1",
    source_dataset: str = "production",
) -> Cheque:
    """Atomically persists complete end-to-end pipeline execution into normalized tables.

    Preserves:
    - Cheque image storage path
    - Processing run timestamp & latency
    - Raw predictions
    - Normalized fields with preserved leading zeros
    - Propagated confidence scores
    - Validation results & review reasons
    """
    # 1. Upsert Cheque document entity
    cheque = get_cheque_by_identifier(db, cheque_identifier)
    status_val = getattr(review_item, "status", getattr(pipeline_result, "status", "PROCESSED"))
    if hasattr(status_val, "value"):
        status_val = status_val.value

    overall_conf = float(getattr(pipeline_result, "overall_confidence", 0.0))
    review_req = bool(getattr(review_item, "review_required", getattr(pipeline_result, "review_required", False)))

    if not cheque:
        cheque = Cheque(
            cheque_identifier=cheque_identifier,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            image_hash=image_hash,
            source_dataset=source_dataset,
            status=status_val,
            overall_confidence=overall_conf,
            review_required=review_req,
        )
        db.add(cheque)
        db.flush()
    else:
        cheque.status = status_val
        cheque.overall_confidence = overall_conf
        cheque.review_required = review_req
        cheque.updated_at = datetime.datetime.utcnow()
        db.flush()

    # 2. Record ProcessingRun
    diag = getattr(pipeline_result, "diagnostics", None)
    total_time = getattr(diag, "processing_time_ms", 0.0) if diag else 0.0
    stage_lat = getattr(diag, "stage_latencies_ms", {}) if diag else {}

    run = ProcessingRun(
        cheque_id=cheque.id,
        status=getattr(pipeline_result, "status", "SUCCESS"),
        detector_model_version=detector_version,
        recognizer_model_version=recognizer_version,
        ocr_engine_version=ocr_version,
        total_latency_ms=total_time,
        stage_latencies_json=stage_lat,
    )
    db.add(run)
    db.flush()

    # 3. Save Extracted Fields & Raw Predictions
    fields_dict = getattr(pipeline_result, "fields", {})
    if isinstance(fields_dict, dict):
        for fname, f_out in fields_dict.items():
            val = getattr(f_out, "value", str(f_out or ""))
            raw = getattr(f_out, "raw_value", val)
            conf = float(getattr(f_out, "confidence", 0.0))
            det_conf = float(getattr(f_out, "detection_confidence", 1.0))
            ext_conf = float(getattr(f_out, "extraction_confidence", 1.0))
            method = getattr(f_out, "method", "ocr")
            tier = getattr(f_out, "confidence_tier", "MEDIUM")
            bbox = getattr(f_out, "bounding_box", None)
            bbox_dict = bbox.model_dump() if hasattr(bbox, "model_dump") else (bbox if isinstance(bbox, dict) else None)

            # Raw Prediction
            pred = Prediction(
                processing_run_id=run.id,
                cheque_id=cheque.id,
                stage=f"{method.upper()}_EXTRACTION",
                model_name=ocr_version if method == "ocr" else recognizer_version,
                model_version="1.0",
                field_name=fname,
                raw_output=raw,
                raw_confidence=ext_conf,
                bounding_box_json=bbox_dict,
            )
            db.add(pred)

            # Extracted Normalized Field
            ef = ExtractedField(
                cheque_id=cheque.id,
                processing_run_id=run.id,
                field_name=fname,
                raw_value=raw,
                normalized_value=val,
                confidence=conf,
                detection_confidence=det_conf,
                extraction_confidence=ext_conf,
                extraction_method=method,
                confidence_tier=tier,
                bounding_box_json=bbox_dict,
            )
            db.add(ef)

    # 4. Save Validation Results & Reasons for Review
    if review_item is not None:
        reasons = getattr(review_item, "reasons_for_review", [])
        priority = getattr(review_item, "review_priority", "MEDIUM")

        if not reasons and status_val == "VERIFIED":
            vr = ValidationResult(
                cheque_id=cheque.id,
                processing_run_id=run.id,
                field_name=None,
                check_type="ALL_GATES",
                validation_status="VALID",
                is_valid=True,
                validation_reason="All field formats, confidence gates, and consistency checks passed successfully",
                review_priority="LOW",
                review_resolution="ACCEPTED",
            )
            db.add(vr)
        else:
            for r in reasons:
                vr = ValidationResult(
                    cheque_id=cheque.id,
                    processing_run_id=run.id,
                    field_name=None,
                    check_type="COMPLIANCE_GATE",
                    validation_status="WARNING" if status_val == "REVIEW_REQUIRED" else "INVALID",
                    is_valid=False,
                    validation_reason=r,
                    review_priority=priority,
                    review_resolution="PENDING",
                )
                db.add(vr)

    db.commit()
    db.refresh(cheque)
    logger.info("Persisted pipeline execution for cheque '%s' (DB ID: %d)", cheque.cheque_identifier, cheque.id)
    return cheque
