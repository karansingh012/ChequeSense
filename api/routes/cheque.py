"""Cheque ingestion, inference processing, and detail query endpoints."""

from __future__ import annotations

import hashlib
import io
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Path as FastPath,
    Query,
    UploadFile,
    status,
)
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.dependencies import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE_BYTES,
    get_db,
    get_pipeline,
    get_review_queue_manager,
    get_upload_dir,
)
from api.schemas import (
    ChequeDetailResponse,
    ChequeListResponse,
    ChequeProcessResponse,
    ChequeUploadResponse,
    ErrorResponse,
)
from src.analytics.trends import detect_amount_anomaly_signal
from src.database.crud import (
    create_cheque,
    get_cheque_by_hash,
    get_cheque_by_id,
    list_cheques,
    save_pipeline_execution,
)
from src.database.models import Cheque
from src.database.schemas import ChequeCreate
from src.pipeline.pipeline import ChequeInferencePipeline
from src.validation.review_queue import ReviewQueueManager

logger = logging.getLogger("chequesense.api.cheque")

router = APIRouter(prefix="/cheques", tags=["Cheques"])


# ------------------------------------------------------------------------------
# 1. Cheque Upload Endpoint
# ------------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=ChequeUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new cheque image",
    description="Validates image format and file size, generates a safe filename, stores the image securely, and creates a database record.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format or corrupted image"},
        413: {"model": ErrorResponse, "description": "Payload exceeds 15MB limit"},
    },
)
async def upload_cheque(
    file: UploadFile = File(..., description="Cheque image file (PNG, JPEG, TIFF, WEBP)"),
    db: Session = Depends(get_db),
    upload_dir: Path = Depends(get_upload_dir),
) -> ChequeUploadResponse:
    # 1. Validate File Extension
    orig_name = file.filename or "cheque.png"
    ext = Path(orig_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Read contents and enforce size limit
    contents = await file.read()
    file_size = len(contents)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({file_size / (1024 * 1024):.1f}MB) exceeds the maximum allowed limit of 15MB.",
        )

    # 3. Validate image integrity and dimensions using PIL
    try:
        with Image.open(io.BytesIO(contents)) as img:
            img.verify()
        with Image.open(io.BytesIO(contents)) as img:
            width, height = img.size
    except Exception as e:
        logger.warning("Corrupted or invalid image upload attempt: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image data: {str(e)}",
        )

    # 4. Compute cryptographic SHA-256 hash for deduplication
    image_hash = hashlib.sha256(contents).hexdigest()

    # 5. Check if duplicate cheque already exists
    existing = get_cheque_by_hash(db, image_hash)
    if existing:
        logger.info("Duplicate cheque image detected via hash %s (ID: %d)", image_hash[:16], existing.id)
        return ChequeUploadResponse.model_validate(existing)

    # 6. Generate safe filename avoiding path traversal
    safe_filename = f"{uuid.uuid4().hex[:12]}_{image_hash[:16]}{ext}"
    dest_path = upload_dir / safe_filename
    with open(dest_path, "wb") as f:
        f.write(contents)

    # 7. Create database record
    cheque_id_code = f"CHQ_{uuid.uuid4().hex[:8].upper()}"
    cheque_in = ChequeCreate(
        cheque_identifier=cheque_id_code,
        image_path=str(dest_path),
        image_width=width,
        image_height=height,
        image_hash=image_hash,
        source_dataset="production_upload",
        status="PROCESSED",
        overall_confidence=0.0,
        review_required=False,
    )
    cheque = create_cheque(db, cheque_in)
    logger.info("Successfully ingested cheque '%s' (DB ID: %d)", cheque.cheque_identifier, cheque.id)

    return ChequeUploadResponse.model_validate(cheque)


# ------------------------------------------------------------------------------
# 2. Cheque Processing Endpoint
# ------------------------------------------------------------------------------

@router.post(
    "/{id}/process",
    response_model=ChequeProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Process cheque through the inference pipeline",
    description="Invokes field detection, handwritten digit recognition, OCR extraction, format validation, and review gating.",
    responses={
        404: {"model": ErrorResponse, "description": "Cheque ID not found or image missing on disk"},
    },
)
def process_cheque(
    id: int = FastPath(..., ge=1, description="Database ID of the cheque to process"),
    db: Session = Depends(get_db),
    pipeline: ChequeInferencePipeline = Depends(get_pipeline),
    review_queue: ReviewQueueManager = Depends(get_review_queue_manager),
) -> ChequeProcessResponse:
    # 1. Fetch cheque from database
    cheque = get_cheque_by_id(db, id)
    if not cheque:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cheque with ID {id} does not exist.",
        )

    # 2. Verify image file exists on disk
    img_file = Path(cheque.image_path)
    if not img_file.exists():
        logger.error("Cheque image missing on storage: %s", cheque.image_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cheque image file not found on storage: '{cheque.image_path}'",
        )

    # 3. Run Inference Pipeline
    try:
        pipeline_result = pipeline.process(cheque.image_path, cheque_id=cheque.cheque_identifier)
    except Exception as e:
        logger.exception("Pipeline execution failed for cheque %d: %s", id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ML Pipeline execution failure: {str(e)}",
        )

    # 4. Run Review Queue Compliance Evaluation
    review_item = review_queue.evaluate_cheque(pipeline_result)

    # 5. Atomically Persist Execution Results into Database
    saved_cheque = save_pipeline_execution(
        db=db,
        cheque_identifier=cheque.cheque_identifier,
        image_path=cheque.image_path,
        image_width=cheque.image_width,
        image_height=cheque.image_height,
        image_hash=cheque.image_hash,
        pipeline_result=pipeline_result,
        review_item=review_item,
        detector_version="fasterrcnn_mobilenet_v3_large_fpn:v1.0",
        recognizer_version="cheque_digit_cnn:v1.0",
        ocr_version="tesseract:5.5.1",
        source_dataset=cheque.source_dataset,
    )

    # 6. Evaluate Statistical Outlier Review Signals (Non-fraud)
    review_signals: List[Dict[str, Any]] = []
    amount_field = pipeline_result.fields.get("amount")
    if amount_field and amount_field.value:
        try:
            clean_amt = "".join(ch for ch in amount_field.value if ch.isdigit() or ch == ".")
            if clean_amt:
                amt_val = float(clean_amt)
                sig = detect_amount_anomaly_signal(db, candidate_amount=amt_val)
                if sig:
                    review_signals.append(sig.model_dump())
        except ValueError:
            pass

    # 7. Build Response Payload
    field_summary = {
        fname: {
            "value": f.value,
            "raw_value": f.raw_value,
            "confidence": f.confidence,
            "confidence_tier": f.confidence_tier,
            "method": f.method,
        }
        for fname, f in pipeline_result.fields.items()
    }

    latency = pipeline_result.diagnostics.processing_time_ms if pipeline_result.diagnostics else 0.0

    return ChequeProcessResponse(
        cheque_id=saved_cheque.id,
        cheque_identifier=saved_cheque.cheque_identifier,
        status=saved_cheque.status,
        overall_confidence=saved_cheque.overall_confidence,
        review_required=saved_cheque.review_required,
        fields=field_summary,
        reasons_for_review=review_item.reasons_for_review,
        review_signals=review_signals,
        processing_time_ms=latency,
        processed_at=saved_cheque.updated_at,
    )


# ------------------------------------------------------------------------------
# 3. Get Cheque Details Endpoint
# ------------------------------------------------------------------------------

@router.get(
    "/{id}",
    response_model=ChequeDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get full cheque details",
    description="Retrieves a cheque document by ID with eager-loaded extracted fields, validation results, and execution runs.",
    responses={
        404: {"model": ErrorResponse, "description": "Cheque ID not found"},
    },
)
def get_cheque(
    id: int = FastPath(..., ge=1, description="Database ID of the cheque"),
    db: Session = Depends(get_db),
) -> ChequeDetailResponse:
    cheque = get_cheque_by_id(db, id, include_relations=True)
    if not cheque:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cheque with ID {id} does not exist.",
        )
    return ChequeDetailResponse.model_validate(cheque)


# ------------------------------------------------------------------------------
# 4. List Cheques Endpoint
# ------------------------------------------------------------------------------

@router.get(
    "",
    response_model=ChequeListResponse,
    status_code=status.HTTP_200_OK,
    summary="List processed cheques with filtering",
    description="Queries cheques with optional filtering by operational status and manual review requirement.",
)
def list_all_cheques(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status (PROCESSED, VERIFIED, REVIEW_REQUIRED, INVALID)"),
    review_required: Optional[bool] = Query(None, description="Filter by manual review flag"),
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50, ge=1, le=100, description="Page size limit"),
    db: Session = Depends(get_db),
) -> ChequeListResponse:
    # Query items with relations for complete view
    items = list_cheques(
        db=db,
        status=status_filter,
        review_required=review_required,
        skip=skip,
        limit=limit,
    )

    # Compute total count
    count_stmt = select(func.count(Cheque.id))
    if status_filter:
        count_stmt = count_stmt.where(Cheque.status == status_filter)
    if review_required is not None:
        count_stmt = count_stmt.where(Cheque.review_required == review_required)
    total_count = db.scalar(count_stmt) or 0

    # Ensure relations are hydrated for response models
    validated_items = []
    for item in items:
        # Load relations if needed
        hydrated = get_cheque_by_id(db, item.id, include_relations=True)
        if hydrated:
            validated_items.append(ChequeDetailResponse.model_validate(hydrated))

    return ChequeListResponse(
        total=total_count,
        skip=skip,
        limit=limit,
        items=validated_items,
    )
