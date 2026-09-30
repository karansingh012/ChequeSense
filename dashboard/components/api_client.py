"""Backend communication client for ChequeSense Streamlit dashboard.

Strictly encapsulates communication with the FastAPI backend and service layer,
ensuring zero ML logic is embedded directly inside UI code.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image

from src.database.connection import db_session_scope
from src.database.crud import (
    get_cheque_by_id,
    list_cheques,
    update_cheque_status,
    update_review_resolution,
)
from src.database.models import Cheque, ExtractedField, ValidationResult
from src.analytics.metrics import calculate_executive_dashboard
from src.analytics.trends import calculate_amount_trend, calculate_volume_trend
from src.pipeline.pipeline import ChequeInferencePipeline
from src.validation.review_queue import ReviewQueueManager

logger = logging.getLogger("chequesense.dashboard.client")

DEFAULT_API_URL = os.getenv("CHEQUESENSE_API_URL", "http://127.0.0.1:8000/api/v1")


class BackendClient:
    """Client proxy for interacting with ChequeSense backend services."""

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http_available: Optional[bool] = None
        self._token: Optional[str] = None

    def _get_headers(self) -> Dict[str, str]:
        """Returns HTTP authorization headers, automatically acquiring token if needed."""
        headers: Dict[str, str] = {}
        token = os.getenv("CHEQUESENSE_API_TOKEN") or self._token
        if not token:
            user = os.getenv("CHEQUESENSE_API_USER", "admin")
            password = os.getenv("CHEQUESENSE_API_PASSWORD", "AdminPassword123!")
            try:
                # Login endpoint is at /auth/login or /api/v1/auth/login
                auth_root = self.base_url.replace("/api/v1", "")
                login_url = f"{auth_root}/auth/login"
                with httpx.Client(timeout=5.0) as client:
                    resp = client.post(
                        login_url,
                        json={"username": user, "password": password},
                    )
                    if resp.status_code == 200:
                        token = resp.json().get("access_token")
                        self._token = token
            except Exception as e:
                logger.debug("Failed to acquire dashboard service token: %s", e)

        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def is_http_backend_available(self) -> bool:
        """Pings the FastAPI health endpoint to check if HTTP service is active."""
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/health")
                self._http_available = (res.status_code == 200)
                return self._http_available
        except Exception:
            self._http_available = False
            return False

    # --------------------------------------------------------------------------
    # Cheque Operations
    # --------------------------------------------------------------------------

    def upload_cheque(self, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """Uploads a cheque image file to the backend."""
        if self.is_http_backend_available():
            try:
                files = {"file": (filename, file_bytes, "image/png")}
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/cheques/upload",
                        files=files,
                        headers=self._get_headers(),
                    )
                    resp.raise_for_status()
                    return resp.json()
            except Exception as e:
                logger.warning("HTTP upload failed, falling back to service layer: %s", e)

        # Service-layer fallback
        from api.routes.cheque import upload_cheque as service_upload
        from api.dependencies import get_upload_dir
        import hashlib, uuid
        from src.database.schemas import ChequeCreate
        from src.database.crud import create_cheque, get_cheque_by_hash

        upload_dir = get_upload_dir()
        img_hash = hashlib.sha256(file_bytes).hexdigest()
        ext = Path(filename).suffix.lower() or ".png"

        with db_session_scope() as session:
            existing = get_cheque_by_hash(session, img_hash)
            if existing:
                return {
                    "id": existing.id,
                    "cheque_identifier": existing.cheque_identifier,
                    "image_path": existing.image_path,
                    "image_width": existing.image_width,
                    "image_height": existing.image_height,
                    "image_hash": existing.image_hash,
                    "status": existing.status,
                    "overall_confidence": existing.overall_confidence,
                    "review_required": existing.review_required,
                    "created_at": existing.created_at.isoformat() if existing.created_at else "",
                }

            with Image.open(io.BytesIO(file_bytes)) as img:
                width, height = img.size

            safe_filename = f"{uuid.uuid4().hex[:12]}_{img_hash[:16]}{ext}"
            dest = upload_dir / safe_filename
            with open(dest, "wb") as f:
                f.write(file_bytes)

            cheque_in = ChequeCreate(
                cheque_identifier=f"CHQ_{uuid.uuid4().hex[:8].upper()}",
                image_path=str(dest),
                image_width=width,
                image_height=height,
                image_hash=img_hash,
                source_dataset="dashboard_upload",
                status="PROCESSED",
                overall_confidence=0.0,
                review_required=False,
            )
            chq = create_cheque(session, cheque_in)
            return {
                "id": chq.id,
                "cheque_identifier": chq.cheque_identifier,
                "image_path": chq.image_path,
                "image_width": chq.image_width,
                "image_height": chq.image_height,
                "image_hash": chq.image_hash,
                "status": chq.status,
                "overall_confidence": chq.overall_confidence,
                "review_required": chq.review_required,
                "created_at": chq.created_at.isoformat() if chq.created_at else "",
            }

    def process_cheque(self, cheque_id: int) -> Dict[str, Any]:
        """Triggers ML inference and review gate evaluation on a cheque."""
        if self.is_http_backend_available():
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/cheques/{cheque_id}/process",
                        headers=self._get_headers(),
                    )
                    resp.raise_for_status()
                    return resp.json()
            except Exception as e:
                logger.warning("HTTP process failed, falling back to service layer: %s", e)

        # Service-layer fallback
        from src.database.crud import save_pipeline_execution
        with db_session_scope() as session:
            chq = get_cheque_by_id(session, cheque_id)
            if not chq:
                raise ValueError(f"Cheque {cheque_id} not found")

            pipeline = ChequeInferencePipeline()
            queue_mgr = ReviewQueueManager()

            pipe_res = pipeline.process(chq.image_path, cheque_id=chq.cheque_identifier)
            review_item = queue_mgr.evaluate_cheque(pipe_res)

            saved = save_pipeline_execution(
                db=session,
                cheque_identifier=chq.cheque_identifier,
                image_path=chq.image_path,
                image_width=chq.image_width,
                image_height=chq.image_height,
                image_hash=chq.image_hash,
                pipeline_result=pipe_res,
                review_item=review_item,
                source_dataset=chq.source_dataset,
            )

            field_summary = {
                fname: {
                    "value": f.value,
                    "raw_value": f.raw_value,
                    "confidence": f.confidence,
                    "confidence_tier": f.confidence_tier,
                    "method": f.method,
                }
                for fname, f in pipe_res.fields.items()
            }

            return {
                "cheque_id": saved.id,
                "cheque_identifier": saved.cheque_identifier,
                "status": saved.status,
                "overall_confidence": saved.overall_confidence,
                "review_required": saved.review_required,
                "fields": field_summary,
                "reasons_for_review": review_item.reasons_for_review,
                "review_signals": [],
                "processing_time_ms": pipe_res.diagnostics.processing_time_ms if pipe_res.diagnostics else 0.0,
                "processed_at": saved.updated_at.isoformat() if saved.updated_at else "",
            }

    def get_cheque(self, cheque_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves full cheque details, fields, validation results, and history."""
        if self.is_http_backend_available():
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(f"{self.base_url}/cheques/{cheque_id}")
                    if resp.status_code == 200:
                        return resp.json()
                    return None
            except Exception:
                pass

        # Service-layer fallback
        with db_session_scope() as session:
            chq = get_cheque_by_id(session, cheque_id, include_relations=True)
            if not chq:
                return None

            fields = [
                {
                    "id": f.id,
                    "field_name": f.field_name,
                    "raw_value": f.raw_value,
                    "normalized_value": f.normalized_value,
                    "confidence": f.confidence,
                    "detection_confidence": f.detection_confidence,
                    "extraction_confidence": f.extraction_confidence,
                    "extraction_method": f.extraction_method,
                    "confidence_tier": f.confidence_tier,
                    "bounding_box_json": f.bounding_box_json,
                }
                for f in chq.extracted_fields
            ]

            validations = [
                {
                    "id": v.id,
                    "field_name": v.field_name,
                    "check_type": v.check_type,
                    "validation_status": v.validation_status,
                    "is_valid": v.is_valid,
                    "validation_reason": v.validation_reason,
                    "review_priority": v.review_priority,
                    "review_resolution": v.review_resolution,
                    "reviewed_at": v.reviewed_at.isoformat() if v.reviewed_at else None,
                }
                for v in chq.validation_results
            ]

            runs = [
                {
                    "id": r.id,
                    "run_timestamp": r.run_timestamp.isoformat() if r.run_timestamp else "",
                    "status": r.status,
                    "detector_model_version": r.detector_model_version,
                    "recognizer_model_version": r.recognizer_model_version,
                    "ocr_engine_version": r.ocr_engine_version,
                    "total_latency_ms": r.total_latency_ms,
                    "stage_latencies_json": r.stage_latencies_json,
                }
                for r in chq.processing_runs
            ]

            return {
                "id": chq.id,
                "cheque_identifier": chq.cheque_identifier,
                "image_path": chq.image_path,
                "image_width": chq.image_width,
                "image_height": chq.image_height,
                "image_hash": chq.image_hash,
                "source_dataset": chq.source_dataset,
                "status": chq.status,
                "overall_confidence": chq.overall_confidence,
                "review_required": chq.review_required,
                "created_at": chq.created_at.isoformat() if chq.created_at else "",
                "updated_at": chq.updated_at.isoformat() if chq.updated_at else "",
                "extracted_fields": fields,
                "validation_results": validations,
                "processing_runs": runs,
            }

    def list_cheques(
        self,
        status: Optional[str] = None,
        review_required: Optional[bool] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Lists cheques matching filters."""
        if self.is_http_backend_available():
            try:
                params: Dict[str, Any] = {"skip": skip, "limit": limit}
                if status:
                    params["status"] = status
                if review_required is not None:
                    params["review_required"] = review_required

                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(
                        f"{self.base_url}/cheques",
                        params=params,
                        headers=self._get_headers(),
                    )
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                pass

        # Service-layer fallback
        with db_session_scope() as session:
            raw_items = list_cheques(session, status=status, review_required=review_required, skip=skip, limit=limit)
            items = []
            for chq in raw_items:
                chq_data = self.get_cheque(chq.id)
                if chq_data:
                    items.append(chq_data)

            return {
                "total": len(items),
                "skip": skip,
                "limit": limit,
                "items": items,
            }

    def submit_review_resolution(
        self,
        validation_id: int,
        resolution: str,  # 'ACCEPTED', 'REJECTED', 'CORRECTED'
        reviewer_id: int = 1,
        corrected_fields: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Applies human reviewer decision to a validation result and optionally updates field value."""
        with db_session_scope() as session:
            update_review_resolution(session, validation_id, reviewer_id, resolution)

            # If fields were corrected, update extracted_fields in DB
            if corrected_fields:
                val_record = session.get(ValidationResult, validation_id)
                if val_record:
                    cheque_id = val_record.cheque_id
                    for fname, new_val in corrected_fields.items():
                        f_rec = session.query(ExtractedField).filter_by(cheque_id=cheque_id, field_name=fname).first()
                        if f_rec:
                            f_rec.normalized_value = new_val
                            f_rec.confidence = 1.0  # Human verified
                            f_rec.confidence_tier = "HIGH"

                    # If all validations for cheque are accepted, update cheque to VERIFIED
                    all_vals = session.query(ValidationResult).filter_by(cheque_id=cheque_id).all()
                    if all(v.review_resolution in ["ACCEPTED", "CORRECTED"] for v in all_vals):
                        update_cheque_status(session, cheque_id, status="VERIFIED", review_required=False)

            session.commit()
            return True

    # --------------------------------------------------------------------------
    # Analytics Operations
    # --------------------------------------------------------------------------

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Fetches consolidated executive metrics."""
        if self.is_http_backend_available():
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(
                        f"{self.base_url}/analytics/summary",
                        headers=self._get_headers(),
                    )
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                pass

        # Service-layer fallback
        with db_session_scope() as session:
            summary = calculate_executive_dashboard(session)
            return summary.model_dump()

    def get_analytics_trends(self, interval: str = "day") -> Dict[str, Any]:
        """Fetches temporal volume and monetary clearing trends."""
        if self.is_http_backend_available():
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(
                        f"{self.base_url}/analytics/trends?interval={interval}",
                        headers=self._get_headers(),
                    )
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                pass

        # Service-layer fallback
        with db_session_scope() as session:
            vol_trends = calculate_volume_trend(session, interval=interval)
            amt_trends = calculate_amount_trend(session, interval=interval)
            return {
                "interval": interval,
                "volume_trends": [t.model_dump() for t in vol_trends],
                "amount_trends": [t.model_dump() for t in amt_trends],
            }


# Global singleton instance
backend_client = BackendClient()
