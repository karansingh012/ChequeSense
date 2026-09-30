"""Review queue management and compliance routing for ChequeSense."""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from src.pipeline.schemas import ChequePipelineResult, FieldOutput, SignatureOutput
from src.validation.confidence import ConfidenceAssessmentResult, ConfidenceEvaluator, ConfidenceThresholdConfig
from src.validation.consistency import ConsistencyCheckResult, CrossFieldConsistencyChecker
from src.validation.field_validator import ChequeProcessingStatus, FieldValidationResult, FieldValidator

logger = logging.getLogger("chequesense.validation.queue")


@dataclass
class PreservedFieldAudit:
    """Audit container preserving raw model output, normalized value, confidence, and review reasons."""

    field_name: str
    raw_model_output: str
    normalized_value: str
    confidence: float
    detection_confidence: float
    extraction_confidence: float
    validation_status: str       # 'VALID', 'INVALID', 'UNCHECKED'
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    reasons_for_review: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "raw_model_output": self.raw_model_output,
            "normalized_value": self.normalized_value,
            "confidence": round(self.confidence, 4),
            "detection_confidence": round(self.detection_confidence, 4),
            "extraction_confidence": round(self.extraction_confidence, 4),
            "validation_status": self.validation_status,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "reasons_for_review": self.reasons_for_review,
        }


@dataclass
class ReviewQueueItem:
    """Full auditable cheque record managed within the teller review queue."""

    cheque_id: str
    status: ChequeProcessingStatus
    overall_confidence: float
    review_priority: str                 # 'HIGH', 'MEDIUM', 'LOW'
    review_required: bool
    reasons_for_review: List[str] = field(default_factory=list)
    fields: Dict[str, PreservedFieldAudit] = field(default_factory=dict)
    signatures: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    resolution: Optional[str] = None     # 'PENDING', 'ACCEPTED', 'REJECTED', 'CORRECTED'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cheque_id": self.cheque_id,
            "status": self.status.value,
            "overall_confidence": round(self.overall_confidence, 4),
            "review_priority": self.review_priority,
            "review_required": self.review_required,
            "reasons_for_review": self.reasons_for_review,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "signatures": self.signatures,
            "timestamp": self.timestamp,
            "resolution": self.resolution or "PENDING",
            "metadata": self.metadata,
        }


class ReviewQueueManager:
    """Applies validation rules, confidence gates, and consistency checks to compile review records."""

    def __init__(
        self,
        confidence_config: Optional[ConfidenceThresholdConfig] = None,
        reference_date: Optional[datetime.date] = None,
    ):
        self.field_validator = FieldValidator()
        self.confidence_evaluator = ConfidenceEvaluator(config=confidence_config)
        self.consistency_checker = CrossFieldConsistencyChecker()
        self.reference_date = reference_date
        self.queue: List[ReviewQueueItem] = []

    def evaluate_cheque(
        self,
        pipeline_result: Union[ChequePipelineResult, Dict[str, Any]],
        bank_name: Optional[str] = None,
        reference_date: Optional[datetime.date] = None,
    ) -> ReviewQueueItem:
        """Evaluates an extraction result, assigns status, and preserves all raw/normalized outputs."""
        ref_date = reference_date or self.reference_date

        if isinstance(pipeline_result, ChequePipelineResult):
            cheque_id = pipeline_result.cheque_id
            fields_data = pipeline_result.fields
            signatures_data = pipeline_result.signatures.model_dump()
            overall_conf = pipeline_result.overall_confidence
        else:
            cheque_id = pipeline_result.get("cheque_id", "unknown_cheque")
            fields_data = pipeline_result.get("fields", {})
            signatures_data = pipeline_result.get("signatures", {})
            overall_conf = float(pipeline_result.get("overall_confidence", 0.0))

        preserved_fields: Dict[str, PreservedFieldAudit] = {}
        all_review_reasons: List[str] = []
        fatal_errors: List[str] = []

        # 1. Field-by-Field Validation
        for fname in ["acno", "ifsc", "date", "amount", "name"]:
            f_obj = fields_data.get(fname)
            val = getattr(f_obj, "value", None)
            raw = getattr(f_obj, "raw_value", None)
            conf = getattr(f_obj, "confidence", 0.0)
            det_conf = getattr(f_obj, "detection_confidence", 1.0)
            ext_conf = getattr(f_obj, "extraction_confidence", 1.0)

            if isinstance(f_obj, dict):
                val = f_obj.get("value", "")
                raw = f_obj.get("raw_value", "")
                conf = f_obj.get("confidence", 0.0)
                det_conf = f_obj.get("detection_confidence", 1.0)
                ext_conf = f_obj.get("extraction_confidence", 1.0)

            val_str = str(val or "")
            raw_str = str(raw or val or "")
            conf_val = float(conf or 0.0)

            field_reasons: List[str] = []

            # Perform validation based on field type
            if fname == "acno":
                vres = self.field_validator.validate_account_number(val_str, raw_str)
            elif fname == "ifsc":
                vres = self.field_validator.validate_ifsc(val_str, raw_str)
            elif fname == "date":
                vres = self.field_validator.validate_date(val_str, raw_str, reference_date=ref_date)
            elif fname == "amount":
                vres = self.field_validator.validate_amount(val_str, raw_str)
            elif fname == "name":
                vres = self.field_validator.validate_payee_name(val_str, raw_str)
            else:
                vres = FieldValidationResult(fname, True, raw_str, val_str)

            # Collect errors and warnings
            if not vres.is_valid:
                v_status = "INVALID"
                fatal_errors.extend(vres.errors)
                field_reasons.extend(vres.errors)
            else:
                v_status = "VALID"

            if vres.warnings:
                field_reasons.extend(vres.warnings)

            # Check field confidence gate
            tier, detail = self.confidence_evaluator.evaluate_field_confidence(
                fname, conf_val, det_conf, ext_conf
            )
            if detail:
                field_reasons.append(detail.reason)

            if field_reasons:
                all_review_reasons.extend([f"[{fname.upper()}] {r}" for r in field_reasons])

            preserved_fields[fname] = PreservedFieldAudit(
                field_name=fname,
                raw_model_output=raw_str,
                normalized_value=vres.normalized_value,
                confidence=conf_val,
                detection_confidence=float(det_conf),
                extraction_confidence=float(ext_conf),
                validation_status=v_status,
                validation_errors=vres.errors,
                validation_warnings=vres.warnings,
                reasons_for_review=field_reasons,
            )

        # 2. Required Fields Presence Check
        req_ok, missing_fields = self.field_validator.validate_required_fields(preserved_fields)
        if not req_ok:
            fatal_errors.extend(missing_fields)
            all_review_reasons.extend(missing_fields)

        # 3. Signature Presence Check
        sign_present = bool(signatures_data.get("present", False))
        sign_conf = float(signatures_data.get("confidence", 0.0))
        if not sign_present or sign_conf < 0.50:
            all_review_reasons.append(
                f"Signature unconfirmed in designated signatory area (confidence: {sign_conf:.2f})"
            )

        # 4. Cross-Field Consistency Checks
        consistency_res = self.consistency_checker.evaluate_all(
            fields=preserved_fields,
            signature_present=sign_present,
            signature_confidence=sign_conf,
            bank_name=bank_name,
        )
        if not consistency_res.all_passed:
            for disc in consistency_res.discrepancies:
                all_review_reasons.append(f"[CONSISTENCY] {disc.description}")
                if disc.severity == "CRITICAL":
                    fatal_errors.append(disc.description)

        # 5. Determine Overall Processing Status
        # Important: NEVER claim fraud based on anomaly detection.
        # Use strictly objective banking categories:
        if fatal_errors:
            # Fatal business failures: non-numeric account number, negative amount, missing mandatory field
            status = ChequeProcessingStatus.INVALID
            priority = "HIGH"
            review_required = True
        elif all_review_reasons:
            # Requires human verification: low confidence, stale cheque, missing signature
            status = ChequeProcessingStatus.REVIEW_REQUIRED
            # Higher priority for large amounts or high discrepancies
            amt_num = preserved_fields.get("amount", PreservedFieldAudit("", "", "", 0, 0, 0, "")).normalized_value
            is_high_val = False
            try:
                is_high_val = float(amt_num) >= 50000.0
            except ValueError:
                pass
            priority = "HIGH" if (is_high_val or len(all_review_reasons) >= 3) else "MEDIUM"
            review_required = True
        elif overall_conf >= self.confidence_evaluator.config.high_threshold:
            # All checks passed cleanly with high confidence
            status = ChequeProcessingStatus.VERIFIED
            priority = "LOW"
            review_required = False
        else:
            # Passed baseline validations with medium confidence
            status = ChequeProcessingStatus.PROCESSED
            priority = "LOW"
            review_required = False

        item = ReviewQueueItem(
            cheque_id=cheque_id,
            status=status,
            overall_confidence=overall_conf,
            review_priority=priority,
            review_required=review_required,
            reasons_for_review=all_review_reasons,
            fields=preserved_fields,
            signatures=signatures_data,
            resolution="PENDING",
            metadata={
                "fatal_error_count": len(fatal_errors),
                "review_reason_count": len(all_review_reasons),
                "reference_date": ref_date.isoformat() if ref_date else None,
            },
        )

        self.queue.append(item)
        logger.info(
            "Evaluated cheque '%s' -> Status: %s | Priority: %s | Review Reasons: %d",
            cheque_id,
            status.value,
            priority,
            len(all_review_reasons),
        )
        return item

    def get_items(
        self,
        status: Optional[ChequeProcessingStatus] = None,
        priority: Optional[str] = None,
    ) -> List[ReviewQueueItem]:
        """Returns filtered items from the review queue sorted by priority."""
        res = list(self.queue)
        if status:
            res = [i for i in res if i.status == status]
        if priority:
            res = [i for i in res if i.review_priority == priority]

        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        res.sort(key=lambda x: priority_order.get(x.review_priority, 99))
        return res
