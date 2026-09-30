"""Confidence estimation and validation assessment for cheque field OCR extractions.

Evaluates token-level engine confidences, validates syntax and domain schemas,
computes composite reliability scores, and flags uncertain fields for human review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from src.ocr.ocr_engine import OCRResult, OCRWordToken
from src.ocr.postprocess import NormalizedFieldResult


@dataclass
class FieldConfidenceReport:
    """Comprehensive confidence and verification status for an extracted field."""

    field_type: str
    raw_text: str
    normalized_text: str
    ocr_mean_confidence: float
    ocr_min_confidence: float
    schema_valid: bool
    composite_score: float  # [0.0, 1.0]
    review_required: bool
    review_reasons: List[str] = field(default_factory=list)
    confidence_tier: str = "HIGH"  # 'HIGH', 'MEDIUM', 'LOW'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_type": self.field_type,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "ocr_mean_confidence": round(self.ocr_mean_confidence, 4),
            "ocr_min_confidence": round(self.ocr_min_confidence, 4),
            "schema_valid": self.schema_valid,
            "composite_score": round(self.composite_score, 4),
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
            "confidence_tier": self.confidence_tier,
        }


class ConfidenceAssessor:
    """Computes field reliability scores and flags anomalies."""

    def __init__(
        self,
        high_confidence_thresh: float = 0.80,
        low_confidence_thresh: float = 0.60,
    ):
        self.high_thresh = high_confidence_thresh
        self.low_thresh = low_confidence_thresh

    def assess_field(
        self,
        ocr_result: OCRResult,
        normalized_result: NormalizedFieldResult,
    ) -> FieldConfidenceReport:
        """Assesses raw OCR and normalized result, returning a detailed reliability report."""
        field_type = normalized_result.field_type
        raw_text = ocr_result.raw_text
        norm_text = normalized_result.normalized_text
        mean_conf = ocr_result.mean_confidence
        min_conf = ocr_result.min_confidence

        review_reasons = []

        # 1. Check if raw text was empty
        if not raw_text.strip():
            review_reasons.append("Empty OCR output")
            return FieldConfidenceReport(
                field_type=field_type,
                raw_text=raw_text,
                normalized_text="",
                ocr_mean_confidence=0.0,
                ocr_min_confidence=0.0,
                schema_valid=False,
                composite_score=0.0,
                review_required=True,
                review_reasons=review_reasons,
                confidence_tier="LOW",
            )

        # 2. Check schema validity
        schema_valid = normalized_result.is_valid
        if not schema_valid:
            review_reasons.append(f"Failed domain schema validation for '{field_type}'")

        # 3. Check OCR Engine Confidence
        if mean_conf < self.low_thresh:
            review_reasons.append(
                f"Low OCR mean confidence ({mean_conf:.2f} < {self.low_thresh:.2f})"
            )
        elif min_conf < 0.40 and len(ocr_result.tokens) > 1:
            review_reasons.append(
                f"Contains low-confidence token ({min_conf:.2f} < 0.40)"
            )

        # 4. Field-specific heuristics
        if field_type == "acno":
            if len(norm_text) < 9 or len(norm_text) > 18:
                review_reasons.append(f"Account number length {len(norm_text)} outside standard [9, 18]")
        elif field_type == "ifsc":
            if len(norm_text) != 11:
                review_reasons.append(f"IFSC length {len(norm_text)} != 11")
            elif not norm_text[4] == "0":
                review_reasons.append("IFSC 5th character is not '0'")
        elif field_type == "date":
            if not schema_valid:
                review_reasons.append("Invalid calendar date values")
        elif field_type == "amount":
            if not norm_text:
                review_reasons.append("No numerical amount extracted")

        # 5. Composite Reliability Score Calculation
        # Base weight: 60% OCR confidence + 40% Schema conformity
        schema_weight = 0.40 if schema_valid else 0.0
        composite = (mean_conf * 0.60) + schema_weight

        # Penalty if review reasons exist
        if review_reasons:
            composite = max(0.0, composite * 0.85)

        # Determine Tier
        if composite >= self.high_thresh and not review_reasons:
            tier = "HIGH"
            review_req = False
        elif composite >= self.low_thresh and schema_valid:
            tier = "MEDIUM"
            review_req = False
        else:
            tier = "LOW"
            review_req = True

        return FieldConfidenceReport(
            field_type=field_type,
            raw_text=raw_text,
            normalized_text=norm_text,
            ocr_mean_confidence=mean_conf,
            ocr_min_confidence=min_conf,
            schema_valid=schema_valid,
            composite_score=composite,
            review_required=review_req,
            review_reasons=review_reasons,
            confidence_tier=tier,
        )
