"""Confidence thresholding, low-confidence detection, and reliability gating."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("chequesense.validation.confidence")


@dataclass
class ConfidenceThresholdConfig:
    """Configurable confidence thresholds for straight-through processing vs review."""

    high_threshold: float = 0.80   # Eligible for automated processing if checks pass
    medium_threshold: float = 0.60 # Acceptable with advisory note
    low_threshold: float = 0.40    # Severe uncertainty threshold


@dataclass
class LowConfidenceDetail:
    """Diagnostic detail for a field failing confidence gates."""

    field_name: str
    confidence: float
    detection_confidence: float
    extraction_confidence: float
    threshold_applied: float
    tier: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "confidence": round(self.confidence, 4),
            "detection_confidence": round(self.detection_confidence, 4),
            "extraction_confidence": round(self.extraction_confidence, 4),
            "threshold_applied": self.threshold_applied,
            "tier": self.tier,
            "reason": self.reason,
        }


@dataclass
class ConfidenceAssessmentResult:
    """Overall confidence evaluation across all extracted cheque fields."""

    overall_confidence: float
    low_confidence_fields: List[LowConfidenceDetail] = field(default_factory=list)
    review_required: bool = False
    reasons: List[str] = field(default_factory=list)
    field_tiers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_confidence": round(self.overall_confidence, 4),
            "review_required": self.review_required,
            "low_confidence_count": len(self.low_confidence_fields),
            "low_confidence_fields": [f.to_dict() for f in self.low_confidence_fields],
            "reasons": self.reasons,
            "field_tiers": self.field_tiers,
        }


class ConfidenceEvaluator:
    """Assesses confidence scores across pipeline field extractions and enforces quality gates."""

    # Core fields whose confidence gates are mandatory
    MANDATORY_GATES = ["acno", "amount", "date", "ifsc", "name"]

    def __init__(self, config: Optional[ConfidenceThresholdConfig] = None):
        self.config = config or ConfidenceThresholdConfig()

    def evaluate_field_confidence(
        self,
        field_name: str,
        confidence: float,
        detection_confidence: float = 1.0,
        extraction_confidence: float = 1.0,
    ) -> Tuple[str, Optional[LowConfidenceDetail]]:
        """Classifies individual field into HIGH, MEDIUM, or LOW tier."""
        c = max(0.0, min(1.0, confidence))

        if c >= self.config.high_threshold:
            tier = "HIGH"
            detail = None
        elif c >= self.config.medium_threshold:
            tier = "MEDIUM"
            detail = None
        else:
            tier = "LOW"
            detail = LowConfidenceDetail(
                field_name=field_name,
                confidence=c,
                detection_confidence=detection_confidence,
                extraction_confidence=extraction_confidence,
                threshold_applied=self.config.medium_threshold,
                tier=tier,
                reason=f"Composite confidence {c:.2f} falls below threshold {self.config.medium_threshold:.2f}",
            )

        return tier, detail

    def evaluate_cheque_confidence(self, fields: Dict[str, Any]) -> ConfidenceAssessmentResult:
        """Evaluates overall confidence across all fields and identifies review requirements."""
        field_tiers: Dict[str, str] = {}
        low_conf_details: List[LowConfidenceDetail] = []
        review_reasons: List[str] = []
        all_confidences: List[float] = []

        for fname, f_data in fields.items():
            conf = getattr(f_data, "confidence", None)
            det_conf = getattr(f_data, "detection_confidence", 1.0)
            ext_conf = getattr(f_data, "extraction_confidence", 1.0)

            if conf is None and isinstance(f_data, dict):
                conf = f_data.get("confidence", 0.0)
                det_conf = f_data.get("detection_confidence", 1.0)
                ext_conf = f_data.get("extraction_confidence", 1.0)

            conf = float(conf or 0.0)
            all_confidences.append(conf)

            tier, detail = self.evaluate_field_confidence(
                field_name=fname,
                confidence=conf,
                detection_confidence=det_conf,
                extraction_confidence=ext_conf,
            )
            field_tiers[fname] = tier

            if detail is not None:
                low_conf_details.append(detail)
                if fname in self.MANDATORY_GATES:
                    review_reasons.append(
                        f"Low confidence on mandatory field '{fname}': {conf:.2f} (Tier: {tier})"
                    )

        # Overall composite score calculation
        if all_confidences:
            overall = float(sum(all_confidences) / len(all_confidences))
        else:
            overall = 0.0

        if overall < self.config.medium_threshold:
            review_reasons.append(
                f"Overall document confidence {overall:.2f} is below acceptance threshold {self.config.medium_threshold:.2f}"
            )

        review_required = len(review_reasons) > 0

        return ConfidenceAssessmentResult(
            overall_confidence=overall,
            low_confidence_fields=low_conf_details,
            review_required=review_required,
            reasons=review_reasons,
            field_tiers=field_tiers,
        )
