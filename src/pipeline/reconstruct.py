"""Field reconstruction, confidence propagation, and multi-modal fusion stage."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.pipeline.detect_fields import DetectedFieldCrop
from src.pipeline.ocr import PipelineOCROutput
from src.pipeline.recognize import RecognitionOutput
from src.pipeline.schemas import (
    BoundingBox,
    ChequePipelineResult,
    FieldOutput,
    PipelineConfig,
    PipelineDiagnostics,
    SignatureOutput,
)

logger = logging.getLogger("chequesense.pipeline.reconstruct")


class FieldReconstructor:
    """Fuses multi-stage extractions, propagates confidences, and compiles structured output."""

    CORE_FIELDS = ["date", "amount", "ifsc", "acno", "name"]

    def __init__(self, config: PipelineConfig):
        self.config = config

    @staticmethod
    def propagate_confidence(detection_conf: float, extraction_conf: float) -> float:
        """Computes propagated composite confidence using balanced geometric-harmonic weighting.

        Ensures that if either detection or recognition severely fails, the propagated score drops accordingly.
        """
        det = max(0.01, min(1.0, detection_conf))
        ext = max(0.01, min(1.0, extraction_conf))
        # Weighted geometric combination: 30% detection, 70% character/text extraction
        propagated = (det ** 0.30) * (ext ** 0.70)
        return float(round(propagated, 4))

    def reconstruct_field(
        self,
        field_name: str,
        det_crop: Optional[DetectedFieldCrop],
        ocr_out: Optional[PipelineOCROutput],
        rec_out: Optional[RecognitionOutput] = None,
    ) -> FieldOutput:
        """Selects the best extraction candidate and propagates confidence."""
        det_conf = det_crop.confidence if det_crop else 0.0
        box = det_crop.bounding_box if det_crop else None

        # 1. Routing for numerical fields (amount, date) that have both OCR and Recognizer candidates
        if field_name in ("amount", "date") and rec_out is not None and rec_out.success and ocr_out is not None:
            # If OCR extracted a schema-valid value with good confidence, prefer OCR
            if ocr_out.is_valid_schema and ocr_out.composite_confidence >= 0.65:
                chosen_val = ocr_out.normalized_text
                chosen_raw = ocr_out.raw_text
                ext_conf = ocr_out.composite_confidence
                method = "ocr"
                tier = ocr_out.confidence_tier
            # If Recognizer extracted digits and OCR was empty or low confidence
            elif rec_out.recognized_text and (not ocr_out.normalized_text or rec_out.confidence > ocr_out.composite_confidence):
                chosen_val = rec_out.recognized_text
                chosen_raw = rec_out.recognized_text
                ext_conf = rec_out.confidence
                method = "recognizer"
                tier = "HIGH" if ext_conf >= self.config.high_confidence_threshold else (
                    "MEDIUM" if ext_conf >= self.config.low_confidence_threshold else "LOW"
                )
            else:
                chosen_val = ocr_out.normalized_text or rec_out.recognized_text
                chosen_raw = ocr_out.raw_text or rec_out.recognized_text
                ext_conf = max(ocr_out.composite_confidence, rec_out.confidence)
                method = "hybrid"
                tier = ocr_out.confidence_tier

        # 2. Standard OCR routing for printed/text fields (ifsc, acno, name)
        elif ocr_out is not None:
            chosen_val = ocr_out.normalized_text
            chosen_raw = ocr_out.raw_text
            ext_conf = ocr_out.composite_confidence
            method = "ocr"
            tier = ocr_out.confidence_tier

        # 3. Fallback when only recognizer was run
        elif rec_out is not None and rec_out.success:
            chosen_val = rec_out.recognized_text
            chosen_raw = rec_out.recognized_text
            ext_conf = rec_out.confidence
            method = "recognizer"
            tier = "MEDIUM" if ext_conf >= 0.50 else "LOW"

        # 4. Completely undetected / missing field
        else:
            chosen_val = ""
            chosen_raw = ""
            ext_conf = 0.0
            method = "none"
            tier = "LOW"

        propagated_conf = self.propagate_confidence(det_conf, ext_conf) if chosen_val else 0.0

        return FieldOutput(
            value=chosen_val,
            raw_value=chosen_raw,
            confidence=propagated_conf,
            detection_confidence=round(det_conf, 4),
            extraction_confidence=round(ext_conf, 4),
            method=method,
            confidence_tier=tier,
            bounding_box=box,
        )

    def assemble_result(
        self,
        cheque_id: str,
        fields: Dict[str, FieldOutput],
        signatures: SignatureOutput,
        diagnostics: PipelineDiagnostics,
    ) -> ChequePipelineResult:
        """Assembles final structured result, computes overall confidence and review flags."""
        review_reasons = list(diagnostics.review_reasons)

        # Compute overall confidence across present core fields
        confidences = [f.confidence for f in fields.values() if f.value]
        if confidences:
            # Weighted average with penalty for missing fields
            avg_conf = float(sum(confidences) / len(confidences))
            missing_count = sum(1 for cf in self.CORE_FIELDS if cf not in fields or not fields[cf].value)
            penalty = 0.15 * missing_count
            overall_conf = max(0.0, min(1.0, avg_conf - penalty))
        else:
            overall_conf = 0.0

        # Review triggers
        for fname, f_out in fields.items():
            if f_out.confidence_tier == "LOW":
                review_reasons.append(f"Low confidence on field '{fname}' ({f_out.confidence:.2f})")
            if not f_out.value and fname in self.CORE_FIELDS:
                review_reasons.append(f"Missing core field: '{fname}'")

        if not signatures.present:
            review_reasons.append("Signature not detected in designated signatory area")

        review_required = len(review_reasons) > 0 or overall_conf < self.config.high_confidence_threshold

        status = "SUCCESS" if (overall_conf >= 0.70 and not any("Missing core field" in r for r in review_reasons)) else (
            "PARTIAL" if overall_conf >= 0.30 else "ERROR"
        )

        diagnostics.review_reasons = review_reasons

        return ChequePipelineResult(
            cheque_id=cheque_id,
            status=status,
            overall_confidence=round(overall_conf, 4),
            review_required=review_required,
            fields=fields,
            signatures=signatures,
            diagnostics=diagnostics,
        )
