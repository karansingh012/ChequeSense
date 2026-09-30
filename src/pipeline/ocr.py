"""OCR and textual transcription stage for ChequeSense."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

from src.ocr.confidence import ConfidenceAssessor, FieldConfidenceReport
from src.ocr.ocr_engine import OCREngine, OCRResult
from src.ocr.postprocess import FieldNormalizer, NormalizedFieldResult
from src.pipeline.schemas import PipelineConfig

logger = logging.getLogger("chequesense.pipeline.ocr")


@dataclass
class PipelineOCROutput:
    """Standardized OCR and post-processing output for a cheque field."""

    field_name: str
    raw_text: str
    normalized_text: str
    mean_confidence: float
    composite_confidence: float
    confidence_tier: str
    is_valid_schema: bool
    review_required: bool
    review_reasons: List[str]
    cleaning_steps: List[str]


class PipelineOCR:
    """Manages OCR execution, deterministic normalizations, and confidence assessments."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.engine = OCREngine(
            tesseract_cmd=config.tesseract_cmd,
            language=config.tesseract_language,
        )
        self.normalizer = FieldNormalizer()
        self.assessor = ConfidenceAssessor(
            high_confidence_thresh=config.high_confidence_threshold,
            low_confidence_thresh=config.low_confidence_threshold,
        )

    def process_field(self, crop_image: Image.Image, field_name: str) -> PipelineOCROutput:
        """Runs OCR extraction, field-specific normalization, and confidence assessment."""
        try:
            ocr_res: OCRResult = self.engine.extract_text(crop_image, field_type=field_name)
            norm_res: NormalizedFieldResult = self.normalizer.normalize(
                ocr_res.raw_text,
                field_type=field_name,
                confidence=ocr_res.mean_confidence,
            )
            conf_rep: FieldConfidenceReport = self.assessor.assess_field(ocr_res, norm_res)

            return PipelineOCROutput(
                field_name=field_name,
                raw_text=ocr_res.raw_text,
                normalized_text=norm_res.normalized_text,
                mean_confidence=round(ocr_res.mean_confidence, 4),
                composite_confidence=round(conf_rep.composite_score, 4),
                confidence_tier=conf_rep.confidence_tier,
                is_valid_schema=norm_res.is_valid,
                review_required=conf_rep.review_required,
                review_reasons=conf_rep.review_reasons,
                cleaning_steps=norm_res.cleaning_steps_applied,
            )
        except Exception as e:
            logger.error("OCR execution failed on field '%s': %s", field_name, e, exc_info=True)
            return PipelineOCROutput(
                field_name=field_name,
                raw_text="",
                normalized_text="",
                mean_confidence=0.0,
                composite_confidence=0.0,
                confidence_tier="LOW",
                is_valid_schema=False,
                review_required=True,
                review_reasons=[f"OCR execution exception: {str(e)}"],
                cleaning_steps=[],
            )
