"""End-to-end ChequeSense inference pipeline.

Integrates preprocessing, deep field detection, region cropping, handwritten recognition,
Tesseract OCR extraction, confidence propagation, and structured JSON compilation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from src.pipeline.detect_fields import DetectedFieldCrop, PipelineFieldDetector
from src.pipeline.ocr import PipelineOCR, PipelineOCROutput
from src.pipeline.preprocess import PipelineImagePreprocessor
from src.pipeline.recognize import PipelineRecognizer, RecognitionOutput
from src.pipeline.reconstruct import FieldReconstructor
from src.pipeline.schemas import (
    ChequePipelineResult,
    FieldOutput,
    PipelineConfig,
    PipelineDiagnostics,
    SignatureOutput,
)

# Configure pipeline logger
logger = logging.getLogger("chequesense.pipeline")


class ChequeInferencePipeline:
    """Production end-to-end inference pipeline for AI cheque analysis."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        logger.info("Initializing ChequeSense pipeline on device: %s", self.config.device)

        self.preprocessor = PipelineImagePreprocessor()
        self.field_detector = PipelineFieldDetector(self.config)
        self.recognizer = PipelineRecognizer(self.config)
        self.ocr = PipelineOCR(self.config)
        self.reconstructor = FieldReconstructor(self.config)

    def process(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
        cheque_id: Optional[str] = None,
    ) -> ChequePipelineResult:
        """Executes full multi-stage processing pipeline on a single cheque image."""
        start_time = time.perf_counter()
        stage_latencies: Dict[str, float] = {}

        # 1. Image Preprocessing Stage
        t0 = time.perf_counter()
        pil_image, inferred_id = self.preprocessor.load_and_validate(image_input)
        effective_id = cheque_id or inferred_id
        orig_w, orig_h = pil_image.size
        stage_latencies["preprocessing_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Field Detection & Cropping Stage
        t1 = time.perf_counter()
        field_crops, signature_status, _ = self.field_detector.detect_and_crop(pil_image)
        stage_latencies["detection_ms"] = (time.perf_counter() - t1) * 1000.0

        # 3. Recognition & OCR Extraction Stage
        t2 = time.perf_counter()
        fields_output: Dict[str, FieldOutput] = {}

        for field_name, crop_info in field_crops.items():
            crop_img = crop_info.crop_image

            # Run OCR
            ocr_out: Optional[PipelineOCROutput] = self.ocr.process_field(crop_img, field_name)

            # For numerical fields, also run handwritten digit sequence recognizer
            rec_out: Optional[RecognitionOutput] = None
            if field_name in ("amount", "date"):
                rec_out = self.recognizer.recognize_field(crop_img, field_name)

            # 4. Field Reconstruction & Confidence Propagation
            field_res = self.reconstructor.reconstruct_field(
                field_name=field_name,
                det_crop=crop_info,
                ocr_out=ocr_out,
                rec_out=rec_out,
            )
            fields_output[field_name] = field_res

        # Account for core fields that were completely missed by detector
        for core_f in self.field_detector.CORE_FIELDS:
            if core_f not in fields_output:
                fields_output[core_f] = self.reconstructor.reconstruct_field(
                    field_name=core_f,
                    det_crop=None,
                    ocr_out=None,
                    rec_out=None,
                )

        stage_latencies["extraction_ms"] = (time.perf_counter() - t2) * 1000.0

        # 5. Compile Diagnostics & Final Result Assembly
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        diagnostics = PipelineDiagnostics(
            processing_time_ms=round(total_time_ms, 2),
            stage_latencies_ms={k: round(v, 2) for k, v in stage_latencies.items()},
            image_width=orig_w,
            image_height=orig_h,
            review_reasons=[],
        )

        result = self.reconstructor.assemble_result(
            cheque_id=effective_id,
            fields=fields_output,
            signatures=signature_status,
            diagnostics=diagnostics,
        )

        logger.info(
            "Processed cheque '%s' in %.1f ms | Status: %s | Overall Conf: %.2f",
            effective_id,
            total_time_ms,
            result.status,
            result.overall_confidence,
        )
        return result


def main():
    """Command-line interface for the ChequeSense pipeline."""
    parser = argparse.ArgumentParser(
        description="ChequeSense AI: End-to-end cheque processing inference pipeline."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to cheque image file (PNG, JPG, TIFF)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save structured JSON output file",
    )
    parser.add_argument(
        "--detector_model",
        default="models/field_detector/best_model.pt",
        help="Path to trained field detector model checkpoint",
    )
    parser.add_argument(
        "--recognizer_model",
        default="models/recognizer/best_model.pt",
        help="Path to trained handwritten character recognizer checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--log_level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level (default: WARNING to keep stdout clean for JSON)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    )

    config = PipelineConfig(
        detector_model_path=args.detector_model,
        recognizer_model_path=args.recognizer_model,
        device=args.device,
    )

    pipeline = ChequeInferencePipeline(config=config)
    result = pipeline.process(args.image)

    # Format structured JSON
    json_output = result.model_dump_json(indent=2)

    # Print to stdout
    print(json_output)

    # Optionally write to file
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(json_output)
        logger.info("Saved pipeline output to %s", out_path)


if __name__ == "__main__":
    main()
