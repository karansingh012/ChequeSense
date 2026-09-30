"""Unit tests for the ChequeSense integrated inference pipeline."""

import json
from pathlib import Path
import numpy as np
import pytest
import cv2
from PIL import Image

from src.pipeline.detect_fields import DetectedFieldCrop, PipelineFieldDetector
from src.pipeline.ocr import PipelineOCR, PipelineOCROutput
from src.pipeline.pipeline import ChequeInferencePipeline
from src.pipeline.preprocess import PipelineImagePreprocessor
from src.pipeline.recognize import PipelineRecognizer, RecognitionOutput
from src.pipeline.reconstruct import FieldReconstructor
from src.pipeline.schemas import (
    BoundingBox,
    ChequePipelineResult,
    FieldOutput,
    PipelineConfig,
    PipelineDiagnostics,
    SignatureOutput,
)


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.detector_model_path == "models/field_detector/best_model.pt"
    assert config.recognizer_model_path == "models/recognizer/best_model.pt"
    assert config.device == "cpu"
    assert config.detection_threshold == 0.50


def test_pipeline_preprocessor_validation():
    prep = PipelineImagePreprocessor()

    # 1. Valid PIL Image
    img = Image.new("RGB", (600, 300), color="white")
    res_img, cid = prep.load_and_validate(img)
    assert isinstance(res_img, Image.Image)
    assert cid == "cheque_input"

    # 2. Too small image should raise ValueError
    tiny = Image.new("RGB", (50, 50), color="white")
    with pytest.raises(ValueError):
        prep.load_and_validate(tiny)

    # 3. Non-existent file should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        prep.load_and_validate("non_existent_file_path.png")


def test_confidence_propagation():
    # Perfect detection and high extraction
    conf1 = FieldReconstructor.propagate_confidence(1.0, 0.90)
    assert 0.85 <= conf1 <= 0.95

    # Very low detection drops composite
    conf2 = FieldReconstructor.propagate_confidence(0.10, 0.90)
    assert conf2 < 0.60

    # Low extraction drops composite
    conf3 = FieldReconstructor.propagate_confidence(0.99, 0.20)
    assert conf3 < 0.40


def test_field_reconstructor_routing():
    config = PipelineConfig()
    reconstructor = FieldReconstructor(config)

    det_crop = DetectedFieldCrop(
        field_name="ifsc",
        crop_image=Image.new("RGB", (100, 30), color="white"),
        confidence=0.99,
        bounding_box=BoundingBox(xmin=10, ymin=10, xmax=110, ymax=40),
    )
    ocr_out = PipelineOCROutput(
        field_name="ifsc",
        raw_text="IFSC: HDFC0001234",
        normalized_text="HDFC0001234",
        mean_confidence=0.95,
        composite_confidence=0.96,
        confidence_tier="HIGH",
        is_valid_schema=True,
        review_required=False,
        review_reasons=[],
        cleaning_steps=["stripped_labels"],
    )

    field_out = reconstructor.reconstruct_field(
        field_name="ifsc",
        det_crop=det_crop,
        ocr_out=ocr_out,
    )

    assert isinstance(field_out, FieldOutput)
    assert field_out.value == "HDFC0001234"
    assert field_out.confidence_tier == "HIGH"
    assert field_out.method == "ocr"
    assert field_out.confidence > 0.90
    assert field_out.bounding_box is not None


def test_full_pipeline_synthetic_cheque():
    test_cheque_path = "artifacts/pipeline/test_cheques/syn_syndicate_syn_0001.png"
    if not Path(test_cheque_path).exists():
        pytest.skip(f"Test image not found: {test_cheque_path}")

    config = PipelineConfig()
    pipeline = ChequeInferencePipeline(config=config)

    result = pipeline.process(test_cheque_path)
    assert isinstance(result, ChequePipelineResult)
    assert result.cheque_id == "syn_syndicate_syn_0001"
    assert result.status in ("SUCCESS", "PARTIAL")
    assert 0.0 <= result.overall_confidence <= 1.0

    # Verify core fields are present in structured output
    for expected_field in ["date", "amount", "ifsc", "acno", "name"]:
        assert expected_field in result.fields
        f = result.fields[expected_field]
        assert isinstance(f, FieldOutput)
        assert 0.0 <= f.confidence <= 1.0

    # Verify signature status
    assert isinstance(result.signatures, SignatureOutput)
    assert result.signatures.present is True
    assert result.signatures.confidence > 0.80

    # Verify telemetry diagnostics
    assert result.diagnostics is not None
    assert result.diagnostics.processing_time_ms > 0
    assert "detection_ms" in result.diagnostics.stage_latencies_ms
    assert "extraction_ms" in result.diagnostics.stage_latencies_ms

    # Verify JSON serialization
    json_str = result.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["cheque_id"] == "syn_syndicate_syn_0001"
    assert "fields" in parsed
    assert "signatures" in parsed
