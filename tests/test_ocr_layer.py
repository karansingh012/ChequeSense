"""Unit tests for the ChequeSense OCR and text-processing module."""

import numpy as np
import pytest
import cv2
from PIL import Image

from src.ocr.confidence import ConfidenceAssessor, FieldConfidenceReport
from src.ocr.ocr_engine import OCREngine, OCRResult, OCRWordToken
from src.ocr.postprocess import FieldNormalizer, NormalizedFieldResult
from src.ocr.preprocessing import OCRPreprocessor


def test_ocr_preprocessor_loading_and_scaling():
    prep = OCRPreprocessor(target_dpi_scale=2.0)

    # 1. Test PIL Image input
    pil_img = Image.new("RGB", (100, 30), color="white")
    arr = prep.load_image(pil_img)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (30, 100, 3)

    # 2. Test preprocessing with small crop upscaling
    proc = prep.preprocess_field(pil_img, field_type="ifsc", binarize=True)
    assert isinstance(proc, np.ndarray)
    # Upscaled because height was 30 < 60
    assert proc.shape[0] >= 60
    assert proc.dtype == np.uint8
    assert proc.min() >= 0 and proc.max() <= 255


def test_ocr_engine_synthetic_field():
    ocr = OCREngine()

    # Generate synthetic image with clear printed IFSC code
    canvas = np.ones((80, 400, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "HDFC0001234", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    res = ocr.extract_text(canvas, field_type="ifsc")
    assert isinstance(res, OCRResult)
    assert "HDFC" in res.raw_text
    assert len(res.tokens) >= 1
    assert 0.0 <= res.mean_confidence <= 1.0


def test_field_normalizer_account_number():
    normalizer = FieldNormalizer()

    # Case 1: Standard account number with label and leading zeros
    res1 = normalizer.normalize("A/c No: 001401000213", field_type="acno")
    assert res1.normalized_text == "001401000213"
    assert res1.is_valid is True
    assert res1.metadata["has_leading_zero"] is True
    assert "extracted_digits_preserving_leading_zeros" in res1.cleaning_steps_applied

    # Case 2: Letter 'O' confusion for digit '0'
    res2 = normalizer.normalize("Acc No: O30002010108841-", field_type="acno")
    assert res2.normalized_text == "030002010108841"
    assert res2.is_valid is True

    # Case 3: Invalid short string
    res3 = normalizer.normalize("No: 12345", field_type="acno")
    assert res3.is_valid is False


def test_field_normalizer_ifsc():
    normalizer = FieldNormalizer()

    # Case 1: Standard clean IFSC with label
    res1 = normalizer.normalize("IFSC : SYNB0003011", field_type="ifsc")
    assert res1.normalized_text == "SYNB0003011"
    assert res1.is_valid is True
    assert res1.metadata["bank_code"] == "SYNB"

    # Case 2: OCR confusion: letter 'O' instead of digit '0' in 5th position
    res2 = normalizer.normalize("RTGS/NEFT: CNRB0002854", field_type="ifsc")
    assert res2.normalized_text == "CNRB0002854"
    assert res2.is_valid is True

    # Case 3: Letter 'O' in 5th pos corrected to '0'
    res3 = normalizer.normalize("HDFCO001234", field_type="ifsc")
    assert res3.normalized_text == "HDFC0001234"
    assert res3.is_valid is True
    assert "corrected_5th_char_O_to_0" in res3.cleaning_steps_applied


def test_field_normalizer_date():
    normalizer = FieldNormalizer()

    # Case 1: Standard slash delimited date
    res1 = normalizer.normalize("Date: 06/05/2022", field_type="date")
    assert res1.normalized_text == "06/05/2022"
    assert res1.is_valid is True

    # Case 2: 8 continuous digits
    res2 = normalizer.normalize("15082023", field_type="date")
    assert res2.normalized_text == "15/08/2023"
    assert res2.is_valid is True

    # Case 3: Strip printed guide text 'DDMM YYYY'
    res3 = normalizer.normalize("25 12 2024 DDMM YYYY", field_type="date")
    assert res3.normalized_text == "25/12/2024"
    assert res3.is_valid is True


def test_field_normalizer_amount():
    normalizer = FieldNormalizer()

    # Case 1: Indian Rupee symbol and delimiter
    res1 = normalizer.normalize("₹ 45,000/-", field_type="amount")
    assert res1.normalized_text == "45000"
    assert res1.is_valid is True
    assert res1.metadata["amount_value"] == 45000.0

    # Case 2: Rs with paise decimal
    res2 = normalizer.normalize("Rs. 1,250.75 /=", field_type="amount")
    assert res2.normalized_text == "1250.75"
    assert res2.is_valid is True


def test_field_normalizer_payee_name():
    normalizer = FieldNormalizer()

    # Case 1: Payee with 'Pay' and 'or Bearer' boilerplate
    res1 = normalizer.normalize("Pay   ETHEL M. BRYSON   or Bearer", field_type="name")
    assert res1.normalized_text == "Ethel M. Bryson"
    assert res1.is_valid is True

    # Case 2: Boilerplate removal
    res2 = normalizer.normalize("Pay  MARCEL ACHEN  ***", field_type="name")
    assert res2.normalized_text == "Marcel Achen"
    assert res2.is_valid is True


def test_field_normalizer_amount_words():
    normalizer = FieldNormalizer()

    res = normalizer.normalize("Rupees  Four Thousand Seven Hundred and Twenty  Only /-", field_type="amt_in_words")
    assert res.normalized_text == "Four Thousand Seven Hundred And Twenty"
    assert res.is_valid is True


def test_confidence_assessor_tiers():
    assessor = ConfidenceAssessor(high_confidence_thresh=0.80, low_confidence_thresh=0.60)

    # 1. High confidence valid extraction
    ocr_high = OCRResult(raw_text="CNRB0002854", mean_confidence=0.92, min_confidence=0.85)
    norm_high = NormalizedFieldResult(
        field_type="ifsc",
        raw_text="CNRB0002854",
        normalized_text="CNRB0002854",
        is_valid=True,
        confidence=0.92,
    )
    rep_high = assessor.assess_field(ocr_high, norm_high)
    assert rep_high.confidence_tier == "HIGH"
    assert rep_high.review_required is False
    assert rep_high.composite_score >= 0.80

    # 2. Low confidence invalid extraction
    ocr_low = OCRResult(raw_text="W#? 12", mean_confidence=0.25, min_confidence=0.10)
    norm_low = NormalizedFieldResult(
        field_type="acno",
        raw_text="W#? 12",
        normalized_text="12",
        is_valid=False,
        confidence=0.25,
    )
    rep_low = assessor.assess_field(ocr_low, norm_low)
    assert rep_low.confidence_tier == "LOW"
    assert rep_low.review_required is True
    assert "Failed domain schema validation for 'acno'" in rep_low.review_reasons
