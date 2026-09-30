"""Unit tests for the ChequeSense validation and human-review layer."""

import datetime
import pytest

from src.validation.confidence import (
    ConfidenceAssessmentResult,
    ConfidenceEvaluator,
    ConfidenceThresholdConfig,
)
from src.validation.consistency import (
    ConsistencyCheckResult,
    CrossFieldConsistencyChecker,
)
from src.validation.field_validator import (
    ChequeProcessingStatus,
    FieldValidationResult,
    FieldValidator,
)
from src.validation.review_queue import (
    PreservedFieldAudit,
    ReviewQueueItem,
    ReviewQueueManager,
)


# ==============================================================================
# 1. Field Format Validation Tests
# ==============================================================================

def test_account_number_validation():
    validator = FieldValidator()

    # Valid standard 14-digit Indian account number
    res1 = validator.validate_account_number("00140100021389", raw_value="Nc No: 00140100021389")
    assert res1.is_valid is True
    assert res1.normalized_value == "00140100021389"
    assert res1.raw_value == "Nc No: 00140100021389"
    assert res1.metadata["has_leading_zero"] is True
    assert len(res1.errors) == 0

    # Non-numeric characters
    res2 = validator.validate_account_number("1234ABCD567")
    assert res2.is_valid is False
    assert any("non-numeric" in err for err in res2.errors)

    # Too short (< 9 digits)
    res3 = validator.validate_account_number("1234567")
    assert res3.is_valid is False
    assert any("below minimum standard" in err for err in res3.errors)

    # Too long (> 18 digits)
    res4 = validator.validate_account_number("12345678901234567890")
    assert res4.is_valid is False
    assert any("exceeds maximum standard" in err for err in res4.errors)


def test_ifsc_validation():
    validator = FieldValidator()

    # Valid Canara Bank IFSC
    res1 = validator.validate_ifsc("CNRB0002854")
    assert res1.is_valid is True
    assert res1.metadata["bank_prefix"] == "CNRB"
    assert res1.metadata["bank_name"] == "Canara Bank"

    # Invalid: 5th character is 'O' instead of '0'
    res2 = validator.validate_ifsc("CNRBO002854")
    assert res2.is_valid is False
    assert any("5th character" in err for err in res2.errors)

    # Invalid: Length not 11
    res3 = validator.validate_ifsc("SBIN001")
    assert res3.is_valid is False
    assert any("not exactly 11 characters" in err for err in res3.errors)

    # Invalid: First 4 characters not alphabetic
    res4 = validator.validate_ifsc("12340001234")
    assert res4.is_valid is False
    assert any("must consist of 4 alphabetic letters" in err for err in res4.errors)


# ==============================================================================
# 2. Required-Field Validation Tests
# ==============================================================================

def test_required_fields_validation():
    validator = FieldValidator()

    # All core fields present
    complete_fields = {
        "acno": {"value": "001401000213"},
        "amount": {"value": "15000"},
        "date": {"value": "06/05/2022"},
        "name": {"value": "Ethel M. Bryson"},
        "ifsc": {"value": "CNRB0002854"},
    }
    ok1, missing1 = validator.validate_required_fields(complete_fields)
    assert ok1 is True
    assert len(missing1) == 0

    # Missing mandatory 'acno' and empty 'ifsc'
    incomplete_fields = {
        "amount": {"value": "15000"},
        "date": {"value": "06/05/2022"},
        "name": {"value": "Ethel M. Bryson"},
        "ifsc": {"value": ""},
    }
    ok2, missing2 = validator.validate_required_fields(incomplete_fields)
    assert ok2 is False
    assert any("'acno' missing" in m for m in missing2)
    assert any("'ifsc' has empty value" in m for m in missing2)


# ==============================================================================
# 3. Date Validation & Cheque Validity Tests
# ==============================================================================

def test_date_validation_calendar_and_validity():
    validator = FieldValidator()
    ref_date = datetime.date(2022, 6, 1)

    # 1. Valid fresh cheque within 90 days
    res1 = validator.validate_date("15/05/2022", reference_date=ref_date)
    assert res1.is_valid is True
    assert res1.normalized_value == "15/05/2022"
    assert res1.metadata["is_stale"] is False
    assert res1.metadata["is_post_dated"] is False

    # 2. Leading zero preservation
    res_lz = validator.validate_date("6/5/2022", reference_date=ref_date)
    assert res_lz.is_valid is True
    assert res_lz.normalized_value == "06/05/2022"
    assert res_lz.metadata["day"] == "06"
    assert res_lz.metadata["month"] == "05"

    # 3. Stale cheque (issued > 90 days ago)
    res_stale = validator.validate_date("01/01/2022", reference_date=ref_date)
    assert res_stale.is_valid is True
    assert res_stale.metadata["is_stale"] is True
    assert any("Stale cheque" in w for w in res_stale.warnings)

    # 4. Post-dated cheque (dated in future relative to reference date)
    res_post = validator.validate_date("15/08/2022", reference_date=ref_date)
    assert res_post.is_valid is True
    assert res_post.metadata["is_post_dated"] is True
    assert any("Post-dated cheque" in w for w in res_post.warnings)

    # 5. Impossible calendar day (Feb 30)
    res_bad_day = validator.validate_date("30/02/2022")
    assert res_bad_day.is_valid is False
    assert any("Invalid day" in err for err in res_bad_day.errors)

    # 6. Impossible calendar month (Month 13)
    res_bad_month = validator.validate_date("10/13/2022")
    assert res_bad_month.is_valid is False
    assert any("Invalid month" in err for err in res_bad_month.errors)


# ==============================================================================
# 4. Numeric & Amount Validation Tests
# ==============================================================================

def test_amount_validation():
    validator = FieldValidator()

    # Standard positive amount
    res1 = validator.validate_amount("4720.50", raw_value="₹ 4,720.50/-")
    assert res1.is_valid is True
    assert res1.normalized_value == "4720.50"
    assert res1.metadata["amount_numeric"] == 4720.50

    # Zero or negative amount -> invalid
    res_zero = validator.validate_amount("0.00")
    assert res_zero.is_valid is False
    assert any("strictly greater than zero" in err for err in res_zero.errors)

    res_neg = validator.validate_amount("-500")
    assert res_neg.is_valid is False

    # High value threshold (>= 500,000 INR)
    res_high = validator.validate_amount("750000")
    assert res_high.is_valid is True
    assert res_high.metadata["is_high_value"] is True
    assert any("Positive Pay" in w for w in res_high.warnings)


# ==============================================================================
# 5. Confidence Thresholding & Low-Confidence Detection Tests
# ==============================================================================

def test_confidence_evaluator():
    evaluator = ConfidenceEvaluator(
        config=ConfidenceThresholdConfig(high_threshold=0.80, medium_threshold=0.60)
    )

    # Individual field tiers
    t1, d1 = evaluator.evaluate_field_confidence("ifsc", 0.88)
    assert t1 == "HIGH"
    assert d1 is None

    t2, d2 = evaluator.evaluate_field_confidence("name", 0.65)
    assert t2 == "MEDIUM"
    assert d2 is None

    t3, d3 = evaluator.evaluate_field_confidence("acno", 0.42)
    assert t3 == "LOW"
    assert d3 is not None
    assert d3.field_name == "acno"

    # Cheque-level evaluation
    fields = {
        "ifsc": {"confidence": 0.90},
        "name": {"confidence": 0.85},
        "acno": {"confidence": 0.45},  # Low confidence on mandatory field
    }
    c_res = evaluator.evaluate_cheque_confidence(fields)
    assert c_res.review_required is True
    assert len(c_res.low_confidence_fields) == 1
    assert any("Low confidence on mandatory field 'acno'" in r for r in c_res.reasons)


# ==============================================================================
# 6. Cross-Field Consistency Tests
# ==============================================================================

def test_cross_field_consistency():
    checker = CrossFieldConsistencyChecker()

    # 1. Words to number conversion
    assert checker.words_to_number("Four Thousand Seven Hundred and Twenty") == 4720.0
    assert checker.words_to_number("One Lakh Fifty Thousand") == 150000.0

    # 2. Consistent figures and words
    ok1, disc1 = checker.check_amount_consistency("4720.00", "Four Thousand Seven Hundred and Twenty")
    assert ok1 is True
    assert disc1 is None

    # 3. Discrepancy between figures and words
    ok2, disc2 = checker.check_amount_consistency("5000.00", "Four Thousand Seven Hundred and Twenty")
    assert ok2 is False
    assert disc2 is not None
    assert disc2.check_name == "amount_words_vs_figures_mismatch"
    assert disc2.severity == "CRITICAL"

    # 4. Bank name vs IFSC prefix match
    ok_bank1, disc_bank1 = checker.check_bank_vs_ifsc("CNRB0002854", "Canara Bank")
    assert ok_bank1 is True

    # 5. Bank name vs IFSC prefix mismatch
    ok_bank2, disc_bank2 = checker.check_bank_vs_ifsc("CNRB0002854", "HDFC Bank")
    assert ok_bank2 is False
    assert disc_bank2 is not None

    # 6. Signature presence
    ok_sign, disc_sign = checker.check_signature_presence(signature_present=False, signature_confidence=0.1)
    assert ok_sign is False
    assert disc_sign.check_name == "missing_or_unconfirmed_signature"


# ==============================================================================
# 7. Review Queue Manager & Processing Status Tests
# ==============================================================================

def test_review_queue_statuses_and_preservation():
    ref_date = datetime.date(2022, 6, 1)
    manager = ReviewQueueManager(reference_date=ref_date)

    # 1. High-confidence valid cheque -> VERIFIED
    clean_cheque = {
        "cheque_id": "cheque_verified_001",
        "overall_confidence": 0.88,
        "fields": {
            "acno": {"value": "001401000213", "raw_value": "Nc: 001401000213", "confidence": 0.88},
            "ifsc": {"value": "CNRB0002854", "raw_value": "IFSC : CNRB0002854", "confidence": 0.92},
            "date": {"value": "15/05/2022", "raw_value": "15/05/2022", "confidence": 0.89},
            "amount": {"value": "4720", "raw_value": "4720", "confidence": 0.87},
            "name": {"value": "Ethel M. Bryson", "raw_value": "Pay Ethel M. Bryson", "confidence": 0.85},
        },
        "signatures": {"present": True, "confidence": 0.99},
    }
    item_verified = manager.evaluate_cheque(clean_cheque, bank_name="Canara Bank")
    assert item_verified.status == ChequeProcessingStatus.VERIFIED
    assert item_verified.review_required is False
    assert item_verified.fields["acno"].raw_model_output == "Nc: 001401000213"
    assert item_verified.fields["acno"].normalized_value == "001401000213"
    assert item_verified.fields["acno"].validation_status == "VALID"

    # 2. Stale cheque with low confidence -> REVIEW_REQUIRED
    stale_cheque = {
        "cheque_id": "cheque_review_002",
        "overall_confidence": 0.62,
        "fields": {
            "acno": {"value": "30002010108841", "raw_value": "30002010108841", "confidence": 0.45},
            "ifsc": {"value": "SYNB0003011", "raw_value": "SYNB0003011", "confidence": 0.70},
            "date": {"value": "01/01/2022", "raw_value": "01/01/2022", "confidence": 0.65},  # Stale!
            "amount": {"value": "1420", "raw_value": "1420", "confidence": 0.50},
            "name": {"value": "Marcel Achen", "raw_value": "Marcel Achen", "confidence": 0.75},
        },
        "signatures": {"present": True, "confidence": 0.95},
    }
    item_review = manager.evaluate_cheque(stale_cheque)
    assert item_review.status == ChequeProcessingStatus.REVIEW_REQUIRED
    assert item_review.review_required is True
    assert any("Stale cheque" in r for r in item_review.reasons_for_review)
    assert any("[ACNO]" in r and "falls below threshold" in r for r in item_review.reasons_for_review)

    # 3. Fatal failure (zero amount and non-numeric account number) -> INVALID
    invalid_cheque = {
        "cheque_id": "cheque_invalid_003",
        "overall_confidence": 0.50,
        "fields": {
            "acno": {"value": "INVALID_ACC", "raw_value": "INVALID_ACC", "confidence": 0.30},
            "ifsc": {"value": "CNRB0002854", "raw_value": "CNRB0002854", "confidence": 0.80},
            "date": {"value": "15/05/2022", "raw_value": "15/05/2022", "confidence": 0.80},
            "amount": {"value": "0.00", "raw_value": "0.00", "confidence": 0.70},
            "name": {"value": "Payee", "raw_value": "Payee", "confidence": 0.80},
        },
        "signatures": {"present": True, "confidence": 0.90},
    }
    item_invalid = manager.evaluate_cheque(invalid_cheque)
    assert item_invalid.status == ChequeProcessingStatus.INVALID
    assert item_invalid.review_required is True
    assert len(item_invalid.fields["acno"].validation_errors) > 0


def test_no_fraudulent_claim_invariant():
    """Verifies that ChequeSense never claims a cheque is fraudulent on anomaly alone."""
    manager = ReviewQueueManager()
    suspicious_cheque = {
        "cheque_id": "anomaly_sample_999",
        "overall_confidence": 0.20,
        "fields": {
            "acno": {"value": "123456789", "raw_value": "???", "confidence": 0.15},
            "ifsc": {"value": "CNRB0002854", "raw_value": "CNRB0002854", "confidence": 0.20},
            "date": {"value": "01/01/2000", "raw_value": "01/01/2000", "confidence": 0.25},
            "amount": {"value": "999999999", "raw_value": "999999999", "confidence": 0.10},
            "name": {"value": "Cash", "raw_value": "Cash", "confidence": 0.20},
        },
        "signatures": {"present": False, "confidence": 0.0},
    }
    item = manager.evaluate_cheque(suspicious_cheque)

    # Invariant: Must use auditable operational statuses, NEVER label as 'FRAUD' or 'FRAUDULENT'
    assert item.status in (ChequeProcessingStatus.REVIEW_REQUIRED, ChequeProcessingStatus.INVALID)
    assert item.status.value != "FRAUD"
    assert item.status.value != "FRAUDULENT"
    for reason in item.reasons_for_review:
        assert "fraud" not in reason.lower()
