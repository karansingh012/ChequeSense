"""Field format, numerical, and date validation rules for ChequeSense."""

from __future__ import annotations

import calendar
import datetime
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("chequesense.validation.field")


class ChequeProcessingStatus(str, Enum):
    """Auditable status of cheque processing according to banking compliance rules."""

    PROCESSED = "PROCESSED"              # Successfully extracted, initial validations pass
    VERIFIED = "VERIFIED"                # High confidence, all schemas & consistency checks passed (eligible for STP)
    REVIEW_REQUIRED = "REVIEW_REQUIRED"  # Requires human teller review (low confidence, stale date, unverified sign)
    INVALID = "INVALID"                  # Fatal business rule failure (non-positive amount, invalid account number)


@dataclass
class FieldValidationResult:
    """Outcome of validation on an individual cheque field."""

    field_name: str
    is_valid: bool
    raw_value: str
    normalized_value: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "is_valid": self.is_valid,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class FieldValidator:
    """Validates banking data formats, calendar dates, and numerical constraints."""

    # Known Indian bank IFSC prefixes
    KNOWN_IFSC_BANKS = {
        "SBIN": "State Bank of India",
        "HDFC": "HDFC Bank",
        "ICIC": "ICICI Bank",
        "CNRB": "Canara Bank",
        "SYNB": "Syndicate Bank",
        "PUNB": "Punjab National Bank",
        "BARB": "Bank of Baroda",
        "UTIB": "Axis Bank",
        "KKBK": "Kotak Mahindra Bank",
        "UBIN": "Union Bank of India",
        "IDIB": "Indian Bank",
        "MAHB": "Bank of Maharashtra",
    }

    # RBI guidelines: Cheques are valid for 3 months (90 days) from date of issue
    CHEQUE_VALIDITY_DAYS = 90

    # High value clearing threshold requiring Positive Pay verification in India
    POSITIVE_PAY_THRESHOLD = 500000.0

    def validate_account_number(self, value: str, raw_value: str = "") -> FieldValidationResult:
        """Validates bank account number with strict leading-zero preservation.

        Indian bank account numbers span 9 to 18 digits.
        Leading zeros are semantically meaningful in banking and MUST NOT be dropped.
        """
        raw = raw_value or value
        val = str(value).strip()
        errors = []
        warnings = []

        if not val:
            errors.append("Account number is empty")
            return FieldValidationResult("acno", False, raw, "", errors, warnings)

        # 1. Non-digit check
        if not val.isdigit():
            errors.append(f"Account number contains non-numeric characters: '{val}'")
            return FieldValidationResult("acno", False, raw, val, errors, warnings)

        # 2. Length check (Indian banking standards: 9 to 18 digits)
        length = len(val)
        if length < 9:
            errors.append(f"Account number length ({length}) is below minimum standard of 9 digits")
        elif length > 18:
            errors.append(f"Account number length ({length}) exceeds maximum standard of 18 digits")

        # 3. Leading zero check (preservation verified)
        has_leading_zero = val.startswith("0")
        if has_leading_zero:
            logger.debug("Preserved leading zero in account number '%s'", val)

        is_valid = len(errors) == 0

        return FieldValidationResult(
            field_name="acno",
            is_valid=is_valid,
            raw_value=raw,
            normalized_value=val,
            errors=errors,
            warnings=warnings,
            metadata={
                "digit_count": length,
                "has_leading_zero": has_leading_zero,
            },
        )

    def validate_ifsc(self, value: str, raw_value: str = "") -> FieldValidationResult:
        """Validates Indian Financial System Code (IFSC).

        Pattern: 4 alphabetic letters + '0' + 6 alphanumeric characters (total 11 chars).
        """
        raw = raw_value or value
        val = str(value).strip().upper()
        errors = []
        warnings = []

        if not val:
            errors.append("IFSC code is empty")
            return FieldValidationResult("ifsc", False, raw, "", errors, warnings)

        if len(val) != 11:
            errors.append(f"IFSC code length ({len(val)}) is not exactly 11 characters")

        bank_prefix = val[:4]
        if not bank_prefix.isalpha():
            errors.append(f"IFSC bank prefix '{bank_prefix}' must consist of 4 alphabetic letters")

        if len(val) >= 5 and val[4] != "0":
            errors.append(f"IFSC 5th character '{val[4]}' must be digit zero ('0')")

        branch_suffix = val[5:] if len(val) >= 6 else ""
        if len(val) == 11 and not branch_suffix.isalnum():
            errors.append(f"IFSC branch suffix '{branch_suffix}' must be alphanumeric")

        known_bank = self.KNOWN_IFSC_BANKS.get(bank_prefix, "Unknown Bank")

        is_valid = len(errors) == 0

        return FieldValidationResult(
            field_name="ifsc",
            is_valid=is_valid,
            raw_value=raw,
            normalized_value=val,
            errors=errors,
            warnings=warnings,
            metadata={
                "bank_prefix": bank_prefix,
                "bank_name": known_bank,
                "branch_suffix": branch_suffix,
            },
        )

    def validate_date(
        self,
        value: str,
        raw_value: str = "",
        reference_date: Optional[datetime.date] = None,
    ) -> FieldValidationResult:
        """Validates calendar date, preserving leading zeros, and checks cheque validity window.

        Format expected: DD/MM/YYYY.
        Checks:
        - Calendar validity (e.g. leap years, valid days in month).
        - Preserves leading zeros: day '06', month '05'.
        - Stale cheque: issue date older than 90 days relative to reference_date.
        - Post-dated cheque: issue date in future relative to reference_date.
        """
        raw = raw_value or value
        val = str(value).strip()
        errors = []
        warnings = []

        if not val:
            errors.append("Date field is empty")
            return FieldValidationResult("date", False, raw, "", errors, warnings)

        # Standardize delimiter
        parts = re.split(r"[\/\-\.]", val)
        if len(parts) != 3:
            errors.append(f"Invalid date format '{val}'. Expected DD/MM/YYYY")
            return FieldValidationResult("date", False, raw, val, errors, warnings)

        day_str, month_str, year_str = parts[0], parts[1], parts[2]

        # Verify digits
        if not (day_str.isdigit() and month_str.isdigit() and year_str.isdigit()):
            errors.append(f"Date contains non-numeric components: '{val}'")
            return FieldValidationResult("date", False, raw, val, errors, warnings)

        day = int(day_str)
        month = int(month_str)
        year = int(year_str)

        # Expand 2-digit year (banking standard assumption: 2000s)
        if year < 100:
            year = 2000 + year

        # Preserved formatted string with leading zeros
        normalized_date_str = f"{day:02d}/{month:02d}/{year:04d}"

        # Calendar check
        if month < 1 or month > 12:
            errors.append(f"Invalid month ({month}). Must be 1-12")
            return FieldValidationResult("date", False, raw, normalized_date_str, errors, warnings)

        _, max_days = calendar.monthrange(year, month)
        if day < 1 or day > max_days:
            errors.append(f"Invalid day ({day}) for month {month}/{year}. Max days is {max_days}")
            return FieldValidationResult("date", False, raw, normalized_date_str, errors, warnings)

        parsed_date = datetime.date(year, month, day)

        # Validity window check against reference_date (if provided)
        is_stale = False
        is_post_dated = False
        days_difference = 0

        if reference_date is not None:
            delta = (reference_date - parsed_date).days
            days_difference = delta

            if delta > self.CHEQUE_VALIDITY_DAYS:
                is_stale = True
                warnings.append(
                    f"Stale cheque: issued {delta} days ago on {normalized_date_str}, exceeding {self.CHEQUE_VALIDITY_DAYS}-day validity"
                )
            elif delta < 0:
                is_post_dated = True
                warnings.append(
                    f"Post-dated cheque: dated in future on {normalized_date_str} (relative to {reference_date})"
                )

        is_valid = len(errors) == 0

        return FieldValidationResult(
            field_name="date",
            is_valid=is_valid,
            raw_value=raw,
            normalized_value=normalized_date_str,
            errors=errors,
            warnings=warnings,
            metadata={
                "parsed_date": parsed_date.isoformat(),
                "day": f"{day:02d}",
                "month": f"{month:02d}",
                "year": year,
                "is_stale": is_stale,
                "is_post_dated": is_post_dated,
                "days_difference": days_difference,
            },
        )

    def validate_amount(self, value: str, raw_value: str = "") -> FieldValidationResult:
        """Validates courtesy numerical amount.

        Rules:
        - Must be positive numerical float > 0.
        - Flags amounts requiring Positive Pay confirmation (>= 500,000 INR).
        """
        raw = raw_value or value
        val = str(value).strip().replace(",", "")
        errors = []
        warnings = []

        if not val:
            errors.append("Amount field is empty")
            return FieldValidationResult("amount", False, raw, "", errors, warnings)

        try:
            amt_num = float(val)
        except ValueError:
            errors.append(f"Amount '{val}' is not a valid numeric value")
            return FieldValidationResult("amount", False, raw, val, errors, warnings)

        if amt_num <= 0.0:
            errors.append(f"Amount {amt_num} must be strictly greater than zero")

        if amt_num >= self.POSITIVE_PAY_THRESHOLD:
            warnings.append(
                f"High-value cheque (₹{amt_num:,.2f} >= ₹{self.POSITIVE_PAY_THRESHOLD:,.2f}): Requires Positive Pay verification"
            )

        is_valid = len(errors) == 0
        normalized_str = f"{amt_num:.2f}" if "." in val else f"{int(amt_num)}"

        return FieldValidationResult(
            field_name="amount",
            is_valid=is_valid,
            raw_value=raw,
            normalized_value=normalized_str,
            errors=errors,
            warnings=warnings,
            metadata={
                "amount_numeric": amt_num,
                "is_high_value": amt_num >= self.POSITIVE_PAY_THRESHOLD,
            },
        )

    def validate_payee_name(self, value: str, raw_value: str = "") -> FieldValidationResult:
        """Validates payee recipient name."""
        raw = raw_value or value
        val = str(value).strip()
        errors = []
        warnings = []

        if not val:
            errors.append("Payee recipient name is empty")
            return FieldValidationResult("name", False, raw, "", errors, warnings)

        if len(val) < 2:
            errors.append(f"Payee name '{val}' is too short (minimum 2 characters)")

        # Alphabetic character count check
        alpha_chars = sum(1 for c in val if c.isalpha())
        if alpha_chars < 2:
            errors.append("Payee name must contain at least 2 alphabetic characters")

        if val.upper() in ("SELF", "CASH"):
            warnings.append(f"Payee is designated as '{val.upper()}' (Cash withdrawal instrument)")

        is_valid = len(errors) == 0

        return FieldValidationResult(
            field_name="name",
            is_valid=is_valid,
            raw_value=raw,
            normalized_value=val,
            errors=errors,
            warnings=warnings,
            metadata={"character_count": len(val), "alpha_count": alpha_chars},
        )

    def validate_required_fields(
        self, fields: Dict[str, Any], required_fields: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """Verifies presence and non-emptiness of all mandatory cheque fields."""
        required = required_fields or ["acno", "amount", "date", "name", "ifsc"]
        missing = []

        for req in required:
            field_obj = fields.get(req)
            if field_obj is None:
                missing.append(f"Mandatory field '{req}' missing from extraction")
            elif hasattr(field_obj, "value") and not str(field_obj.value).strip():
                missing.append(f"Mandatory field '{req}' has empty value")
            elif isinstance(field_obj, dict) and not str(field_obj.get("value", "")).strip():
                missing.append(f"Mandatory field '{req}' has empty value")

        return len(missing) == 0, missing
