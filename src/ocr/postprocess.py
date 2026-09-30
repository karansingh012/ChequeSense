"""Post-processing and deterministic text normalization for cheque fields.

Cleans OCR artifacts, normalizes whitespace, sanitizes numerical strings,
preserves leading zeros, and standardizes field-specific formats without unnecessary NLP.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from src.ocr.ocr_engine import OCRResult, OCRWordToken


@dataclass
class NormalizedFieldResult:
    """Post-processed and sanitized field recognition result."""

    field_type: str
    raw_text: str
    normalized_text: str
    is_valid: bool
    confidence: float
    cleaning_steps_applied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_type": self.field_type,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "is_valid": self.is_valid,
            "confidence": round(self.confidence, 4),
            "cleaning_steps": self.cleaning_steps_applied,
            "metadata": self.metadata,
        }


class FieldNormalizer:
    """Deterministic normalizer tailored for banking and cheque fields."""

    # Common OCR character confusions for numerical / alphanumeric corrections
    CHAR_TO_DIGIT = {
        "O": "0", "o": "0", "D": "0", "Q": "0",
        "I": "1", "l": "1", "|": "1", "!": "1",
        "Z": "2", "z": "2",
        "B": "8",
        "S": "5", "s": "5",
        "G": "6",
        "T": "7",
    }

    DIGIT_TO_CHAR = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "8": "B",
    }

    # Boilerplate prefixes/suffixes in cheque fields
    NAME_BOILERPLATE = [
        r"^\s*pay\b",
        r"^\s*self\b",
        r"\bor\s+bearer\b",
        r"\ba/c\s+payee\b",
        r"\baccount\s+payee\b",
        r"\bonly\b",
        r"[\*#~|]+",
    ]

    AMOUNT_WORDS_BOILERPLATE = [
        r"\brupees\b",
        r"\bonly\b",
        r"[\*#~|/=-]+",
    ]

    def normalize(
        self,
        raw_text: str,
        field_type: str = "general",
        confidence: float = 1.0,
    ) -> NormalizedFieldResult:
        """Routes raw OCR string to field-specific deterministic normalizer."""
        field_lower = field_type.lower().strip()

        if field_lower in ("acno", "account_number", "acc_no"):
            return self.normalize_account_number(raw_text, confidence)
        elif field_lower in ("ifsc", "ifsc_code"):
            return self.normalize_ifsc(raw_text, confidence)
        elif field_lower in ("date", "cheque_date"):
            return self.normalize_date(raw_text, confidence)
        elif field_lower in ("amount", "amt_in_figures", "courtesy_amount"):
            return self.normalize_amount_figures(raw_text, confidence)
        elif field_lower in ("name", "payee_name", "payee"):
            return self.normalize_payee_name(raw_text, confidence)
        elif field_lower in ("amt_in_words", "legal_amount", "amount_words"):
            return self.normalize_amount_words(raw_text, confidence)
        elif field_lower in ("bank_name", "bank"):
            return self.normalize_bank_name(raw_text, confidence)
        else:
            return self.normalize_general_text(raw_text, field_type, confidence)

    def normalize_account_number(self, raw_text: str, confidence: float = 1.0) -> NormalizedFieldResult:
        """Normalizes bank account numbers: preserves leading zeros, removes non-digits.

        Indian bank account numbers typically span 9 to 18 digits.
        Crucial requirement: Leading zeros MUST NOT be dropped (e.g. '001401000213').
        """
        steps = []
        text = raw_text

        # 1. Strip common prefixes like 'A/c No:', 'Acc No.', 'No.'
        cleaned = re.sub(r"(?i)\b(a/c|acc|account|no|num|number)\b[:.]*", "", text)
        if cleaned != text:
            steps.append("stripped_acno_label_prefix")

        # 2. Extract continuous digits or convert common OCR letter confusions
        candidate = []
        for ch in cleaned:
            if ch.isdigit():
                candidate.append(ch)
            elif ch in self.CHAR_TO_DIGIT:
                candidate.append(self.CHAR_TO_DIGIT[ch])
                steps.append(f"mapped_char_{ch}_to_digit")

        digits_only = "".join(candidate)
        steps.append("extracted_digits_preserving_leading_zeros")

        # Valid Indian bank account number rule: 9 to 18 digits
        is_valid = bool(re.match(r"^\d{9,18}$", digits_only))

        return NormalizedFieldResult(
            field_type="acno",
            raw_text=raw_text,
            normalized_text=digits_only,
            is_valid=is_valid,
            confidence=confidence,
            cleaning_steps_applied=list(set(steps)),
            metadata={
                "digit_count": len(digits_only),
                "has_leading_zero": digits_only.startswith("0") if digits_only else False,
            },
        )

    def normalize_ifsc(self, raw_text: str, confidence: float = 1.0) -> NormalizedFieldResult:
        """Normalizes Indian Financial System Code (IFSC).

        IFSC Standard Structure: 11 characters:
        - Characters 1-4: Alphabetic bank code (e.g. 'SBIN', 'HDFC', 'CNRB')
        - Character 5: Always '0' (digit zero)
        - Characters 6-11: Alphanumeric branch code
        """
        steps = []
        text = raw_text.upper()

        # 1. Remove label noise
        cleaned = re.sub(r"(?i)\b(ifsc|code|rtgs|neft|branch)\b[:.]*", "", text)
        if cleaned != text:
            steps.append("removed_ifsc_labels")

        # 2. Strip whitespace, hyphens, and non-alphanumeric noise
        alphanumeric = re.sub(r"[^A-Z0-9]", "", cleaned)
        steps.append("stripped_non_alphanumeric")

        # 3. Match 11-char pattern or attempt corrective alignment
        normalized = alphanumeric
        if len(alphanumeric) >= 11:
            # If extra characters exist, locate candidate matching 4 letters + 0 + 6 chars
            match = re.search(r"[A-Z0-9]{11}", alphanumeric)
            if match:
                normalized = match.group(0)

        # 4. Correct 5th character: MUST BE '0' (digit zero)
        chars = list(normalized)
        if len(chars) == 11:
            # First 4 must be letters
            for idx in range(4):
                if chars[idx].isdigit() and chars[idx] in self.DIGIT_TO_CHAR:
                    chars[idx] = self.DIGIT_TO_CHAR[chars[idx]]
                    steps.append(f"corrected_bank_letter_at_{idx}")

            # 5th char must be '0'
            if chars[4] in ("O", "o", "Q", "D"):
                chars[4] = "0"
                steps.append("corrected_5th_char_O_to_0")
            normalized = "".join(chars)

        is_valid = bool(re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", normalized))

        bank_prefix = normalized[:4] if len(normalized) >= 4 else ""
        branch_code = normalized[5:11] if len(normalized) == 11 else ""

        return NormalizedFieldResult(
            field_type="ifsc",
            raw_text=raw_text,
            normalized_text=normalized,
            is_valid=is_valid,
            confidence=confidence,
            cleaning_steps_applied=list(set(steps)),
            metadata={
                "bank_code": bank_prefix,
                "branch_code": branch_code,
                "length": len(normalized),
            },
        )

    def normalize_date(self, raw_text: str, confidence: float = 1.0) -> NormalizedFieldResult:
        """Normalizes date fields, standardizing to DD/MM/YYYY with preserved leading zeros."""
        steps = []
        text = raw_text

        # 1. Strip label prefixes
        cleaned = re.sub(r"(?i)\b(date|dt|dated)\b[:.]*", " ", text).strip()

        # 2. Strip printed guide characters: 'DDMM YYYY', 'D D M M Y Y Y Y'
        guide_stripped = re.sub(r"(?i)\b[d\s]{1,4}[m\s]{1,4}[y\s]{1,8}\b", " ", cleaned)
        guide_stripped = re.sub(r"(?i)ddmm\s*y*", " ", guide_stripped)
        if guide_stripped != cleaned:
            steps.append("stripped_printed_date_guide_labels")
            cleaned = guide_stripped

        # Extract digits
        digits_only = "".join(c for c in cleaned if c.isdigit())
        normalized = ""
        is_valid = False

        if len(digits_only) == 8:
            dd = digits_only[0:2]
            mm = digits_only[2:4]
            yyyy = digits_only[4:8]
            normalized = f"{dd}/{mm}/{yyyy}"
            steps.append("formatted_from_8_digits")
        elif len(digits_only) == 6:
            dd = digits_only[0:2]
            mm = digits_only[2:4]
            yy = digits_only[4:6]
            year = int(yy)
            # Century assumption: 2000s for modern cheques
            yyyy = f"20{yy}" if year < 50 else f"19{yy}"
            normalized = f"{dd}/{mm}/{yyyy}"
            steps.append("formatted_from_6_digits_with_century")
        else:
            # Check for delimiter separated: DD/MM/YY or DD-MM-YYYY
            match = re.search(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})", cleaned)
            if match:
                d, m, y = match.groups()
                dd = f"{int(d):02d}"
                mm = f"{int(m):02d}"
                yyyy = y if len(y) == 4 else f"20{y}"
                normalized = f"{dd}/{mm}/{yyyy}"
                steps.append("formatted_from_delimited_pattern")

        # Validate day and month ranges
        if normalized:
            try:
                parts = normalized.split("/")
                d_val, m_val, y_val = int(parts[0]), int(parts[1]), int(parts[2])
                if 1 <= d_val <= 31 and 1 <= m_val <= 12 and 1990 <= y_val <= 2050:
                    is_valid = True
            except ValueError:
                is_valid = False

        return NormalizedFieldResult(
            field_type="date",
            raw_text=raw_text,
            normalized_text=normalized or raw_text.strip(),
            is_valid=is_valid,
            confidence=confidence,
            cleaning_steps_applied=steps,
            metadata={"formatted_date": normalized},
        )

    def normalize_amount_figures(self, raw_text: str, confidence: float = 1.0) -> NormalizedFieldResult:
        """Normalizes courtesy numerical amount: removes ₹, Rs, /-, commas, leading symbols."""
        steps = []
        text = raw_text

        # 1. Remove currency symbols and noise characters
        cleaned = re.sub(r"(?i)[₹\$\€\£]|rs\.?|inr|/-|/=|=", "", text)
        if cleaned != text:
            steps.append("stripped_currency_symbols")

        # 2. Remove commas, spaces, dashes
        cleaned = re.sub(r"[\s,\*#_]", "", cleaned)

        # 3. Match decimal or integer numerical amount
        match = re.search(r"\d+(\.\d{1,2})?", cleaned)
        if match:
            normalized = match.group(0)
            steps.append("extracted_valid_numeric_amount")
            is_valid = float(normalized) > 0.0
        else:
            normalized = ""
            is_valid = False

        return NormalizedFieldResult(
            field_type="amount",
            raw_text=raw_text,
            normalized_text=normalized,
            is_valid=is_valid,
            confidence=confidence,
            cleaning_steps_applied=steps,
            metadata={"amount_value": float(normalized) if is_valid else None},
        )

    def normalize_payee_name(self, raw_text: str, confidence: float = 1.0) -> NormalizedFieldResult:
        """Normalizes payee recipient names: removes boilerplate, standardizes whitespace and case."""
        steps = []
        text = raw_text

        # 1. Strip boilerplate patterns
        cleaned = text
        for bp in self.NAME_BOILERPLATE:
            prev = cleaned
            cleaned = re.sub(bp, " ", cleaned, flags=re.IGNORECASE)
            if prev != cleaned:
                steps.append(f"removed_name_boilerplate_{bp}")

        # 2. Remove non-name punctuation (keep dots and hyphens in names like 'Ethel M. Bryson')
        cleaned = re.sub(r"[^\w\s\.\-]", " ", cleaned)

        # 3. Normalize whitespace (collapse multiple spaces, tabs, newlines)
        normalized = " ".join(cleaned.split()).strip()
        steps.append("collapsed_whitespace")

        # 4. Standardize casing: Title Case if all uppercase or lowercase
        if normalized.isupper() or normalized.islower():
            normalized = normalized.title()
            steps.append("standardized_to_title_case")

        # Valid payee name: at least 2 alphabetic characters
        is_valid = bool(re.search(r"[A-Za-z]{2,}", normalized))

        return NormalizedFieldResult(
            field_type="name",
            raw_text=raw_text,
            normalized_text=normalized,
            is_valid=is_valid,
            confidence=confidence,
            cleaning_steps_applied=list(set(steps)),
            metadata={"word_count": len(normalized.split()) if normalized else 0},
        )

    def normalize_amount_words(self, raw_text: str, confidence: float = 1.0) -> NormalizedFieldResult:
        """Normalizes legal amount written in words."""
        steps = []
        cleaned = raw_text

        # Remove boilerplate
        for bp in self.AMOUNT_WORDS_BOILERPLATE:
            cleaned = re.sub(bp, " ", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"[^\w\s\-]", " ", cleaned)
        normalized = " ".join(cleaned.split()).strip().title()
        steps.append("normalized_whitespace_and_title_case")

        is_valid = len(normalized.split()) >= 1 and bool(re.search(r"[A-Za-z]", normalized))

        return NormalizedFieldResult(
            field_type="amt_in_words",
            raw_text=raw_text,
            normalized_text=normalized,
            is_valid=is_valid,
            confidence=confidence,
            cleaning_steps_applied=steps,
            metadata={"words": normalized.split()},
        )

    def normalize_bank_name(self, raw_text: str, confidence: float = 1.0) -> NormalizedFieldResult:
        """Normalizes printed bank title header."""
        cleaned = " ".join(re.sub(r"[^\w\s\.\-&]", " ", raw_text).split()).strip().title()
        return NormalizedFieldResult(
            field_type="bank_name",
            raw_text=raw_text,
            normalized_text=cleaned,
            is_valid=len(cleaned) >= 3,
            confidence=confidence,
            cleaning_steps_applied=["cleaned_punctuation_and_whitespace"],
        )

    def normalize_general_text(self, raw_text: str, field_type: str, confidence: float = 1.0) -> NormalizedFieldResult:
        """General text cleaning fallback."""
        cleaned = " ".join(raw_text.split()).strip()
        return NormalizedFieldResult(
            field_type=field_type,
            raw_text=raw_text,
            normalized_text=cleaned,
            is_valid=len(cleaned) > 0,
            confidence=confidence,
            cleaning_steps_applied=["collapsed_whitespace"],
        )
