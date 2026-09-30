"""Cross-field consistency verification and multi-modal reconciliation for ChequeSense."""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("chequesense.validation.consistency")


@dataclass
class DiscrepancyDetail:
    """Detail of a consistency discrepancy across two or more cheque fields."""

    check_name: str
    fields_involved: List[str]
    description: str
    severity: str = "WARNING"  # 'CRITICAL' or 'WARNING'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "fields_involved": self.fields_involved,
            "description": self.description,
            "severity": self.severity,
        }


@dataclass
class ConsistencyCheckResult:
    """Consolidated outcome of cross-field consistency analysis."""

    all_passed: bool
    checks_evaluated: List[str] = field(default_factory=list)
    discrepancies: List[DiscrepancyDetail] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "checks_evaluated": self.checks_evaluated,
            "discrepancy_count": len(self.discrepancies),
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "metadata": self.metadata,
        }


class CrossFieldConsistencyChecker:
    """Verifies semantic coherence across distinct cheque regions."""

    # Bank IFSC 4-letter prefix mapping
    BANK_IFSC_MAP = {
        "SBIN": ["STATE BANK", "SBI"],
        "HDFC": ["HDFC"],
        "ICIC": ["ICICI"],
        "CNRB": ["CANARA"],
        "SYNB": ["SYNDICATE", "CANARA"],
        "PUNB": ["PUNJAB NATIONAL", "PNB"],
        "BARB": ["BANK OF BARODA", "BOB"],
        "UTIB": ["AXIS"],
        "KKBK": ["KOTAK"],
        "UBIN": ["UNION BANK"],
        "IDIB": ["INDIAN BANK"],
        "MAHB": ["MAHARASHTRA"],
    }

    # Number word lookup table
    WORD_TO_NUMBER = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }

    SCALES = {
        "hundred": 100,
        "thousand": 1000,
        "lakh": 100000,
        "lakhs": 100000,
        "lac": 100000,
        "lacs": 100000,
        "crore": 10000000,
        "crores": 10000000,
        "million": 1000000,
    }

    def words_to_number(self, words_str: str) -> Optional[float]:
        """Converts Indian and Western English amount words to numerical float."""
        clean = re.sub(r"[^\w\s]", " ", words_str.lower())
        tokens = [w for w in clean.split() if w not in ("and", "rupees", "rupee", "only", "paise")]

        if not tokens:
            return None

        total = 0.0
        current = 0.0

        for tok in tokens:
            if tok in self.WORD_TO_NUMBER:
                current += self.WORD_TO_NUMBER[tok]
            elif tok in self.SCALES:
                scale = self.SCALES[tok]
                if current == 0:
                    current = 1.0
                if scale >= 1000:
                    total += current * scale
                    current = 0.0
                else:
                    current *= scale
            else:
                # Unrecognized word token
                pass

        total += current
        return total if total > 0 else None

    def check_amount_consistency(
        self, amount_figures_str: str, amount_words_str: str
    ) -> Tuple[bool, Optional[DiscrepancyDetail]]:
        """Compares courtesy numerical amount with legal amount written in words."""
        if not amount_figures_str or not amount_words_str:
            return True, None

        # Clean figures
        cleaned_fig = re.sub(r"[^\d.]", "", amount_figures_str)
        if not cleaned_fig:
            return True, None

        try:
            fig_val = float(cleaned_fig)
        except ValueError:
            return True, None

        word_val = self.words_to_number(amount_words_str)
        if word_val is None:
            # Could not reliably parse words
            return True, None

        # Allow minor rounding discrepancy for decimal paise
        if abs(fig_val - word_val) > 1.0:
            return False, DiscrepancyDetail(
                check_name="amount_words_vs_figures_mismatch",
                fields_involved=["amount", "amt_in_words"],
                description=f"Mismatch between amount in figures (₹{fig_val:,.2f}) and amount in words (₹{word_val:,.2f})",
                severity="CRITICAL",
            )

        return True, None

    def check_bank_vs_ifsc(
        self, ifsc_str: str, bank_name_str: str
    ) -> Tuple[bool, Optional[DiscrepancyDetail]]:
        """Verifies that the IFSC code prefix aligns with the declared bank name."""
        if not ifsc_str or not bank_name_str or len(ifsc_str) < 4:
            return True, None

        prefix = ifsc_str[:4].upper()
        bank_upper = bank_name_str.upper()

        expected_patterns = self.BANK_IFSC_MAP.get(prefix)
        if expected_patterns:
            matches = any(pat in bank_upper for pat in expected_patterns)
            if not matches:
                return False, DiscrepancyDetail(
                    check_name="bank_name_ifsc_prefix_mismatch",
                    fields_involved=["ifsc", "bank_name"],
                    description=f"IFSC prefix '{prefix}' does not match bank name '{bank_name_str}'",
                    severity="WARNING",
                )

        return True, None

    def check_signature_presence(
        self, signature_present: bool, signature_confidence: float
    ) -> Tuple[bool, Optional[DiscrepancyDetail]]:
        """Verifies that the authorized signatory area contains a confirmed signature stroke."""
        if not signature_present or signature_confidence < 0.50:
            return False, DiscrepancyDetail(
                check_name="missing_or_unconfirmed_signature",
                fields_involved=["sign"],
                description=f"Signature mark not detected with sufficient confidence ({signature_confidence:.2f} < 0.50)",
                severity="CRITICAL",
            )
        return True, None

    def evaluate_all(
        self,
        fields: Dict[str, Any],
        signature_present: bool = True,
        signature_confidence: float = 1.0,
        bank_name: Optional[str] = None,
    ) -> ConsistencyCheckResult:
        """Executes full suite of cross-field consistency checks."""
        evaluated = []
        discrepancies: List[DiscrepancyDetail] = []

        # 1. Amount Figures vs Amount Words
        evaluated.append("amount_figures_vs_words")
        fig_obj = fields.get("amount")
        words_obj = fields.get("amt_in_words")

        fig_str = getattr(fig_obj, "value", str(fig_obj or ""))
        words_str = getattr(words_obj, "value", str(words_obj or ""))

        ok_amt, disc_amt = self.check_amount_consistency(fig_str, words_str)
        if not ok_amt and disc_amt:
            discrepancies.append(disc_amt)

        # 2. Bank Name vs IFSC
        evaluated.append("bank_name_vs_ifsc")
        ifsc_obj = fields.get("ifsc")
        ifsc_str = getattr(ifsc_obj, "value", str(ifsc_obj or ""))
        b_name = bank_name or (getattr(fields.get("bank_name"), "value", None))

        if ifsc_str and b_name:
            ok_bank, disc_bank = self.check_bank_vs_ifsc(ifsc_str, b_name)
            if not ok_bank and disc_bank:
                discrepancies.append(disc_bank)

        # 3. Signature Verification Presence
        evaluated.append("signature_presence")
        ok_sign, disc_sign = self.check_signature_presence(signature_present, signature_confidence)
        if not ok_sign and disc_sign:
            discrepancies.append(disc_sign)

        all_passed = len(discrepancies) == 0

        return ConsistencyCheckResult(
            all_passed=all_passed,
            checks_evaluated=evaluated,
            discrepancies=discrepancies,
        )
