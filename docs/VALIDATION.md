# ChequeSense: Validation & Human-Review Layer
**Document Version:** 1.0.0  
**Date:** October 1, 2026  
**Module:** `src/validation/` (Phase 6)  
**Role:** Lead Machine Learning Engineer  
**Status:** Validated, Audited & Unit-Tested

---

## 1. Executive Summary & Compliance Context

Phase 6 implements the **Validation and Human-Review Layer** for the ChequeSense banking automation platform. In commercial banking and the Indian Cheque Truncation System (CTS-2010), optical and AI extractions must never be accepted into core banking ledgers without deterministic business rule enforcement, confidence gating, cross-field reconciliation, and auditable teller exception routing.

### 1.1 Fundamental Compliance Mandate: Zero Unsubstantiated Fraud Claims

> [!IMPORTANT]
> **Strict Operational Rule:** Anomaly detection or low extraction confidence must **NEVER** be used to label a cheque or transaction as "fraudulent".  
> Fraud determination is a legal and regulatory outcome reserved for human fraud risk officers following rigorous evidentiary investigation. ChequeSense strictly assigns auditable, objective operational statuses (`PROCESSED`, `VERIFIED`, `REVIEW_REQUIRED`, `INVALID`) accompanied by transparent, neutral, and verifiable `reasons_for_review` (e.g. *"Stale cheque: issued 151 days ago, exceeding 90-day validity"*, *"Low confidence on account number (0.45)"*).

---

## 2. Auditable Cheque Processing Statuses

The validation layer classifies every evaluated cheque document into one of four mutually exclusive, standardized banking statuses:

| Status | Definition | Routing Action |
| :--- | :--- | :--- |
| **`VERIFIED`** | All required fields detected with high confidence ($\ge 0.80$); all format, calendar, and numerical rules satisfied; cross-field consistency confirmed; signature stroke present. | **Straight-Through Processing (STP)** to clearing house without human intervention. |
| **`PROCESSED`** | All required fields present and syntactically valid with acceptable medium confidence ($0.60 - 0.79$). No critical discrepancies. | Eligible for automated batching with advisory logging. |
| **`REVIEW_REQUIRED`** | Extraction succeeds but triggers one or more quality gates: low confidence ($< 0.60$), stale issue date ($> 90\text{ days}$), post-dated cheque, unconfirmed signature, or word/figure mismatch. | **Routed to Human Teller Review Queue** with explicit priority ranking and detailed review reasons. |
| **`INVALID`** | Fatal business rule failure: zero or negative amount, non-numeric account number, impossible calendar date, or complete absence of required financial fields. | **Rejected** with fatal compliance errors logged. |

---

## 3. Preserved Audit Trail Specification

In compliance with banking audit standards, the validation layer strictly preserves the full lifecycle of every field extraction in [`PreservedFieldAudit`](file:///Users/karansingh/ChequeSense/src/validation/review_queue.py):

For every field and every cheque, the record preserves:
1. **`raw_model_output`**: Exact string or token stream output by the OCR or recognizer prior to any sanitation.
2. **`normalized_value`**: Sanitized, standardized value conforming to banking schema.
3. **`confidence`**: Propagated composite reliability score ($0.0$ to $1.0$).
4. **`validation_status`**: Format validation outcome (`VALID`, `INVALID`, `UNCHECKED`).
5. **`validation_errors`**: Specific syntax or schema violations (e.g. non-numeric digits).
6. **`validation_warnings`**: Business advisory flags (e.g. High-value Positive Pay trigger).
7. **`reasons_for_review`**: Explicit explanations of why human teller review is triggered.

---

## 4. Validation Rules & Quality Gates

### 4.1 Account Number Validation (`validate_account_number`)
- **Strict Leading-Zero Preservation:** Account numbers starting with zero (e.g. `001401000213`) are strictly preserved as string identifiers. They are never cast to integers, which would cause catastrophic digit loss.
- **Length Constraint:** Validated against Indian banking standards ($9 \le \text{length} \le 18$ digits).
- **Format:** Strictly numeric (`^\d{9,18}$`).

### 4.2 IFSC Code Validation (`validate_ifsc`)
- **Structure:** Exactly 11 alphanumeric characters (`^[A-Z]{4}0[A-Z0-9]{6}$`).
- **Bank Code:** First 4 characters must be alphabetic letters representing the bank (e.g. `CNRB` for Canara Bank, `SBIN` for State Bank of India).
- **Fifth Character Rule:** Strictly digit zero (`'0'`). Common OCR letter `'O'` or `'Q'` confusions are corrected or flagged.
- **Branch Suffix:** Characters 6–11 must be alphanumeric.

### 4.3 Date Validation & Validity Window (`validate_date`)
- **Calendar Logic:** Enforces `DD/MM/YYYY` format; validates day ranges per month and leap years (e.g. Feb 29 on leap years, rejects Feb 30). Preserves leading zeros (`06/05/2022`).
- **RBI Cheque Validity Rule:** Cheques are valid for strictly **3 months (90 days)** from the date of issue.
  - *Stale Cheque:* If issue date is $> 90\text{ days}$ prior to reference processing date, the cheque is marked **`REVIEW_REQUIRED`** with an explicit warning (cannot be cleared without revalidation).
  - *Post-Dated Cheque:* If issue date is in the future relative to the reference date, marked **`REVIEW_REQUIRED`** (cannot be cleared before maturity).

### 4.4 Amount & Positive Pay Validation (`validate_amount`)
- **Positive Non-Zero Rule:** Courtesy amount must be strictly $> 0.00$. Amounts $\le 0.00$ trigger immediate `INVALID` status.
- **Positive Pay Mechanism (PPM):** Cheques with amount $\ge ₹500,000.00$ are flagged with a Positive Pay advisory requiring secondary verification of issuer details.

### 4.5 Payee Name Validation (`validate_payee_name`)
- Must contain at least 2 alphabetic characters.
- Flags self-withdrawal designations (`SELF`, `CASH`) for cash counter handling.

### 4.6 Required-Field Presence Gate (`validate_required_fields`)
- Enforces non-empty presence of all 5 mandatory core fields: `acno`, `amount`, `date`, `name`, and `ifsc`.

---

## 5. Confidence Thresholding & Low-Confidence Detection

The [`ConfidenceEvaluator`](file:///Users/karansingh/ChequeSense/src/validation/confidence.py) enforces three-tier reliability gating:

```
Composite Confidence (C):
  C >= 0.80  ──►  HIGH Tier    (Eligible for STP automation if all validation rules pass)
  0.60 <= C < 0.80 ──► MEDIUM Tier  (Accepted with advisory logging)
  C < 0.60   ──►  LOW Tier     (Mandatory trigger for Human Teller Review Queue)
```

- Any core mandatory field falling into the `LOW` tier ($< 0.60$) automatically moves the entire cheque to **`REVIEW_REQUIRED`**.
- Detailed diagnostics in [`LowConfidenceDetail`](file:///Users/karansingh/ChequeSense/src/validation/confidence.py) capture detection confidence vs extraction confidence to pinpoint whether localization or OCR caused the low score.

---

## 6. Cross-Field Consistency Checks

The [`CrossFieldConsistencyChecker`](file:///Users/karansingh/ChequeSense/src/validation/consistency.py) reconciles disparate visual regions:

1. **Courtesy Amount vs Legal Amount in Words:**
   - Implements deterministic Indian and Western word-to-number parsing (supporting *Crores*, *Lakhs*, *Thousands*, *Hundreds*).
   - Flags discrepancies where numerical figures (e.g. `₹4,720.00`) disagree with written words (e.g. *"Four Thousand Seven Hundred and Twenty"*).
2. **Bank Name vs IFSC Prefix:**
   - Compares the 4-letter IFSC prefix with the printed bank title (e.g. `CNRB` requires Canara Bank; mismatch with HDFC Bank triggers review).
3. **Signature Presence & Authorization:**
   - Verifies that a valid biometric stroke pattern was detected in the authorized signatory ROI with confidence $\ge 0.50$.
   - Absence triggers **`REVIEW_REQUIRED`** with reason: *"Signature unconfirmed in designated signatory area"*.

---

## 7. Review Queue Management & Prioritization

The [`ReviewQueueManager`](file:///Users/karansingh/ChequeSense/src/validation/review_queue.py) ingests extraction results, compiles full audit trails, and ranks items for human tellers:

### Priority Ranking Algorithm:
- **`HIGH` Priority:** Fatal errors, high-value cheques ($\ge ₹50,000$), or $\ge 3$ review reasons.
- **`MEDIUM` Priority:** Single low-confidence field, stale date, or minor formatting warning.
- **`LOW` Priority:** Verified or processed cheques requiring spot-check auditing.

### Example Review Queue Item (JSON):
```json
{
  "cheque_id": "syn_syndicate_syn_0001",
  "status": "REVIEW_REQUIRED",
  "overall_confidence": 0.479,
  "review_priority": "HIGH",
  "review_required": true,
  "reasons_for_review": [
    "[ACNO] Composite confidence 0.45 falls below threshold 0.60",
    "[AMOUNT] Composite confidence 0.43 falls below threshold 0.60",
    "[DATE] Stale cheque: issued 151 days ago on 01/01/2022, exceeding 90-day validity"
  ],
  "fields": {
    "acno": {
      "raw_model_output": "Nc No] 30002010108841— 7",
      "normalized_value": "300020101088417",
      "confidence": 0.5911,
      "validation_status": "VALID",
      "reasons_for_review": [
        "Composite confidence 0.59 falls below threshold 0.60"
      ]
    }
  },
  "resolution": "PENDING"
}
```

---

## 8. Automated Test Suite Verification

Comprehensive unit tests covering all validation rules are implemented in [`tests/test_validation_layer.py`](file:///Users/karansingh/ChequeSense/tests/test_validation_layer.py):

```bash
$ PYTHONPATH=. pytest tests/
============================== 53 passed in 13.89s ==============================
```

### Complete Test Suite Summary:
- `tests/test_data_layer.py`: 15 passed (deduplication, hashes, splits, validation)
- `tests/test_detection_layer.py`: 7 passed (IoU, mAP, RoIAlign, Faster R-CNN)
- `tests/test_ocr_layer.py`: 9 passed (TSV parsing, CLAHE, IFSC/acno normalization)
- `tests/test_pipeline_layer.py`: 5 passed (end-to-end integration, CLI output)
- `tests/test_recognition_layer.py`: 8 passed (ChequeDigitCNN, glyph extraction)
- `tests/test_validation_layer.py`: 9 passed (calendar rules, leading zeros, consistency, no-fraud invariant)

*Execution stopped after Phase 6 per instructions.*
