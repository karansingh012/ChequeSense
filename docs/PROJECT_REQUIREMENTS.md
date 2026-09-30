# ChequeSense: Technical Specifications & Project Requirements
**Document Version:** 1.0.0  
**Date:** October 1, 2026  
**Role:** Lead Machine Learning Engineer  
**Project:** ChequeSense — AI-Powered Handwritten Cheque Processing & Banking Analytics System  
**Status:** Approved for Implementation Roadmap

---

## 1. Project Overview & Business Objectives

**ChequeSense** is an enterprise-grade artificial intelligence system designed to automate the ingestion, analysis, verification, and fraud detection of bank cheques compliant with the **Cheque Truncation System (CTS-2010)** standards established by the Reserve Bank of India (RBI).

### Core Capabilities
1. **Automated Document Rectification:** Automatic boundary detection, deskewing, orientation correction, and DPI normalization for scanned or mobile-captured cheque images.
2. **Key Field Localization:** Fast, robust localization of all mandatory cheque regions: Date Grid, Payee Line, Legal Amount (words), Courtesy Amount (figures), Bank & IFSC, Account Number, Signature Box, and bottom MICR strip.
3. **Hybrid OCR/HTR Engine:** Multi-stage recognition pipeline separating machine-printed text (IFSC, Bank Name, Account Number), specialized E-13B fonts (MICR band), and unconstrained cursive handwriting (Payee Name, Amount in Words).
4. **Financial Validation & Fraud Analytics:**
   - **Cross-Validation Engine:** Algorithmic verification ensuring that the amount written in words matches the numerical amount in figures.
   - **Stale/Post-Date Check:** Verification that the cheque date is within the regulatory 3-month validity window from the date of presentation.
   - **Signature Verification:** High-confidence detection of signature presence and potential forgery/blank submission.
   - **Risk Scoring:** An automated risk index evaluating overall confidence, field completeness, and tampering likelihood.

---

## 2. System Architecture & ML Pipeline

ChequeSense operates as a modular multi-stage processing pipeline:

```
[Cheque Image Input] 
         │
         ▼
┌──────────────────────────────────────────────┐
│ Stage 1: Document Preprocessing & Quality     │
│ - Grayscale/RGB normalization                │
│ - Deskewing & perspective rectification      │
│ - Binarization & contrast enhancement        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 2: Field Localization (Object Detection)│
│ Model: YOLOv8 / Faster R-CNN                 │
│ Output: Bounding Boxes (ROIs) for:           │
│   • date          • payee_name               │
│   • amount_words  • amount_figures           │
│   • ifsc          • account_number           │
│   • signature     • micr_band                │
└──────────────────────┬───────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌───────────────────────┐   ┌───────────────────────────┐
│ Stage 3A: Printed Text │   │ Stage 3B: Handwritten HTR │
│ & MICR Recognition    │   │ Model: TrOCR / CRNN       │
│ • Tesseract / EasyOCR │   │ • Payee Name Transcription│
│ • E-13B MICR Parser   │   │ • Amount in Words HTR     │
│ • IFSC & A/C extractor│   │ • Handwritten Date Digits │
└───────────┬───────────┘   └─────────────┬─────────────┘
            └─────────────┬───────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────┐
│ Stage 4: Banking Rule & Verification Engine   │
│ • Legal vs Courtesy Amount Discrepancy Check │
│ • CTS-2010 3-Month Date Window Validation    │
│ • IFSC Checksum & RBI Bank Registry Lookup   │
│ • Account Number Structure Verification      │
│ • Signature Presence Verification            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 5: Banking Analytics & Output Payload   │
│ • Fraud Risk Score (0 - 100)                 │
│ • Confidence Aggregation                     │
│ • Standardized Financial JSON Payload        │
└──────────────────────────────────────────────┘
```

---

## 3. Detailed Component Requirements

### 3.1 Stage 1: Preprocessing & Quality Assurance
- **Input Formats:** TIFF, PNG, JPEG, PDF (single-page).
- **Resolution Handling:** Rescale inputs to standardized 300 DPI (`~2365 × 1100 px`).
- **Deskewing:** Detect skew angle via Hough Line Transform or Radon Transform with automatic rotation up to ±45°.
- **Quality Flags:** Detect excessive blur (Laplacian variance < 100), overexposure/underexposure, and cropped borders.

### 3.2 Stage 2: Field Localization (Object Detection)
- **Model Architecture:** YOLOv8-Detection (Nano/Small for low-latency edge serving, Medium for maximum accuracy).
- **Target Classes (7 classes):**
  1. `date`: 8-digit date grid box in upper right.
  2. `amount_figures`: Bounded numerical box on mid-right.
  3. `amount_words`: Upper and lower legal amount lines.
  4. `payee_name`: Payee line horizontal strip.
  5. `ifsc_branch`: Bank logo, branch details, and IFSC code block.
  6. `account_number`: Dedicated printed account number band.
  7. `signature`: Lower right signature space.
  8. `micr_band`: Bottom 5/8-inch clear band containing E-13B characters.
- **Evaluation Target:** mAP@50 ≥ 0.95, mAP@50-95 ≥ 0.80.

### 3.3 Stage 3: Recognition Engines (OCR & HTR)
1. **Printed Text OCR:**
   - Targets: Bank Name, Branch Address, IFSC code, Printed Account Number.
   - Target Metric: Character Error Rate (CER) < 1.5%, Word Error Rate (WER) < 3.0%.
2. **MICR Recognition (E-13B):**
   - Targets: 6-digit cheque number, 9-digit routing transit number (3-digit city, 3-digit bank, 3-digit branch), account/SAN number, transaction code.
   - Parsing: Custom regex grammar parsing standard Indian CTS-2010 MICR delimiters.
   - Target Metric: 100% field accuracy on clean scans, > 98% on degraded scans.
3. **Handwritten Text Recognition (HTR):**
   - Model: Transformer OCR (TrOCR) fine-tuned on handwritten lines and cheque crops.
   - Targets: Amount in Words (cursive/printed English), Payee Name.
   - Target Metric: CER < 6.0%, WER < 12.0%.

### 3.4 Stage 4: Financial Verification & Business Rules
1. **Legal Amount vs. Courtesy Amount Consistency:**
   - Convert transcribed words into normalized numerical integers via algorithmic word-to-number parser (handling Indian numbering conventions: *Lakhs*, *Crores*).
   - Match parsed number against recognized digits in courtesy amount.
   - Flag: `AMOUNT_MISMATCH` if divergence exceeds 0.
2. **Date Sanity Check:**
   - Parse `DD/MM/YYYY`, `DD-MM-YYYY`, `DDMMYYYY`.
   - Calculate elapsed time against presentation date.
   - Flags: `STALE_CHEQUE` if date > 90 days in past; `POST_DATED_CHEQUE` if date > presentation date.
3. **Signature Presence & Verification:**
   - Detect ink presence, connected components, and stroke density in signature bounding box.
   - Flag: `MISSING_SIGNATURE` if stroke density is below threshold.
4. **IFSC Checksum Verification:**
   - Regex validation: `^[A-Z]{4}0[A-Z0-9]{6}$`.
   - 5th character must be '0'.

### 3.5 Stage 5: Output JSON Schema
The pipeline must return a deterministic JSON payload:
```json
{
  "cheque_id": "CHQ_20261001_001",
  "processing_timestamp": "2026-10-01T00:30:00Z",
  "document_quality": {
    "dpi": 300,
    "skew_angle_deg": -0.42,
    "blur_score": 384.2,
    "is_acceptable": true
  },
  "fields": {
    "bank_name": "Axis Bank Ltd",
    "ifsc": "UTIB0000426",
    "date": {
      "raw_text": "18012016",
      "formatted_date": "2016-01-18",
      "status": "VALID",
      "confidence": 0.98
    },
    "payee": {
      "raw_text": "T. Rameshwar",
      "confidence": 0.94
    },
    "amount_words": {
      "raw_text": "Five Lakh Twenty Thousand",
      "parsed_value": 520000,
      "confidence": 0.91
    },
    "amount_figures": {
      "raw_text": "5,20,000",
      "parsed_value": 520000,
      "confidence": 0.99
    },
    "account_number": {
      "raw_text": "911010049001545",
      "confidence": 0.97
    },
    "micr": {
      "cheque_number": "120613",
      "sort_code": "500211025",
      "san": "426160",
      "transaction_code": "31",
      "confidence": 0.99
    },
    "signature": {
      "is_present": true,
      "confidence": 0.96
    }
  },
  "validation": {
    "amount_match": true,
    "date_valid": true,
    "ifsc_valid": true,
    "discrepancies": []
  },
  "fraud_risk_assessment": {
    "risk_score": 5.2,
    "risk_level": "LOW",
    "tampering_detected": false
  }
}
```

---

## 4. Dataset Strategy & Data Partitioning

### 4.1 Remediation of Data Leakage (Synthetic Dataset)
1. **Deduplication:** Hash all 295 rows by image bytes (`SHA-256`) to isolate the **72 unique cheque templates**.
2. **Leak-Free Partitioning:**
   - **Training Set (70%):** 50 unique images.
   - **Validation Set (15%):** 11 unique images.
   - **Test Set (15%):** 11 unique images.
3. **Data Augmentation:** Apply spatial transforms (affine rotations ±5°, brightness/contrast variations ±15%, perspective distortion, synthetic crease/shadow effects) to expand the 50 training images to 1,500+ training instances.

### 4.2 Tri-Partitioning of Handwritten & Cheques Dataset
- **Cheque VQA (1,500 samples):** Clean train/test split (1,331 train / 169 test) for Key-Value Information Extraction and amount discrepancy training.
- **Handwritten Line OCR (1,200 samples):** Clean train/test split (1,067 train / 133 test) for HTR fine-tuning.
- **IDRBT Cheques (112 samples):** Retain as gold-standard benchmark validation set.

---

## 5. Non-Functional Requirements & Performance SLAs

1. **Processing Latency:**
   - Full pipeline execution < 1.5 seconds per cheque image on GPU (NVIDIA T4 / Apple Silicon Metal / CUDA).
   - CPU-only fallback latency < 4.0 seconds per cheque.
2. **Throughput:** Capable of batch processing at least 40 cheques per minute per worker.
3. **Reliability:** Graceful error handling for corrupt, unreadable, or blank documents with zero unhandled exceptions.
4. **Security & Privacy:**
   - Zero persistent storage of unencrypted PII in public cache directories.
   - Masking of bank account numbers in logging outputs (e.g., `XXXXXX01545`).
   - All dataset binaries, models, and `.env` files quarantined via `.gitignore`.

---

## 6. Project Directory Structure

```
ChequeSense/
├── Dataset/                     # (Git-ignored) Local datasets (IDRBT, synthetic, handwritten)
├── docs/                        # Project documentation and specifications
│   ├── DATASET_AUDIT.md         # Full audit report with comparison tables
│   └── PROJECT_REQUIREMENTS.md  # System requirements and technical roadmap
├── src/                         # Modular core application code
│   ├── __init__.py
│   ├── preprocessing/           # Deskewing, DPI normalization, binarization
│   ├── detection/               # YOLO/bounding box localization pipeline
│   ├── ocr/                     # Printed OCR, HTR (TrOCR), and MICR extraction
│   ├── validation/              # Business rules, amount matching, date checking
│   └── pipeline.py              # End-to-end orchestration pipeline
├── tests/                       # Unit and integration test suite
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_detection.py
│   ├── test_ocr.py
│   └── test_validation.py
├── notebooks/                   # Research, data exploration, and model validation
│   └── 01_data_exploration.ipynb
├── models/                      # (Git-ignored) Saved model checkpoints and weights
├── artifacts/                   # (Git-ignored) Audit outputs, cached test crops
├── .env                         # (Git-ignored) Environment secrets
├── .gitignore                   # Repository exclusion rules
└── README.md                    # Project landing page
```

---

## 7. Immediate Next Steps (Phase 2 Roadmap)

1. **Step 2.1 — Data Pipeline & Split Sanitizer:**
   Implement a Python data loader utility in `src/utils/data_loader.py` that loads and deduplicates the 72 unique synthetic images, eliminates cross-split leakage, and constructs standardized COCO / YOLO bounding-box annotation formats.
2. **Step 2.2 — Preprocessing Pipeline Implementation:**
   Implement deskewing, grayscale conversion, and 300 DPI normalization in `src/preprocessing/`.
3. **Step 2.3 — Rule-Based Validation Prototype:**
   Implement Indian number-in-words parser and discrepancy detection unit tests in `src/validation/` to validate against existing VQA amount pairs.
