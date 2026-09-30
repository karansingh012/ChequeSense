# ChequeSense: Comprehensive Dataset Audit Report
**Date:** October 1, 2026  
**Auditor / Role:** Lead ML Engineer  
**Project:** ChequeSense (AI-powered Handwritten Cheque Processing and Banking Analytics System)  
**Status:** Audit Complete — Phase 1 Baseline

---

## 1. Executive Summary

ChequeSense aims to deliver an end-to-end intelligent banking pipeline capable of:
1. Localizing critical document regions on Indian cheques (date, payee, amount in figures, amount in words, signature, account number, IFSC, and MICR band).
2. Performing Optical Character Recognition (OCR) and Handwritten Text Recognition (HTR) on cursive and unconstrained handwriting.
3. Validating legal amount (words) against courtesy amount (numbers), account verification, and signature presence.
4. Extracting key financial entities and detecting document anomalies.

An exhaustive audit of the 1.38 GB of local data located in `Dataset/` was conducted. Three distinct datasets were audited:
- **`Dataset/IDRBT/300`**: Raw high-resolution 300 DPI TIFF scans (112 files, 825.02 MB) from the Institute for Development and Research in Banking Technology (IDRBT). Contains real Indian banking cheques with full MICR, IFSC, SAN, and signatures.
- **`Dataset/synthetic`**: Parquet-packaged high-resolution synthetic cheques (295 rows, 452 MB) with 6 ground-truth bounding box fields (`date`, `amount`, `ifsc`, `acno`, `sign`, `name`).
- **`Dataset/handwritten_and_cheques_dataset`**: Multi-modal VQA and OCR instruction dataset (2,812 rows, 106 MB) comprising handwritten text line OCR (1,200 samples), synthetic cheque key-value extraction (1,500 samples), and full entity ground truth for IDRBT cheques (112 samples).

### Critical Audit Findings
1. **Critical Synthetic Data Leakage:** The `synthetic` dataset exhibits severe data leakage. Of 295 total rows, only **72 unique images** exist. 100% of the validation split (17 unique images) and 71.4% of the test split (15 unique images) overlap with the training split.
2. **Hidden IDRBT Ground Truth Discovered:** While `Dataset/IDRBT/300` contains unannotated raw TIFF files, the exact ground-truth entity labels for all 112 cheques were discovered inside `Dataset/handwritten_and_cheques_dataset` under the query `"Extract all entities from the image."`.
3. **High PII & Regulatory Sensitivity:** `Dataset/IDRBT/300` contains real Indian bank account numbers, branch IFSC codes, CTS-2010 printer watermarks, and actual human signatures. Strict data governance is required.
4. **Git Hygiene Resolved:** No dataset files or environment secrets were committed to Git. `.gitignore` has been updated to exclude `Dataset/`, all `.parquet`, `.tif`, model checkpoints, and `.env` files.

---

## 2. Dataset Comparative Analysis Table

| Metric / Dimension | Dataset 1: IDRBT/300 | Dataset 2: Synthetic Cheques | Dataset 3: Handwritten & Cheques |
| :--- | :--- | :--- | :--- |
| **Directory / Path** | `Dataset/IDRBT/300/` | `Dataset/synthetic/data/*.parquet` | `Dataset/handwritten_and_cheques_dataset/data/*.parquet` |
| **Total Files / Records** | 112 image files | 295 records (across 3 splits) | 2,812 records (across 2 splits) |
| **Disk Size** | 825.02 MB | 452.12 MB | 106.18 MB |
| **File Format** | Raw TIFF (`.tif`, uncompressed) | Apache Parquet (embedded PNG bytes) | Apache Parquet (embedded JPEG bytes) |
| **Color Space / Channels** | RGB (3-channel, 24-bit) | RGB (3-channel, 24-bit) | Grayscale (`L`, 1-channel, 8-bit) |
| **Image Resolution (W × H)** | 2365 × 1065 to 2387 × 1100 px (Avg: 2366.6 × 1087.9 px, 300 DPI) | 2365 × 1065 to 2365 × 1100 px (Avg: 2365.0 × 1082.5 px) | **Bimodal:**<br>• Cheques: 512 × 256 to 2387 × 1100 px<br>• Line OCR: 220–3780 × 128 px (Avg: 1863 × 128 px) |
| **Existing Splits** | None (Single unsegmented directory) | Train: 235<br>Validation: 30<br>Test: 30 | Train: 2,500<br>Test: 312<br>Validation: None |
| **Corrupt / Unreadable Files**| 0 (100% integrity) | 0 (100% integrity) | 0 (100% integrity) |
| **Unique Image Hashes** | 112 unique (0 duplicates) | **72 unique** (223 duplicates across rows) | 2,812 unique (0 duplicates) |
| **Data Leakage Across Splits** | N/A (unsplit) | **Severe:** 17/17 val hashes in train; 15/21 test hashes in train | **Zero:** 0 image hash overlap between train & test |
| **Annotations / Label Type** | None in-folder (raw images) | Exact Bounding Boxes `(xmin, ymin, xmax, ymax)` | Key-Value Entity Pairs + Text Strings (VQA prompts) |
| **Available Bounding Boxes** | None in folder | 6 fields: `date`, `amount`, `ifsc`, `acno`, `sign`, `name` | None |
| **Available Text Fields** | None in folder (visual only) | Field names only (no transcribed OCR text) | • Cheque: `amt_in_words`, `amt_in_figures`, `payee_name`, `bank_name`, `cheque_date`<br>• Line OCR: Full line text transcription<br>• IDRBT: `bank`, `branch_address`, `ifsc`, `date`, `payee`, `amount_in_words`, `amount_in_figures`, `account_number`, `san`/`sappm` |
| **Banking Entities Covered** | Syndicate, Axis, Canara, SBI, ICICI | Axis, Syndicate, Canara, ICICI | Axis, Canara, ICICI, HSBC |
| **Personally Identifiable Info**| **High** (Real account numbers, names, IFSC, live signatures) | **Low–Medium** (Synthetic overlays on real branch templates) | **Low** (Synthetic payee names, standard IAM handwriting lines) |
| **Primary Recommended Use** | Real-world benchmark test set & end-to-end validation | Object Detection / Field Localization (YOLO / LayoutLM) | HTR (Handwriting OCR), VQA, Amount Discrepancy Verification |

---

## 3. Deep Dive: Dataset 1 — IDRBT Cheque Dataset (`Dataset/IDRBT/300`)

### 3.1 Overview & Provenance
The dataset originates from the **Institute for Development and Research in Banking Technology (IDRBT)**, Hyderabad, an R&D institute established by the Reserve Bank of India (RBI). The directory contains 112 TIFF files scanned at 300 DPI, which is the standardized regulatory resolution mandated by the Cheque Truncation System (CTS-2010) in India.

### 3.2 Technical Specifications
- **File Count:** 112 `.tif` files.
- **Total Storage:** 825.02 MB (Average: 7.37 MB per file).
- **Compression:** Raw uncompressed TIFF (`compression: 'raw'`).
- **Resolution Distribution:**
  - `(2365, 1079)`: 51 files (45.5%)
  - `(2365, 1100)`: 44 files (39.3%)
  - `(2387, 1093)`: 7 files (6.2%)
  - `(2365, 1065)`: 6 files (5.4%)
  - `(2372, 1093)`: 4 files (3.6%)
- **Data Integrity:** All 112 files open cleanly with PIL / OpenCV. No corrupted headers or EOF truncation.
- **SHA-256 Uniqueness:** 112 unique SHA-256 hashes. No duplicate images within the folder.

### 3.3 Visual & Banking Elements Present
Each cheque exhibits standard Indian banking layout standards (CTS-2010):
- **Bank Logos & Headers:** Syndicate Bank, Axis Bank, Canara Bank, State Bank of India, ICICI Bank.
- **Branch Information:** IFSC code, branch address, phone numbers.
- **Date Grid Box:** Formatted as `DDMMYYYY` with 8 individual digit boxes.
- **Payee Line:** Handwritten or stamped payee names.
- **Rupees in Words Line:** Handwritten amount in Indian English words (e.g., *"Rupees Eight Lakh only"*).
- **Rupees in Figures Box:** Bounded box prefixed with ₹ symbol.
- **Account Number / SAN:** Account identifier band.
- **Authorized Signatory Box:** Handwritten signatures.
- **Bottom MICR Band:** 6-digit cheque number, 9-digit MICR routing code (City-Bank-Branch), account number/SAN, and 2-digit transaction code printed in E-13B font.

---

## 4. Deep Dive: Dataset 2 — Synthetic Cheque Dataset (`Dataset/synthetic`)

### 4.1 Overview & Architecture
The synthetic dataset is packaged in Apache Parquet format across three predefined splits: `train`, `validation`, and `test`. It contains full-resolution cheque images with annotated bounding boxes targeting 6 key visual regions.

### 4.2 Split Statistics & Distribution
- **Total Records:** 295 rows (Train: 235, Validation: 30, Test: 30).
- **Total Parquet Size:** 452.12 MB.
- **Image Storage:** Embedded PNG binary bytes inside a PyArrow struct schema: `struct<bytes: binary, path: string>`.
- **Bank Distribution Across Splits:**
  - `train` (235): Canara: 78 (33.2%), Axis: 65 (27.7%), ICICI: 51 (21.7%), Syndicate: 41 (17.4%)
  - `validation` (30): Syndicate: 13 (43.3%), Axis: 6 (20.0%), ICICI: 6 (20.0%), Canara: 5 (16.7%)
  - `test` (30): Syndicate: 8 (26.7%), Axis: 8 (26.7%), Canara: 8 (26.7%), ICICI: 6 (20.0%)

### 4.3 Bounding Box Analysis
All 295 records contain valid bounding box coordinates (`xmax > xmin` and `ymax > ymin`):
- **`date`**: Average dimensions `612 × 147 px`. Located in top-right corner.
- **`amount`**: Average dimensions `640 × 152 px`. Located in mid-right courtesy amount box.
- **`ifsc`**: Average dimensions `309 × 60 px`. Located in header / branch details block.
- **`acno`**: Average dimensions `873 × 129 px`. Located in mid-left account number field.
- **`sign`**: Average dimensions `359 × 228 px`. Located in bottom-right signature area.
- **`name`**: Average dimensions `2264 × 117 px`. Full-width horizontal band spanning the payee line.

### 4.4 Critical Finding: Synthetic Data Duplication & Leakage
An exhaustive cryptographic SHA-256 analysis of the image byte streams revealed:
1. **Redundancy:** Across the 295 rows, there are only **72 unique images**. 223 rows are byte-for-byte duplicates of existing images.
2. **Identical Annotations:** Duplicate images carry 100% identical bounding boxes. For example, hash `da6bfe8adc01` appears 15 times with identical bounding boxes (`date`: 1762, 65, 2329, 186; `amount`: 1686, 377, 2298, 513).
3. **Severe Cross-Split Leakage:**
   - **Validation Leakage:** All 17 unique images in `validation` (100%) are identical to images in `train`.
   - **Test Leakage:** 15 of the 21 unique images in `test` (71.4%) are identical to images in `train`.
   - **Consequence:** Training an object detector (e.g., YOLOv8, Faster R-CNN) on this dataset as currently partitioned will lead to catastrophic evaluation bias: test performance will report artificially near-perfect mAP due to memorization of training images.

---

## 5. Deep Dive: Dataset 3 — Handwritten & Cheques Dataset (`Dataset/handwritten_and_cheques_dataset`)

### 5.1 Overview & Architecture
This dataset is packaged in two Parquet files (`train`: 2,500 records; `test`: 312 records; Total: 2,812 records). It follows an instruction-tuning / Visual Question Answering (VQA) paradigm with three schema columns: `image` (struct containing JPEG bytes), `query` (instruction string), and `answers` (ground truth string).

### 5.2 Tri-Modal Sub-Task Decomposition
A granular audit revealed that this dataset is not homogeneous; it integrates three distinct data sub-tasks:

#### Sub-Task A: Downscaled Synthetic Cheque Key-Value Extraction (1,500 samples)
- **Train:** 1,331 records | **Test:** 169 records | **Total:** 1,500 records.
- **Image Dimensions:** Grayscale JPEG, downscaled to `512 × 256 px` (aspect ratio 2:1).
- **Query Phrasing (5 variations):**
  - *"Retrieve the amount in words, amount in figures, name of the payee, date on the cheque, and the bank name from the image."* (306 train+test)
  - *"Please extract the written amount, numerical amount, recipient's name, date on the cheque, and the bank's name from the image."* (305 train+test)
  - *"Identify and extract the amount in words, amount in numbers, payee name, cheque date, and bank name from the image."* (303 train+test)
  - *"Can you pull out the amount in words, numerical amount, payee's name, date of the cheque, and the bank's name from the image?"* (292 train+test)
  - *"Extract the amount written in words, the amount in digits, the name of the payee, the date on the cheque, and the bank's name from the image."* (294 train+test)
- **Target Fields:** List of dictionaries containing `amt_in_words`, `amt_in_figures`, `payee_name`, `bank_name`, `cheque_date`.
- **Deliberate Amount Mismatch Discovery:** Multiple samples feature intentional discrepancies between amount in words and amount in figures (e.g., `amt_in_words`: *"Nine"*, `amt_in_figures`: *"40"*). This offers direct utility for developing ChequeSense's legal-versus-courtesy discrepancy detector.

#### Sub-Task B: Handwritten Line OCR / HTR (1,200 samples)
- **Train:** 1,067 records | **Test:** 133 records | **Total:** 1,200 records.
- **Image Dimensions:** Single-line horizontal crops with fixed height of `128 px` and variable widths ranging from `220 px` to `3,780 px` (Avg: `1,863 × 128 px`).
- **Query Phrasing (5 variations):**
  - *"Could you decipher the writing in the image?"*
  - *"What is the content of the text in the image?"*
  - *"What does the image's text read?"*
  - *"What words are displayed in the image?"*
  - *"Can you tell me what the text in the image says?"*
- **Ground Truth:** Direct plain text transcription. Character lengths range from 4 to 74 characters (Avg: 43.3 characters, 8.5 words per line). Sourced from academic handwriting corpora (IAM Handwriting benchmark).

#### Sub-Task C: IDRBT Full-Cheque Entity Ground Truth (112 samples)
- **Train:** 102 records | **Test:** 10 records | **Total:** 112 records.
- **Image Dimensions:** Full resolution scans (`2365 × 1079` to `2387 × 1093 px`).
- **Query:** Exactly `"Extract all entities from the image."`
- **Answers:** Complete structured JSON dictionary:
  ```json
  {
    "bank": "Axis Bank Ltd",
    "branch_address": "Mehdipatnam, Hyderabad [AP], Hyderabad, 500028",
    "ifsc": "UTIB0000426",
    "date": "18/01/2016",
    "payee": "T. Rameshwar",
    "amount_in_words": "Five Lakh Twenty Thousand",
    "amount_in_figures": "5,20,000",
    "account_number": "911010049001545",
    "san": "426160"
  }
  ```
- **Significance:** This provides the missing ground-truth key-value labels for the 112 unannotated TIFF images in `Dataset/IDRBT/300`.

### 5.3 Split Quality & Duplication
- **Total Records:** 2,812 rows.
- **Duplicate Images:** 0 duplicates. All 2,500 train images and 312 test images have unique SHA-256 hashes.
- **Train/Test Leakage:** 0 hashes overlap between train and test. The split is mathematically clean.

---

## 6. Privacy, Security & Git Safety Audit

### 6.1 Personally Identifiable Information (PII) & Banking Compliance
1. **`Dataset/IDRBT/300`:**
   - **Classification:** SENSITIVE / REGULATORY COMPLIANCE REQUIRED.
   - **Identified PII:** Scanned real bank cheques featuring authentic account numbers (`account_number`, `san`), actual branch IFSC codes, CTS-2010 printer vendor watermarks (e.g., *Manipal Technologies Ltd.*), payee names, and live human signatures.
   - **Compliance Obligation:** These files must remain strictly restricted to local ML training. They must never be logged in raw forms to external monitoring services, included in client-side bundles, or committed to public version control.
2. **`Dataset/synthetic`:**
   - **Classification:** LOW SENSITIVITY.
   - Overlays synthetic dummy names and amounts onto synthetic or anonymized bank backgrounds.
3. **`Dataset/handwritten_and_cheques_dataset`:**
   - **Classification:** PUBLIC ACADEMIC / SYNTHETIC.
   - Payee names are generated synthetic identities (*"Searlas Grenier"*, *"Ethel M. Bryson"*), and handwritten lines are standard academic transcription benchmarks.

### 6.2 Git Tracking & Secrets Status
- **Current Git Tracking:** Verified using `git ls-files ChequeSense`. No dataset files or binaries are currently tracked in the repository.
- **Git Root Observation:** The user's home directory (`/Users/karansingh`) contains a global Git root. To prevent accidental commits or staging of workspace artifacts, `.gitignore` inside `ChequeSense/` was updated and verified.
- **`.gitignore` Rules:** Excludes `Dataset/`, `*.parquet`, `*.tif`, `models/*`, `artifacts/*`, `__pycache__/`, `.env`, and OS artifacts. Verified via `git check-ignore -v`.
- **Environment Secrets:** `.env` contains RocketRide platform connection keys. Verified that `.env` is ignored by `.gitignore` and no secrets have been exposed or committed.

---

## 7. Recommended Purpose & Model Architecture Alignment

| Dataset | Recommended Component in ChequeSense Pipeline | Model Architecture Recommendation |
| :--- | :--- | :--- |
| **`Dataset/synthetic`** | **Cheque Field Localization & Bounding Box Detection** | YOLOv8-Detection / Faster R-CNN / RT-DETR trained on deduplicated 72 images with aggressive data augmentation (random rotation, jitter, perspective warping) |
| **`Dataset/handwritten_and_cheques_dataset` (Line OCR)** | **Handwritten Text Recognition (HTR)** | TrOCR (Transformer OCR), CRNN (CNN + BiLSTM + CTC), or PaddleOCR fine-tuned for cursive writing |
| **`Dataset/handwritten_and_cheques_dataset` (Cheque VQA)** | **End-to-End Key-Value Extraction & Discrepancy Detection** | Donut / LayoutLMv3 / Nougat for structured extraction; Rule-based Python NLP engine for Amount-in-Words vs Amount-in-Figures validation |
| **`Dataset/IDRBT/300` + Linked Entity Ground Truth** | **End-to-End System Evaluation & CTS-2010 Real-World Benchmark** | Held-out validation and benchmark testing for the integrated end-to-end ChequeSense processing pipeline |

---

## 8. Summary of Identified Risks & Mitigation Plan

1. **Risk 1: Synthetic Dataset Overfitting & Leakage**
   - *Issue:* 223 duplicate rows and 100% validation leakage.
   - *Mitigation:* Perform hash-based grouping. Re-partition the 72 unique images into a leak-free 52 train / 10 val / 10 test split before any detector training.
2. **Risk 2: Low-Resolution Cheque VQA Images (512 × 256)**
   - *Issue:* 1,500 cheque images in `handwritten_and_cheques_dataset` are heavily downscaled, causing severe blur in small text (MICR and IFSC).
   - *Mitigation:* Use these images exclusively for text extraction of large fields (payee, amounts). Use full-resolution IDRBT scans for MICR and small-font field extraction.
3. **Risk 3: Unannotated MICR Band in Synthetic Data**
   - *Issue:* The synthetic dataset annotates 6 fields but omits the bottom MICR band (`cheque_no`, `sort_code`, `account_id`, `trans_code`).
   - *Mitigation:* Implement a deterministic geometric heuristic crop for the bottom 15–20% of the cheque or use regex-based E-13B OCR templates for MICR extraction.
4. **Risk 4: PII in Real Cheque Scans**
   - *Issue:* Scanned real cheques in IDRBT contain sensitive banking information.
   - *Mitigation:* Enforce strict local containment. Add synthetic masking / blurring utility for signatures and account numbers prior to any UI display.
