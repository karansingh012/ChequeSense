# ChequeSense — ML Pipeline

## Overview

The ChequeSense ML pipeline integrates three independent model components—field detection, handwritten digit recognition, and OCR—into a single, modular inference workflow. This document describes each component's architecture, training procedure, inference configuration, and confidence propagation logic.

---

## Pipeline Workflow

![Workflow Diagram](workflow_diagram.jpg)

```
Input Image
    │
    ├─ 1. Preprocessing  ──→ Resized + enhanced image
    │
    ├─ 2. Field Detection ─→ Bounding boxes + detection confidence per field
    │
    ├─ 3. Region Cropping ─→ 6 cropped image patches
    │
    ├─ 4a. Tesseract OCR ──→ Raw text + OCR confidence (all fields)
    │  4b. CNN Recogniser ─→ Digit class + softmax prob (amount, date)
    │
    ├─ 5. Reconstruction ──→ Composite confidence + normalised values
    │
    └─ 6. Validation ──────→ Status (PROCESSED / VERIFIED / REVIEW_REQUIRED / INVALID)
```

---

## Stage 1: Image Preprocessing (`src/pipeline/preprocess.py`)

### Operations

1. **Load:** Accept file path, PIL Image, NumPy array, or raw bytes.
2. **Format validation:** Ensure the file is a valid image (PIL verify).
3. **Colour normalisation:** Convert to RGB if necessary.
4. **Resize:** Scale to fit within 1333×800 whilst preserving aspect ratio (same as Faster R-CNN default transform).
5. **CLAHE:** Apply Contrast Limited Adaptive Histogram Equalisation to improve contrast in low-quality scans.
6. **ID inference:** If no `cheque_id` is provided, infer one from the filename stem.

### Output

- `PIL.Image` ready for detection
- Inferred cheque identifier string

---

## Stage 2: Field Detection (`src/pipeline/detect_fields.py`)

### Model

**Architecture:** Faster R-CNN with ResNet-50 FPN backbone.

The model is loaded from `torchvision.models.detection.fasterrcnn_resnet50_fpn`, with the final classification head replaced to output `num_classes = 7` (background + 6 field classes).

### Training

| Parameter | Value |
|---|---|
| Base weights | COCO pre-trained (ImageNet backbone) |
| Input resize | Max 1333×800 |
| Augmentations | Horizontal flip (p=0.5), colour jitter, ±10° rotation |
| Optimiser | SGD, lr=0.005, momentum=0.9, weight decay=5e-4 |
| LR Scheduler | StepLR, step=15, γ=0.1 |
| Epochs | 20 |
| Batch size | 4 |
| Training set | ~280 synthetic cheque images |

### Inference

```python
detector = PipelineFieldDetector(config)
field_crops, signature_status, raw_detections = detector.detect_and_crop(pil_image)
```

- Predictions are filtered by a configurable confidence threshold (default: 0.5 at detection stage).
- For each field class, the highest-confidence box is selected.
- Each box is expanded by a small padding margin before cropping.
- If a field is not detected, it is recorded with `detection_confidence = 0.0`.

### Fields Detected

`date`, `amount`, `ifsc`, `acno`, `sign`, `name`

Signature (`sign`) is detected for presence only; no value is extracted.

### Checkpoint

```
models/field_detector/best_model.pt
```

---

## Stage 3: OCR (`src/pipeline/ocr.py`)

### Engine

**Tesseract 5.x** via `pytesseract`.

The Tesseract binary path is configurable via the `TESSERACT_CMD` environment variable.

### Field-Specific Configuration

| Field | PSM | Whitelist |
|---|---|---|
| `name` | 7 (single text line) | None |
| `date` | 7 (single text line) | 0-9/- |
| `amount` | 8 (single word) | 0-9., |
| `ifsc` | 7 (single text line) | A-Z0-9 |
| `acno` | 8 (single word) | 0-9 |

### Confidence Extraction

Tesseract's `image_to_data` output includes a per-word confidence value (0–100). The OCR extraction confidence is the mean word confidence divided by 100, clipped to [0, 1].

If Tesseract returns no text for a field, the OCR confidence is 0.0.

### Output (`PipelineOCROutput`)

```python
@dataclass
class PipelineOCROutput:
    raw_text: str          # Exact Tesseract output, no post-processing
    confidence: float      # Mean Tesseract word confidence in [0, 1]
    field_name: str
```

---

## Stage 4: Handwritten Digit Recognition (`src/pipeline/recognize.py`)

### Model

**Architecture:** Four-layer CNN:

```
Conv2d(1, 32, 3) → ReLU → MaxPool(2)
Conv2d(32, 64, 3) → ReLU → MaxPool(2)
Flatten
Linear(64×5×5, 128) → ReLU → Dropout(0.5)
Linear(128, 10)     → Softmax
```

Trained on MNIST (60,000 samples) for 10-class digit classification.

### Training

| Parameter | Value |
|---|---|
| Dataset | MNIST |
| Optimiser | Adam, lr=1e-3 |
| Epochs | 10 |
| Batch size | 64 |
| MNIST test accuracy | ~99% |

### Inference

The recogniser is currently applied to `amount` and `date` field crops. The crop is converted to greyscale, resized to 28×28, normalised, and passed through the CNN.

Output is the predicted digit class (0–9) and the softmax probability for that class.

> **Important:** The recogniser classifies a single isolated digit. It is applied to each detected digit crop in sequence. Sequence assembly (combining individual digit predictions into a multi-digit number) is performed by the reconstruction stage and is currently basic (concatenation of top-1 predictions). This is a known limitation.

### Checkpoint

```
models/recognizer/best_model.pt
```

---

## Stage 5: Field Reconstruction and Confidence Propagation (`src/pipeline/reconstruct.py`)

### Composite Confidence

For each field, composite confidence is computed as:

```python
composite_confidence = detection_confidence * extraction_confidence
```

Where `extraction_confidence` is the higher of the OCR confidence and the recognition confidence (if both are available for that field).

### Confidence Tiers

| Tier | Threshold |
|---|---|
| HIGH | ≥ 0.85 |
| MEDIUM | 0.60 – 0.85 |
| LOW | < 0.60 |

### Value Normalisation

| Field | Normalisation |
|---|---|
| `date` | Parse and reformat to DD/MM/YYYY |
| `amount` | Strip non-numeric characters; preserve decimals |
| `ifsc` | Uppercase; strip whitespace |
| `acno` | Preserve leading zeros; strip non-digits |
| `name` | Strip excess whitespace; title-case |

### Output (`ChequePipelineResult`)

A Pydantic model with:
- `cheque_id`
- Per-field `FieldOutput`: `value`, `confidence`, `raw_value`, `detection_confidence`, `extraction_confidence`, `confidence_tier`, `bounding_box`
- `status` (initial)
- `diagnostics`: per-stage latencies, image dimensions

---

## Confidence Threshold Configuration

The global confidence threshold (default: `0.60`) is configurable in `PipelineConfig` and can be overridden via environment variable `CONFIDENCE_THRESHOLD`.

Fields below this threshold are flagged in the validation stage as `REVIEW_REQUIRED`.

---

## Model Versioning

Each model version is recorded in every `ProcessingRun` database record:

```python
detector_model_version    # e.g., "faster_rcnn_resnet50_fpn_v1"
recognizer_model_version  # e.g., "mnist_cnn_v1"
ocr_engine_version        # e.g., "tesseract-5.3.3"
```

This allows analytics queries to filter performance by model version.

---

## Training Scripts

| Script | Purpose |
|---|---|
| `src/detection/train.py` | Train Faster R-CNN on synthetic cheque dataset |
| `src/recognition/train.py` | Train CNN on MNIST |

Example invocations:

```bash
# Field detector training
python src/detection/train.py \
  --data_dir Dataset/synthetic_cheques \
  --output_dir models/field_detector \
  --epochs 20

# Digit recogniser training
python src/recognition/train.py \
  --output_dir models/recognizer \
  --epochs 10
```

---

## CLI Usage

```bash
python -m src.pipeline.pipeline \
  --image path/to/cheque.jpg \
  --output result.json \
  --detector_model models/field_detector/best_model.pt \
  --recognizer_model models/recognizer/best_model.pt \
  --device cpu \
  --log_level INFO
```

Output is a JSON file conforming to the `ChequePipelineResult` schema.
