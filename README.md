# ChequeSense

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()
[![Code Architecture](https://img.shields.io/badge/architecture-modular-green.svg)]()

> **AI-Powered Handwritten Cheque Processing and Banking Analytics System**

ChequeSense is an end-to-end intelligent banking document pipeline compliant with **CTS-2010 (Cheque Truncation System)** standards. It localizes key cheque fields, extracts printed and cursive handwritten text, parses magnetic E-13B MICR bands, and cross-validates legal amount (words) against courtesy amount (figures) for automated banking verification and fraud detection.

---

## Repository Structure

```text
ChequeSense/
├── Dataset/                     # (Git-ignored) Raw benchmark & synthetic datasets
│   ├── synthetic/               # 295 synthetic cheques with 6 bounding box fields
│   ├── IDRBT/300/               # 112 authentic 300 DPI Indian banking cheque scans
│   └── handwritten_and_cheques_dataset/ # 2,812 samples (OCR lines & Cheque VQA)
├── data/                        # Generated data layer
│   ├── raw/                     # (Git-ignored) On-demand extracted image cache
│   ├── processed/               # (Git-ignored) Preprocessed image derivatives
│   ├── annotations/             # COCO-format detection annotations
│   ├── splits/                  # Leak-free split mappings (JSON)
│   └── manifests/               # Machine-readable dataset manifests (JSON & CSV)
├── docs/                        # Engineering reports & technical specifications
│   ├── DATASET_AUDIT.md         # Full audit report & comparative dataset analysis
│   └── PROJECT_REQUIREMENTS.md  # System requirements, pipeline specs & SLAs
├── src/                         # Modular application source code
│   └── data/                    # Data preparation layer
│       ├── dataset_loader.py    # In-memory & lazy dataset loaders
│       ├── dataset_validator.py # Integrity, corruption & leakage validation
│       ├── image_preprocessor.py# Configurable document preprocessing engine
│       ├── split_data.py        # Hash-based deduplication & leak-free splitter
│       └── build_manifest.py    # Master manifest & COCO annotation builder
├── tests/                       # Automated test suite
│   └── test_data_layer.py       # 15 unit & integration tests
├── notebooks/                   # Exploration & visualization
│   └── 01_dataset_visualization.ipynb
├── models/                      # (Git-ignored) Model weights & checkpoints
└── artifacts/                   # (Git-ignored) Intermediate artifacts & logs
```

---

## Data Preparation & Leakage Prevention

During the dataset audit, a critical flaw was identified in the raw synthetic dataset: 223 duplicate rows out of 295 total records, where 100% of validation images and 71.4% of test images overlapped with the training set. 

ChequeSense resolves this through **cryptographic hash grouping**:
1. **Deduplication by Image SHA-256:** The 72 unique cheque templates are partitioned as atomic groups, ensuring identical images never cross split boundaries.
2. **Dedicated Evaluation Benchmark:** Real Indian cheques from **IDRBT/300** and their corresponding entity VQA ground truth (112 samples) are strictly isolated into the `evaluation` split (0% training exposure).

---

## Reproducing the Data Preparation Layer

### 1. Partition Data (Leak-Free Splitter)
Partition the datasets deterministically using cryptographic image hash grouping:
```bash
python3 -m src.data.split_data --base_dir Dataset --output_dir data/splits --seed 42
```
This produces:
- `data/splits/synthetic_splits.json` (52 train, 10 val, 10 test unique image groups)
- `data/splits/handwritten_splits.json` (HTR: 960 train, 107 val, 133 test; VQA: 1,198 train, 133 val, 169 test; IDRBT: 112 evaluation)
- `data/splits/idrbt_splits.json` (112 evaluation)
- `data/splits/combined_splits_summary.json`

### 2. Generate Master Manifests & COCO Annotations
Compile unified machine-readable JSON/CSV manifests and COCO-format detection files:
```bash
python3 -m src.data.build_manifest --base_dir Dataset
```
Outputs generated in `data/`:
- `data/manifests/dataset_manifest.json` (3,219 total records)
- `data/manifests/dataset_manifest.csv`
- Split-specific manifests: `train_manifest.json`, `val_manifest.json`, `test_manifest.json`, `evaluation_manifest.json`
- COCO detection annotations: `synthetic_coco_train.json`, `synthetic_coco_val.json`, `synthetic_coco_test.json`

*(Optional)* To extract images to `data/raw/` for standard disk loaders, append `--cache_images`:
```bash
python3 -m src.data.build_manifest --base_dir Dataset --cache_images
```

---

## Image Preprocessing Pipeline

Preprocessing in ChequeSense is **never applied blindly** to raw data. The original images are strictly preserved, and transformations are executed on-demand through explicit configuration (`PreprocessingConfig`):

```python
from PIL import Image
from src.data.image_preprocessor import ImagePreprocessor, PreprocessingConfig

# Configure tailored preprocessing for document OCR / detection
config = PreprocessingConfig(
    color_mode="grayscale",          # 'rgb', 'grayscale', 'unchanged'
    target_size=(2365, 1100),        # Scale to standard 300 DPI dimensions
    maintain_aspect_ratio=True,      # Preserves geometry with white margin padding
    denoise=True,                    # Bilateral filtering preserves crisp text edges
    denoise_method="bilateral",
    enhance_contrast=True,           # Contrast Limited Adaptive Histogram Equalization
    contrast_method="clahe",
    clahe_clip_limit=2.0,
    threshold_mode="otsu",           # Optional Otsu binarization for HTR
    deskew=True,                     # Automatic skew angle detection and correction
    normalize=False                  # Scale to [0.0, 1.0] float array if True
)

preprocessor = ImagePreprocessor(config)
result = preprocessor.process(image_input)

# Access outputs
processed_img = result.image         # Transformed PIL Image or numpy array
scale_factors = result.scale_factors # (scale_x, scale_y) for bounding box remapping
padding = result.padding             # (pad_left, pad_top, pad_right, pad_bottom)
skew_angle = result.skew_angle       # Estimated skew in degrees
```

### Bounding Box Transformation
When resizing or padding images with bounding box annotations, remap bounding boxes with:
```python
adjusted_bbox = ImagePreprocessor.transform_bounding_box(
    original_bbox, result.scale_factors, result.padding
)
```

---

## Cheque Field Detection (Phase 2)

ChequeSense implements a specialized **Faster R-CNN with MobileNetV3-Large FPN** architecture for localizing the 6 core cheque fields without hallucinating unannotated regions:
- `date`: 8-digit date grid box (`DDMMYYYY`)
- `amount`: Numerical courtesy amount box prefixed with ₹
- `ifsc`: Branch IFSC and bank routing block
- `acno`: Printed account number band
- `sign`: Authorized signatory area
- `name`: Payee recipient horizontal line

### 1. Training the Detector
Train the field detector on the leak-free deduplicated synthetic training split:
```bash
python3 -m src.detection.train --epochs 10 --batch_size 4 --lr 0.0005
```
Checkpoints are automatically saved to `models/field_detector/best_model.pt` when validation mAP@50 improves.

### 2. Evaluating Model Performance
Evaluate precision, recall, mAP@50, and mAP@[50:95] on the held-out test split:
```bash
python3 -m src.detection.evaluate --manifest_path data/manifests/test_manifest.json
```
**Held-Out Test Set Metrics:**
- **mAP@50:** **1.0000** (100.0%)
- **mAP@50-95:** **0.7819**
- **Mean Precision@50:** **0.9826**
- **Mean Recall@50:** **1.0000**
- **Mean F1 Score@50:** **0.9908**

### 3. Running Field Detection Inference
Run inference on any cheque image to extract original-resolution bounding boxes:
```python
from src.detection.inference import FieldDetector

detector = FieldDetector(model_path="models/field_detector/best_model.pt")
prediction = detector.predict("Dataset/IDRBT/300/Cheque 083654.tif")

print("Detected cheque fields:")
for field_name, box in prediction.fields.items():
    print(f"  {field_name}: conf={box.confidence:.2f}, coords=({box.xmin}, {box.ymin}, {box.xmax}, {box.ymax})")
```

### 4. Visualizing Predicted Bounding Boxes
Render and save color-coded bounding boxes on sample cheques:
```bash
python3 -m src.detection.visualize --image_path "Dataset/IDRBT/300/Cheque 083654.tif" --output_path "artifacts/field_detection/sample_prediction.png"
```

---

## Verification & Automated Tests

Execute the complete test suite (22 unit & integration tests covering data loading, corruption detection, preprocessing, splitting, manifest compilation, IoU/mAP metrics, and detector inference):

```bash
python3 -m pytest tests/ -v
```

---

## Dataset Exploration

To interactively visualize sample cheques, bounding boxes, handwritten text lines, and image preprocessing steps, launch Jupyter and open:
```bash
jupyter notebook notebooks/01_dataset_visualization.ipynb
```

