# ChequeSense — Dataset Documentation

## Overview

ChequeSense uses three distinct datasets across its training, validation, and evaluation stages. This document describes each dataset's provenance, structure, usage, and limitations. Dataset files are excluded from the Docker image and the Git repository.

---

## 1. Synthetic Cheque Dataset

### Purpose

Training and evaluating the Faster R-CNN field detector.

### Generation

The dataset was programmatically generated using the `scripts/generate_synthetic_cheques.py` script, which uses the Python Imaging Library (Pillow) to render realistic Indian bank cheque templates.

Each image is generated with:
- Randomised cheque background (colour, border style)
- Randomised field values:
  - **date:** Random valid dates in DD/MM/YYYY format
  - **amount:** Random numeric amounts with decimal places
  - **IFSC code:** Randomly generated 11-character codes matching the regex `^[A-Z]{4}0[A-Z0-9]{6}$`
  - **account number:** Random 11–16 digit numeric strings with preserved leading zeros
  - **name:** Random payee names drawn from a name list
  - **signature:** Random scribble or signature-like strokes
- Randomised field positions within realistic layout bounds
- Randomised font sizes and styles

### Annotation Format

Ground-truth bounding boxes are stored in **COCO JSON** format with six field classes:

| Class ID | Field Name |
|---|---|
| 1 | `date` |
| 2 | `amount` |
| 3 | `ifsc` |
| 4 | `acno` (account number) |
| 5 | `sign` (signature) |
| 6 | `name` |

### Split

| Split | Images |
|---|---|
| Train | ~280 |
| Validation | ~57 |
| Test | 43 |

### Limitations

- The dataset is synthetically generated. Field layouts, fonts, and backgrounds do not cover the full diversity of real Indian bank cheques (different bank templates, thermal-paper scans, stamps, folding artefacts).
- All images are RGB JPEG. Edge cases (greyscale, very low resolution, severe skew) are not represented.
- The train/test distributions are from the same generator; results on real cheques are expected to be lower.

### File Location

```
Dataset/
└── synthetic_cheques/
    ├── train/
    │   ├── images/
    │   └── annotations.json
    ├── val/
    │   ├── images/
    │   └── annotations.json
    └── test/
        ├── images/
        └── annotations.json
```

---

## 2. MNIST Handwritten Digit Dataset

### Purpose

Training the CNN-based handwritten digit recogniser.

### Source

The standard MNIST dataset by Yann LeCun, Corinna Cortes, and Christopher Burges.  
URL: http://yann.lecun.com/exdb/mnist/

MNIST is loaded automatically via `torchvision.datasets.MNIST` and does not need to be manually downloaded.

### Structure

- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Image size:** 28×28 pixels, greyscale
- **Classes:** 10 (digits 0–9)

### Usage in ChequeSense

The CNN is trained on the full MNIST training set and evaluated on the MNIST test set to establish a baseline. It is then applied to isolated digit crops extracted from cheque field regions.

### Domain Gap

> **Important:** MNIST digits are clean, centred, well-isolated, and drawn on a uniform white background. Real handwritten cheque digit crops are written on coloured/printed cheque templates, may have varying stroke widths, pen colours, background interference, and are not centred. This domain gap is the primary reason the recogniser achieves ~43% accuracy on real cheque crops as opposed to ~99% on the MNIST test set.

### Compliance

MNIST is freely available for academic use. No redistribution restrictions apply to the dataset itself. Trained model weights derived from MNIST are stored in `models/` and are gitignored.

---

## 3. IDRBT Cheque Image Dataset

### Purpose

OCR and end-to-end pipeline evaluation on real-world cheque images.

### Source

The Institute for Development and Research in Banking Technology (IDRBT), Hyderabad, India.  
URL: https://www.idrbt.ac.in/

Access to this dataset requires institutional or research-level registration with IDRBT.

### Structure

- **Format:** JPEG images of scanned Indian bank cheques
- **Content:** Real cheque images with handwritten field values
- **Annotations:** Field-level text annotations for evaluation

### Usage in ChequeSense

The IDRBT dataset was used **exclusively for evaluation**. It was not used during training of any model component.

**Test samples used:**
- Pipeline and system evaluation: 212 images
- OCR and field extraction evaluation: 169 images
- Handwritten recognition evaluation: 182 digit crops

### Compliance

The IDRBT dataset is used under the terms of institutional research access. It is:
- Excluded from the Git repository (listed in `.gitignore`)
- Excluded from the Docker image (listed in `.dockerignore`)
- Not redistributed

---

## Dataset Split Summary

| Dataset | Train | Val | Test | Use |
|---|---|---|---|---|
| Synthetic Cheques | ~280 | ~57 | 43 | Field detector training & eval |
| MNIST | 60,000 | — | 10,000 | Recogniser training & baseline eval |
| IDRBT Cheques | — | — | 169–212 | OCR & pipeline evaluation only |

---

## Data Compliance and Privacy

- No personally identifiable information (PII) from real individuals is included in the training datasets.
- The synthetic dataset uses fictitious names, amounts, and account numbers.
- IDRBT cheque images, if they contain real personal data, are handled under the terms of the institutional research agreement and are never committed to source control.

---

## Reproducing the Synthetic Dataset

```bash
python scripts/generate_synthetic_cheques.py \
  --output_dir Dataset/synthetic_cheques \
  --train_count 280 \
  --val_count 57 \
  --test_count 43
```

> Requires Pillow (`pip install pillow`) and the font assets in `scripts/fonts/`.
