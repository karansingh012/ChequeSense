"""Unit and integration tests for ChequeSense data layer."""

import io
import json
import numpy as np
import pytest
from PIL import Image

from src.data.dataset_loader import ChequeSample, DatasetLoader
from src.data.dataset_validator import DatasetValidator, ValidationReport
from src.data.image_preprocessor import ImagePreprocessor, PreprocessingConfig
from src.data.split_data import DataSplitter
from src.data.build_manifest import ManifestBuilder


@pytest.fixture
def sample_pil_image():
    """Generates a synthetic 200x100 RGB document-like image."""
    arr = np.ones((100, 200, 3), dtype=np.uint8) * 240
    # Add a black box simulating text
    arr[40:60, 20:180] = 0
    return Image.fromarray(arr)


@pytest.fixture
def sample_cheque_sample(sample_pil_image):
    """Creates a sample ChequeSample with bounding boxes."""
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="PNG")
    b = buf.getvalue()
    h = DatasetLoader.compute_sha256(b)
    return ChequeSample(
        sample_id="test_sample_01",
        dataset_name="synthetic",
        source_dataset="synthetic",
        source_file="test_sample_01.png",
        width=200,
        height=100,
        color_mode="RGB",
        image_hash=h,
        labels=["date", "amount"],
        annotation_type="bounding_box",
        annotations={
            "date": {"xmin": 120, "ymin": 10, "xmax": 190, "ymax": 30},
            "amount": {"xmin": 120, "ymin": 50, "xmax": 180, "ymax": 80},
        },
        _image_bytes=b,
    )


# -------------------------------------------------------------
# 1. Dataset Loading Tests
# -------------------------------------------------------------
def test_dataset_loader_idrbt():
    loader = DatasetLoader()
    samples = list(loader.load_idrbt(lazy=True))
    assert len(samples) == 112
    first = samples[0]
    assert first.source_dataset == "IDRBT"
    assert first.dataset_name == "idrbt_300"
    assert first.width > 2000
    assert first.height > 1000
    img = first.get_image()
    assert img.size == (first.width, first.height)


def test_dataset_loader_synthetic():
    loader = DatasetLoader()
    samples = list(loader.load_synthetic(split="train", lazy=True))
    assert len(samples) == 235
    first = samples[0]
    assert first.dataset_name == "synthetic"
    assert "date" in first.annotations
    assert "amount" in first.annotations
    assert first.annotations["date"]["xmax"] > first.annotations["date"]["xmin"]


def test_dataset_loader_handwritten():
    loader = DatasetLoader()
    ocr_samples = list(loader.load_handwritten_and_cheques(split="train", category="handwritten_ocr"))
    assert len(ocr_samples) == 1067
    first = ocr_samples[0]
    assert first.annotation_type == "ocr_text"
    assert "text" in first.annotations["ground_truth"]

    vqa_samples = list(loader.load_handwritten_and_cheques(split="train", category="cheque_vqa"))
    assert len(vqa_samples) == 1331


# -------------------------------------------------------------
# 2. Corrupted Image Detection Tests
# -------------------------------------------------------------
def test_corrupted_image_detection():
    # Empty bytes
    valid, err, size = DatasetValidator.validate_image_bytes(b"")
    assert not valid
    assert "Empty" in err

    # Garbage bytes
    valid, err, size = DatasetValidator.validate_image_bytes(b"NOT_A_VALID_IMAGE_HEADER_1234567890")
    assert not valid
    assert "failed" in err.lower()

    # Valid bytes
    img = Image.new("RGB", (50, 50), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    valid, err, size = DatasetValidator.validate_image_bytes(buf.getvalue())
    assert valid
    assert size == (50, 50)


def test_sample_validation_invalid_bbox(sample_cheque_sample):
    # Set invalid inverted bbox
    sample_cheque_sample.annotations["date"]["xmax"] = 100
    sample_cheque_sample.annotations["date"]["xmin"] = 150  # xmax < xmin
    is_valid, issues = DatasetValidator.validate_sample(sample_cheque_sample)
    assert not is_valid
    assert any("invalid x-range" in issue for issue in issues)


# -------------------------------------------------------------
# 3. Preprocessing Tests
# -------------------------------------------------------------
def test_preprocessing_color_conversion(sample_pil_image):
    config = PreprocessingConfig(color_mode="grayscale")
    proc = ImagePreprocessor(config)
    res = proc.process(sample_pil_image)
    assert isinstance(res.image, Image.Image)
    assert res.image.mode == "L"


def test_preprocessing_resize_aspect_ratio(sample_pil_image):
    config = PreprocessingConfig(
        target_size=(300, 300),
        maintain_aspect_ratio=True,
    )
    proc = ImagePreprocessor(config)
    res = proc.process(sample_pil_image)
    assert res.processed_size == (300, 300)
    assert res.padding[1] > 0 or res.padding[0] > 0  # Padding applied to maintain 2:1 ratio in 1:1 box


def test_preprocessing_denoise_and_contrast(sample_pil_image):
    config = PreprocessingConfig(
        denoise=True,
        denoise_method="bilateral",
        enhance_contrast=True,
        contrast_method="clahe",
    )
    proc = ImagePreprocessor(config)
    res = proc.process(sample_pil_image)
    assert res.processed_size == sample_pil_image.size


def test_preprocessing_threshold_otsu(sample_pil_image):
    config = PreprocessingConfig(
        color_mode="grayscale",
        threshold_mode="otsu",
    )
    proc = ImagePreprocessor(config)
    res = proc.process(sample_pil_image)
    # Check pixels are binarized (0 or 255)
    arr = np.array(res.image)
    unique_vals = np.unique(arr)
    assert set(unique_vals).issubset({0, 255})


def test_preprocessing_normalization(sample_pil_image):
    config = PreprocessingConfig(
        normalize=True,
        normalization_type="scale_0_1",
    )
    proc = ImagePreprocessor(config)
    res = proc.process(sample_pil_image)
    assert isinstance(res.image, np.ndarray)
    assert res.image.dtype == np.float32
    assert res.image.min() >= 0.0
    assert res.image.max() <= 1.0


def test_bbox_transformation():
    original_bbox = {"xmin": 50, "ymin": 20, "xmax": 150, "ymax": 60}
    scale = (2.0, 2.0)
    padding = (10, 20, 10, 20)
    transformed = ImagePreprocessor.transform_bounding_box(original_bbox, scale, padding)
    assert transformed["xmin"] == 50 * 2 + 10  # 110
    assert transformed["ymin"] == 20 * 2 + 20  # 60
    assert transformed["xmax"] == 150 * 2 + 10  # 310
    assert transformed["ymax"] == 60 * 2 + 20  # 140


# -------------------------------------------------------------
# 4. Duplicate Detection & Leakage Prevention Tests
# -------------------------------------------------------------
def test_duplicate_detection(sample_cheque_sample):
    dup_sample = ChequeSample(
        sample_id="test_sample_02_duplicate",
        dataset_name=sample_cheque_sample.dataset_name,
        source_dataset=sample_cheque_sample.source_dataset,
        source_file="test_sample_02.png",
        width=sample_cheque_sample.width,
        height=sample_cheque_sample.height,
        image_hash=sample_cheque_sample.image_hash,  # Same hash
        _image_bytes=sample_cheque_sample.image_bytes,
    )

    report = DatasetValidator.audit_collection([sample_cheque_sample, dup_sample])
    assert report.total_samples == 2
    assert report.duplicate_groups == 1
    assert report.duplicate_instances == 1
    assert sample_cheque_sample.image_hash in report.duplicates_by_hash


def test_leakage_detection(sample_cheque_sample):
    dup_sample = ChequeSample(
        sample_id="test_sample_02_val",
        dataset_name=sample_cheque_sample.dataset_name,
        source_dataset=sample_cheque_sample.source_dataset,
        source_file="test_sample_02.png",
        width=sample_cheque_sample.width,
        height=sample_cheque_sample.height,
        image_hash=sample_cheque_sample.image_hash,
        _image_bytes=sample_cheque_sample.image_bytes,
    )
    splits = {
        "train": [sample_cheque_sample],
        "val": [dup_sample],
    }
    leakage = DatasetValidator.detect_leakage(splits)
    assert "train_val_leakage" in leakage
    assert len(leakage["train_val_leakage"]) == 1


# -------------------------------------------------------------
# 5. Manifest Generation & Splitting Tests
# -------------------------------------------------------------
def test_data_splitter_no_leakage():
    splitter = DataSplitter(base_dir="Dataset", seed=42)
    syn_res = splitter.split_synthetic()
    train_h = set(syn_res["hash_splits"]["train_hashes"])
    val_h = set(syn_res["hash_splits"]["val_hashes"])
    test_h = set(syn_res["hash_splits"]["test_hashes"])

    assert len(train_h.intersection(val_h)) == 0
    assert len(train_h.intersection(test_h)) == 0
    assert len(val_h.intersection(test_h)) == 0


def test_manifest_builder():
    builder = ManifestBuilder()
    summary = builder.build_all(cache_images_to_raw=False)
    assert summary["total_records"] == 3219
    assert "train" in summary["splits"]
    assert "val" in summary["splits"]
    assert "test" in summary["splits"]
    assert "evaluation" in summary["splits"]

    # Verify manifest file exists and has rows
    with open(summary["manifest_json"]) as f:
        records = json.load(f)
    assert len(records) == 3219
    sample = records[0]
    for required_key in [
        "dataset_name",
        "image_path",
        "image_width",
        "image_height",
        "labels",
        "annotation_type",
        "split",
        "source_dataset",
    ]:
        assert required_key in sample
