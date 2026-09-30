"""Unit tests for the ChequeSense character/digit recognition module."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.recognition.dataset import (
    CLASS_TO_IDX,
    DIGIT_CLASSES,
    IDX_TO_CLASS,
    NUM_CLASSES,
    ChequeDigitDataset,
)
from src.recognition.evaluate import DigitClassMetric, RecognitionEvaluator, RecognitionResult
from src.recognition.model import ChequeDigitCNN
from src.recognition.predict import CharacterRecognizer, FieldRecognizer
from src.recognition.preprocessing import GlyphPreprocessor
from src.recognition.visualize import draw_confusion_matrix_image, draw_prediction_grid


def test_digit_classes_schema():
    assert NUM_CLASSES == 10
    assert len(DIGIT_CLASSES) == 10
    assert DIGIT_CLASSES == [str(i) for i in range(10)]
    for i in range(10):
        s = str(i)
        assert CLASS_TO_IDX[s] == i
        assert IDX_TO_CLASS[i] == s


def test_glyph_preprocessor_normalization():
    prep = GlyphPreprocessor(target_size=(32, 32), pad_ratio=0.125)

    # 1. White canvas with dark digit stroke
    img = Image.new("L", (50, 80), color=255)
    # Draw stroke
    arr = np.array(img)
    arr[20:60, 20:30] = 0

    norm = prep.normalize_glyph(arr)
    assert isinstance(norm, np.ndarray)
    assert norm.shape == (32, 32)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0

    # Foreground stroke should be white (> 0.5) centered
    assert np.mean(norm) > 0.0


def test_glyph_preprocessor_field_segmentation():
    prep = GlyphPreprocessor(target_size=(32, 32))

    # Field image with 3 distinct dark vertical bars on light background
    canvas = np.ones((50, 150), dtype=np.uint8) * 255
    canvas[10:40, 20:28] = 0
    canvas[10:40, 60:68] = 0
    canvas[10:40, 100:108] = 0

    glyphs = prep.segment_field_into_glyphs(canvas)
    assert len(glyphs) == 3
    for norm_glyph, box in glyphs:
        assert norm_glyph.shape == (32, 32)
        assert len(box) == 4
        x, y, w, h = box
        assert w > 0 and h > 0


def test_cheque_digit_cnn_forward():
    model = ChequeDigitCNN(num_classes=10)
    model.eval()

    batch = torch.randn(4, 1, 32, 32)
    logits = model(batch)
    assert logits.shape == (4, 10)

    probs = model.predict_proba(batch)
    assert probs.shape == (4, 10)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)

    preds, confs = model.predict_glyph(batch)
    assert preds.shape == (4,)
    assert confs.shape == (4,)
    assert (preds >= 0).all() and (preds < 10).all()
    assert (confs >= 0.0).all() and (confs <= 1.0).all()


def test_recognition_evaluator_synthetic():
    evaluator = RecognitionEvaluator(low_confidence_threshold=0.70)

    # 10 samples: 8 correct, 2 incorrect (one 1->4, one 7->1)
    y_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_pred = [0, 4, 2, 3, 4, 5, 6, 1, 8, 9]
    confs = [0.95, 0.55, 0.90, 0.88, 0.92, 0.85, 0.89, 0.45, 0.94, 0.91]

    res = evaluator.evaluate(y_true, y_pred, confs)
    assert res.total_samples == 10
    assert np.isclose(res.accuracy, 0.80)
    assert len(res.confusing_pairs) == 2
    assert len(res.low_confidence_cases) == 2  # 0.55 and 0.45 < 0.70
    assert len(res.incorrect_cases) == 2

    # Confusion matrix diagonal checks
    cm = np.array(res.confusion_matrix)
    assert cm[0, 0] == 1
    assert cm[1, 4] == 1
    assert cm[7, 1] == 1


def test_character_and_field_recognizer():
    cr = CharacterRecognizer(model_path="models/recognizer/best_model.pt")
    dummy_glyph = np.zeros((32, 32), dtype=np.uint8)
    single_res = cr.predict_glyph(dummy_glyph)
    assert single_res.char in DIGIT_CLASSES
    assert 0.0 <= single_res.confidence <= 1.0
    assert len(single_res.probabilities) == 10

    fr = FieldRecognizer(model_path="models/recognizer/best_model.pt")
    field_crop = np.ones((40, 100), dtype=np.uint8) * 255
    field_crop[10:30, 20:25] = 0
    field_crop[10:30, 60:65] = 0
    field_res = fr.recognize_field(field_crop)
    assert len(field_res.characters) == 2
    assert len(field_res.predicted_text) == 2
    assert 0.0 <= field_res.aggregate_confidence <= 1.0


def test_cheque_digit_dataset():
    dataset = ChequeDigitDataset(manifest_path="data/manifests/val_manifest.json", augment=False)
    assert len(dataset) > 0

    tensor, label, meta = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 32, 32)
    assert 0 <= label <= 9
    assert "char" in meta
    assert "source_sample_id" in meta
    assert "field_type" in meta


def test_draw_confusion_matrix_and_grid():
    cm = [[5, 1], [0, 4]]
    labels = ["0", "1"]
    cm_img = draw_confusion_matrix_image(cm, labels=labels, output_path=None)
    assert isinstance(cm_img, Image.Image)

    samples = [
        {
            "glyph_array": np.zeros((32, 32), dtype=np.float32),
            "true_digit": "0",
            "pred_digit": "0",
            "confidence": 0.95,
            "is_correct": True,
        }
    ]
    grid_img = draw_prediction_grid(samples, output_path="artifacts/recognition/visualizations/test_grid.png", max_samples=4)
    assert isinstance(grid_img, Image.Image)
