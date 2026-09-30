"""Visualization and error analysis utilities for ChequeSense character/digit recognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from src.recognition.dataset import DIGIT_CLASSES


def draw_confusion_matrix_image(
    confusion_matrix: List[List[int]],
    labels: Optional[List[str]] = None,
    output_path: Optional[str] = "artifacts/recognition/visualizations/confusion_matrix.png",
    title: str = "Cheque Digit Recognition Confusion Matrix",
) -> Image.Image:
    """Renders an annotated confusion matrix heatmap using PIL."""
    labels = labels or DIGIT_CLASSES
    cm = np.array(confusion_matrix)
    n_classes = len(labels)

    cell_size = 56
    margin_left = 90
    margin_top = 80
    margin_bottom = 60
    margin_right = 90

    width = margin_left + n_classes * cell_size + margin_right
    height = margin_top + n_classes * cell_size + margin_bottom

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((width // 2 - 170, 20), title, fill=(20, 20, 30))

    max_val = max(int(cm.max()), 1)

    # Draw cells
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            norm_val = val / max_val

            # Soft blue/cyan gradient: white -> (40, 110, 200)
            r = int(255 - norm_val * (255 - 40))
            g = int(255 - norm_val * (255 - 110))
            b = int(255 - norm_val * (255 - 210))

            x0 = margin_left + j * cell_size
            y0 = margin_top + i * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b), outline=(210, 215, 225), width=1)

            text_color = (255, 255, 255) if norm_val > 0.45 else (20, 25, 35)
            val_str = str(val)
            draw.text((x0 + cell_size // 2 - 5, y0 + cell_size // 2 - 8), val_str, fill=text_color)

    # Draw axes labels
    for j in range(n_classes):
        x = margin_left + j * cell_size + cell_size // 2 - 4
        y = margin_top - 25
        draw.text((x, y), labels[j], fill=(40, 50, 70))

    for i in range(n_classes):
        x = margin_left - 30
        y = margin_top + i * cell_size + cell_size // 2 - 8
        draw.text((x, y), labels[i], fill=(40, 50, 70))

    draw.text((width // 2 - 50, height - 35), "Predicted Digit", fill=(30, 40, 60))
    draw.text((15, height // 2 - 10), "True", fill=(30, 40, 60))

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_p)
        print(f"Saved confusion matrix plot to: {out_p}")

    return img


def draw_prediction_grid(
    samples: List[Dict[str, Any]],
    output_path: str,
    title: str = "Predictions Grid",
    max_samples: int = 16,
) -> Image.Image:
    """Renders a grid of character predictions with ground-truth and confidence using PIL."""
    n = min(len(samples), max_samples)
    if n == 0:
        return Image.new("RGB", (100, 100), color="white")

    cols = 4
    rows = int(np.ceil(n / cols))

    card_w = 160
    card_h = 160
    pad = 16
    title_h = 45

    width = pad + cols * (card_w + pad)
    height = title_h + rows * (card_h + pad) + pad

    canvas = Image.new("RGB", (width, height), color=(245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad + 10, 15), title, fill=(30, 40, 55))

    for idx in range(n):
        r = idx // cols
        c = idx % cols

        x0 = pad + c * (card_w + pad)
        y0 = title_h + r * (card_h + pad)
        x1 = x0 + card_w
        y1 = y0 + card_h

        item = samples[idx]
        true_ch = item.get("true_digit", "")
        pred_ch = item.get("pred_digit", "")
        conf = item.get("confidence", 0.0)
        is_corr = item.get("is_correct", true_ch == pred_ch)

        # Card background
        border_col = (46, 160, 67) if is_corr else (220, 53, 69)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=border_col, width=2)

        # Glyph image
        img_arr = item.get("glyph_array")
        if img_arr is not None:
            if img_arr.dtype in (np.float32, np.float64):
                uint_img = (img_arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                uint_img = img_arr
            glyph_pil = Image.fromarray(uint_img).convert("RGB")
            # Invert for white paper background
            glyph_pil = Image.fromarray(255 - np.array(glyph_pil))
            glyph_resized = glyph_pil.resize((72, 72), Image.Resampling.NEAREST)

            gx = x0 + (card_w - 72) // 2
            gy = y0 + 12
            canvas.paste(glyph_resized, (gx, gy))

        # Status text
        lbl_pred = f"Pred: '{pred_ch}' ({conf:.2f})"
        lbl_true = f"True: '{true_ch}'"
        draw.text((x0 + 12, y0 + 95), lbl_pred, fill=border_col)
        draw.text((x0 + 12, y0 + 122), lbl_true, fill=(80, 90, 105))

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_p)
    print(f"Saved prediction grid to: {out_p}")
    return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize recognition metrics and error analysis.")
    parser.add_argument("--metrics_file", default="artifacts/recognition/test_metrics.json", help="Path to test metrics JSON")
    parser.add_argument("--error_file", default="artifacts/recognition/error_analysis.json", help="Path to error analysis JSON")
    parser.add_argument("--manifest_path", default="data/manifests/test_manifest.json", help="Test manifest for glyph images")
    parser.add_argument("--output_dir", default="artifacts/recognition/visualizations", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    if Path(args.metrics_file).exists():
        with open(args.metrics_file) as f:
            metrics = json.load(f)
        cm_path = out_dir / "confusion_matrix.png"
        draw_confusion_matrix_image(metrics["confusion_matrix"], output_path=str(cm_path))

    # 2. Misclassified and Low-Confidence Grids
    if Path(args.error_file).exists() and Path(args.manifest_path).exists():
        with open(args.error_file) as f:
            err_data = json.load(f)

        from src.recognition.dataset import ChequeDigitDataset
        test_dataset = ChequeDigitDataset(manifest_path=args.manifest_path, augment=False)

        # Attach glyph arrays
        incorrect_samples = []
        for case in err_data.get("incorrect_cases", []):
            s_idx = case["sample_index"]
            if s_idx < len(test_dataset.samples):
                case_copy = dict(case)
                case_copy["glyph_array"] = test_dataset.samples[s_idx]["glyph_array"]
                incorrect_samples.append(case_copy)

        low_conf_samples = []
        for case in err_data.get("low_confidence_cases", []):
            s_idx = case["sample_index"]
            if s_idx < len(test_dataset.samples):
                case_copy = dict(case)
                case_copy["glyph_array"] = test_dataset.samples[s_idx]["glyph_array"]
                low_conf_samples.append(case_copy)

        if incorrect_samples:
            draw_prediction_grid(
                incorrect_samples,
                output_path=str(out_dir / "misclassified_digits.png"),
                title="Misclassified Digit Predictions (Held-Out Test Set)",
                max_samples=16,
            )

        if low_conf_samples:
            draw_prediction_grid(
                low_conf_samples,
                output_path=str(out_dir / "low_confidence_digits.png"),
                title="Low-Confidence Predictions (< 0.70 Confidence)",
                max_samples=16,
            )

    print(f"\nVisualizations successfully generated in: {out_dir}")
