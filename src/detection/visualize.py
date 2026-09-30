"""Visualization utilities for ChequeSense field detections.

Draws high-contrast, color-coded bounding boxes and confidence labels on cheque
images and saves visual artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.detection.inference import FieldBox, FieldDetector, FieldPrediction

# Distinct color palette (RGB)
FIELD_COLORS_RGB = {
    "date": (0, 220, 255),      # Cyan
    "amount": (0, 230, 70),     # Bright Green
    "ifsc": (255, 140, 0),      # Dark Orange
    "acno": (220, 20, 180),     # Magenta / Purple
    "sign": (255, 50, 50),      # Crimson Red
    "name": (255, 215, 0),      # Golden Yellow
}

# OpenCV BGR equivalents
FIELD_COLORS_BGR = {
    k: (v[2], v[1], v[0]) for k, v in FIELD_COLORS_RGB.items()
}


def draw_field_detections(
    image: Union[Image.Image, np.ndarray],
    prediction: FieldPrediction,
    show_confidence: bool = True,
    line_thickness: int = 4,
) -> Image.Image:
    """Draws predicted bounding boxes and labels onto the cheque image."""
    if isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image).convert("RGB")

    draw = ImageDraw.Draw(pil_img)

    for field_name, box in prediction.fields.items():
        color = FIELD_COLORS_RGB.get(field_name, (0, 120, 255))
        coords = [box.xmin, box.ymin, box.xmax, box.ymax]

        # Draw thick rectangle
        for offset in range(line_thickness):
            draw.rectangle(
                [box.xmin - offset, box.ymin - offset, box.xmax + offset, box.ymax + offset],
                outline=color,
            )

        # Label badge
        label_text = f"{field_name.upper()}"
        if show_confidence:
            label_text += f" ({box.confidence:.2f})"

        # Background badge for text
        text_bbox = draw.textbbox((box.xmin, box.ymin - 24), label_text)
        draw.rectangle(
            [text_bbox[0] - 4, text_bbox[1] - 4, text_bbox[2] + 4, text_bbox[3] + 4],
            fill=(20, 20, 20),
        )
        draw.text((box.xmin, box.ymin - 24), label_text, fill=color)

    return pil_img


def draw_side_by_side(
    image: Image.Image,
    gt_boxes: Dict[str, Dict[str, int]],
    pred_boxes: Dict[str, FieldBox],
    title: str = "Ground Truth vs Predictions",
) -> Image.Image:
    """Creates a side-by-side comparison image of Ground Truth and Predicted fields."""
    orig_w, orig_h = image.size

    # 1. Ground Truth visualization
    gt_img = image.copy().convert("RGB")
    draw_gt = ImageDraw.Draw(gt_img)
    for fname, coords in gt_boxes.items():
        color = FIELD_COLORS_RGB.get(fname, (0, 120, 255))
        draw_gt.rectangle([coords["xmin"], coords["ymin"], coords["xmax"], coords["ymax"]], outline=color, width=4)
        draw_gt.text((coords["xmin"], coords["ymin"] - 20), f"GT: {fname.upper()}", fill=color)

    # 2. Prediction visualization
    pred_img = image.copy().convert("RGB")
    draw_pred = ImageDraw.Draw(pred_img)
    for fname, box in pred_boxes.items():
        color = FIELD_COLORS_RGB.get(fname, (0, 120, 255))
        draw_pred.rectangle([box.xmin, box.ymin, box.xmax, box.ymax], outline=color, width=4)
        draw_pred.text((box.xmin, box.ymin - 20), f"PRED: {fname.upper()} ({box.confidence:.2f})", fill=color)

    # Stack vertically or horizontally
    target_w = 1200
    ratio = target_w / orig_w
    target_h = int(orig_h * ratio)

    gt_resized = gt_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    pred_resized = pred_img.resize((target_w, target_h), Image.Resampling.BILINEAR)

    combined = Image.new("RGB", (target_w, target_h * 2 + 30), color=(30, 30, 30))
    combined.paste(gt_resized, (0, 10))
    combined.paste(pred_resized, (0, target_h + 20))

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize field detection predictions.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to cheque image")
    parser.add_argument("--model_path", type=str, default="models/field_detector/best_model.pt")
    parser.add_argument("--output_path", type=str, default="artifacts/field_detection/sample_prediction.png")
    args = parser.parse_args()

    detector = FieldDetector(model_path=args.model_path)
    pred = detector.predict(args.image_path)
    img = Image.open(args.image_path)
    vis = draw_field_detections(img, pred)

    out_p = Path(args.output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_p)
    print(f"Visualization saved to: {out_p}")
