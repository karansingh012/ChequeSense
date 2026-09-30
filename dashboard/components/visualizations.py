"""Visualization components and image bounding box renderers for ChequeSense dashboard."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image, ImageDraw, ImageFont


# Professional color palette for field bounding boxes
FIELD_COLORS = {
    "cheque_number": (37, 99, 235),   # Blue
    "amount": (16, 185, 129),          # Emerald Green
    "date": (217, 119, 6),             # Amber
    "payee_name": (139, 92, 246),      # Purple
    "account_number": (14, 165, 233),  # Cyan
    "ifsc": (236, 72, 153),            # Pink
    "signatures": (245, 158, 11),      # Orange
    "default": (100, 116, 139),        # Slate Gray
}


def draw_bounding_boxes_on_cheque(
    image_input: Union[str, Path, Image.Image],
    extracted_fields: List[Dict[str, Any]],
    highlight_low_confidence: bool = True,
) -> Image.Image:
    """Draws colored bounding boxes with field name tags onto the cheque image."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.copy().convert("RGB")

    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    for f in extracted_fields:
        bbox = f.get("bounding_box_json") or f.get("bounding_box")
        if not bbox or not isinstance(bbox, dict):
            continue

        xmin = int(bbox.get("xmin", 0))
        ymin = int(bbox.get("ymin", 0))
        xmax = int(bbox.get("xmax", 0))
        ymax = int(bbox.get("ymax", 0))

        # Clamp to image bounds
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(xmin + 1, min(xmax, w))
        ymax = max(ymin + 1, min(ymax, h))

        fname = f.get("field_name", "field")
        conf = float(f.get("confidence", 0.0))
        is_low_conf = conf < 0.70

        # Choose color
        if highlight_low_confidence and is_low_conf:
            box_color = (220, 38, 38)     # Crimson alert
            fill_color = (220, 38, 38, 30) # Translucent crimson
        else:
            base_col = FIELD_COLORS.get(fname, FIELD_COLORS["default"])
            box_color = base_col
            fill_color = (*base_col, 25)  # Translucent fill

        # Draw translucent rectangle and crisp border
        draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color, outline=box_color, width=3)

        # Draw field label badge above box
        label_text = f"{fname} ({int(conf * 100)}%)"
        text_bbox = draw.textbbox((xmin, max(0, ymin - 16)), label_text)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=box_color,
        )
        draw.text((xmin, max(0, ymin - 16)), label_text, fill=(255, 255, 255))

    return img


def format_currency_inr(val: Union[float, int, str]) -> str:
    """Formats a monetary value into Indian Rupee style string (e.g. ₹ 25,000.00)."""
    try:
        clean = "".join(ch for ch in str(val) if ch.isdigit() or ch == ".")
        num = float(clean)
        return f"₹ {num:,.2f}"
    except (ValueError, TypeError):
        return f"₹ {val}"
