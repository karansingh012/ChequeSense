"""Preprocessing and character glyph segmentation utilities for ChequeSense recognition."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image


class GlyphPreprocessor:
    """Preprocesses and normalizes character and digit glyphs for recognition."""

    def __init__(self, target_size: Tuple[int, int] = (32, 32), pad_ratio: float = 0.15):
        self.target_size = target_size
        self.pad_ratio = pad_ratio

    def normalize_glyph(self, glyph_input: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Normalizes a single character glyph into a standardized centered square image.

        Output is a (target_size) float32 numpy array with values in [0.0, 1.0],
        where foreground stroke is white (> 0.5) on dark background (< 0.5).
        """
        if isinstance(glyph_input, np.ndarray) and glyph_input.shape == self.target_size and glyph_input.dtype == np.float32:
            return glyph_input

        if isinstance(glyph_input, Image.Image):
            arr = np.array(glyph_input.convert("L"))
        else:
            raw_arr = glyph_input if len(glyph_input.shape) == 2 else cv2.cvtColor(glyph_input, cv2.COLOR_BGR2GRAY)
            if raw_arr.dtype in (np.float32, np.float64):
                arr = (raw_arr * 255.0).clip(0, 255).astype(np.uint8) if raw_arr.max() <= 1.0 else raw_arr.astype(np.uint8)
            else:
                arr = raw_arr.astype(np.uint8)

        # Standardize polarity: cheque documents are dark text on light background
        # If mean is bright (> 127), invert so stroke is white (255) on black (0)
        if np.mean(arr) > 127:
            _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find tight bounding box of stroke pixels
        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            tight_crop = binary[y : y + h, x : x + w]
        else:
            tight_crop = binary

        h, w = tight_crop.shape
        target_w, target_h = self.target_size

        # Usable dimension after border margin
        usable_w = int(target_w * (1.0 - 2 * self.pad_ratio))
        usable_h = int(target_h * (1.0 - 2 * self.pad_ratio))

        scale = min(usable_w / max(w, 1), usable_h / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(tight_crop, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

        # Center in square canvas of target_size
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        # Normalize to [0.0, 1.0] float32
        return (canvas.astype(np.float32) / 255.0)

    def segment_field_into_glyphs(
        self, field_image: Union[Image.Image, np.ndarray], min_char_h: int = 8, min_char_w: int = 3
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Segments a multi-character field crop (e.g. date box or amount figures box) into glyphs.

        Returns:
            List of (normalized_glyph_32x32, (x, y, w, h)_in_field_image)
        """
        if isinstance(field_image, Image.Image):
            arr = np.array(field_image.convert("L"))
        else:
            arr = field_image if len(field_image.shape) == 2 else cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)

        # Invert so stroke is white
        if np.mean(arr) > 127:
            _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find connected contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter noise and large borders
            if w >= min_char_w and h >= min_char_h and w <= arr.shape[1] * 0.4 and h <= arr.shape[0] * 0.9:
                candidates.append((x, y, w, h))

        # Sort horizontally (left-to-right)
        candidates.sort(key=lambda b: b[0])

        # Filter overlapping or duplicate contours
        filtered = []
        for b in candidates:
            if not filtered or b[0] > filtered[-1][0] + 3:
                filtered.append(b)

        glyphs = []
        pad = 2
        for x, y, w, h in filtered:
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(arr.shape[1], x + w + pad)
            y2 = min(arr.shape[0], y + h + pad)
            crop = arr[y1:y2, x1:x2]
            norm = self.normalize_glyph(crop)
            glyphs.append((norm, (x, y, w, h)))

        return glyphs
