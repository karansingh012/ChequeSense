"""Image preprocessing utilities tailored for OCR on cropped cheque fields.

Enhances text legibility, normalizes contrast, removes paper texture and line artifacts,
and rescales cropped regions to optimize optical character recognition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


class OCRPreprocessor:
    """Preprocesses cropped cheque fields prior to OCR text extraction."""

    def __init__(
        self,
        target_dpi_scale: float = 2.0,
        enable_clahe: bool = True,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.dpi_scale = target_dpi_scale
        self.enable_clahe = enable_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def load_image(self, image_input: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        """Loads and standardizes input to BGR numpy array."""
        if isinstance(image_input, (str, Path)):
            p = str(image_input)
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(f"Image not found or unreadable: {p}")
            return img
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input)
            if len(rgb.shape) == 2:
                return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            return image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def preprocess_field(
        self,
        image_input: Union[Image.Image, np.ndarray, str, Path],
        field_type: str = "general",
        binarize: bool = False,
    ) -> np.ndarray:
        """Applies field-specific preprocessing pipeline.

        Supported field types:
        - 'acno': Account number (numerical, printed or handwritten)
        - 'ifsc': IFSC code (alphanumeric, printed, high precision)
        - 'date': Cheque date (digits inside boxes/delimiters)
        - 'amount': Courtesy amount in figures (numerical)
        - 'name': Payee name (text line, cursive or printed)
        - 'amt_in_words': Legal amount in words (multi-word text)
        - 'general': Default robust preprocessing
        """
        img_bgr = self.load_image(image_input)
        h, w = img_bgr.shape[:2]

        if h == 0 or w == 0:
            return img_bgr

        # 1. Upscale if cropped region is small (Tesseract needs >= 30px x-height)
        min_dim = min(h, w)
        if min_dim < 50 or h < 60:
            scale = max(2.0, 70.0 / max(h, 1))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif self.dpi_scale > 1.0 and (h < 120 or w < 300):
            new_w = int(round(w * self.dpi_scale))
            new_h = int(round(h * self.dpi_scale))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 2. Convert to Grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 3. Denoising & Background Texture Suppression
        if field_type in ("name", "amt_in_words"):
            # Bilateral filter preserves soft cursive ink edges while smoothing security pantographs
            denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        else:
            # Gaussian blur for printed codes and digits
            denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # 4. Contrast Enhancement via CLAHE
        if self.enable_clahe:
            enhanced = self.clahe.apply(denoised)
        else:
            enhanced = denoised

        # 5. Optional Binarization / Thresholding
        if binarize or field_type in ("ifsc", "acno", "date"):
            # Use Otsu thresholding with slight margin
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Check polarity: ensure dark text (0) on light background (255)
            if np.mean(thresh) < 127:
                thresh = cv2.bitwise_not(thresh)

            # Morphological noise removal for salt-and-pepper artifacts
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            return cleaned

        return enhanced

    def preprocess_to_pil(
        self,
        image_input: Union[Image.Image, np.ndarray, str, Path],
        field_type: str = "general",
        binarize: bool = False,
    ) -> Image.Image:
        """Preprocesses field and returns a PIL Image."""
        processed_arr = self.preprocess_field(image_input, field_type=field_type, binarize=binarize)
        return Image.fromarray(processed_arr)
