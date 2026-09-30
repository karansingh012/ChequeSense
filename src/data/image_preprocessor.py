"""Configurable image preprocessing pipeline for ChequeSense.

Preserves original images and applies transformations (color conversion, resizing,
denoising, contrast enhancement, thresholding, deskewing) strictly through
explicit configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessingConfig:
    """Configuration options for image preprocessing in ChequeSense."""

    # Color space
    color_mode: str = "unchanged"  # 'rgb', 'grayscale', 'unchanged'

    # Resizing
    target_size: Optional[Tuple[int, int]] = None  # (width, height)
    maintain_aspect_ratio: bool = True
    padding_color: int = 255  # White padding for document images

    # Denoising
    denoise: bool = False
    denoise_method: str = "bilateral"  # 'bilateral', 'gaussian', 'nlmeans'
    denoise_strength: int = 5

    # Contrast Enhancement
    enhance_contrast: bool = False
    contrast_method: str = "clahe"  # 'clahe', 'hist_eq'
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)

    # Thresholding / Binarization
    threshold_mode: Optional[str] = None  # 'otsu', 'adaptive', None
    adaptive_block_size: int = 11
    adaptive_c: int = 2

    # Deskewing
    deskew: bool = False
    max_skew_angle: float = 25.0  # Ignore angles beyond this threshold (safety guard)

    # Normalization
    normalize: bool = False
    normalization_type: str = "scale_0_1"  # 'scale_0_1', 'zscore'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "color_mode": self.color_mode,
            "target_size": list(self.target_size) if self.target_size else None,
            "maintain_aspect_ratio": self.maintain_aspect_ratio,
            "denoise": self.denoise,
            "denoise_method": self.denoise_method,
            "enhance_contrast": self.enhance_contrast,
            "contrast_method": self.contrast_method,
            "threshold_mode": self.threshold_mode,
            "deskew": self.deskew,
            "normalize": self.normalize,
            "normalization_type": self.normalization_type,
        }


@dataclass
class PreprocessedResult:
    """Output container for preprocessed image and transformation metadata."""

    image: Union[np.ndarray, Image.Image]
    original_size: Tuple[int, int]  # (width, height)
    processed_size: Tuple[int, int]  # (width, height)
    scale_factors: Tuple[float, float] = (1.0, 1.0)  # (scale_x, scale_y)
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (pad_left, pad_top, pad_right, pad_bottom)
    skew_angle: float = 0.0
    config: PreprocessingConfig = None


class ImagePreprocessor:
    """Applies modular, deterministic image processing operations."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    @staticmethod
    def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
        """Converts PIL Image to OpenCV BGR or Grayscale numpy array."""
        if pil_img.mode == "L":
            return np.array(pil_img)
        elif pil_img.mode == "RGBA":
            rgb = pil_img.convert("RGB")
            return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        else:
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
        """Converts OpenCV numpy array back to PIL Image."""
        if len(cv_img.shape) == 2:
            return Image.fromarray(cv_img)
        elif len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        return Image.fromarray(cv_img)

    def estimate_skew_angle(self, cv_img: np.ndarray) -> float:
        """Estimates the skew angle of a document using text contours."""
        # Convert to grayscale if necessary
        gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Invert colors and threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Morphological dilation to merge text into horizontal text blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        angles: List[float] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 1000:
                continue

            rect = cv2.minAreaRect(c)
            angle = rect[-1]

            # Normalize angle to [-45, 45] range
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle

            if abs(angle) <= self.config.max_skew_angle:
                angles.append(angle)

        if not angles:
            return 0.0

        # Return median angle
        return float(np.median(angles))

    def rotate_image(self, cv_img: np.ndarray, angle: float) -> np.ndarray:
        """Rotates image around its center by angle degrees with white background."""
        if abs(angle) < 0.2:
            return cv_img

        h, w = cv_img.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            cv_img,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(self.config.padding_color, self.config.padding_color, self.config.padding_color)
            if len(cv_img.shape) == 3
            else self.config.padding_color,
        )
        return rotated

    def resize_with_aspect_ratio(
        self, cv_img: np.ndarray, target_w: int, target_h: int
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int, int, int]]:
        """Resizes image to target dimensions, optionally padding to preserve aspect ratio."""
        h, w = cv_img.shape[:2]

        if not self.config.maintain_aspect_ratio:
            resized = cv2.resize(cv_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            scale_x = target_w / w
            scale_y = target_h / h
            return resized, (scale_x, scale_y), (0, 0, 0, 0)

        # Compute scaling ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=interp)

        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top

        pad_val = (
            (self.config.padding_color, self.config.padding_color, self.config.padding_color)
            if len(cv_img.shape) == 3
            else self.config.padding_color
        )
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=pad_val,
        )

        return padded, (scale, scale), (pad_left, pad_top, pad_right, pad_bottom)

    def process(
        self,
        image_input: Union[Image.Image, np.ndarray],
        return_pil: bool = True,
    ) -> PreprocessedResult:
        """Executes configured preprocessing sequence on a single image."""
        if isinstance(image_input, Image.Image):
            cv_img = self.pil_to_cv2(image_input)
            orig_size = image_input.size
        elif isinstance(image_input, np.ndarray):
            cv_img = image_input.copy()
            orig_size = (cv_img.shape[1], cv_img.shape[0])
        else:
            raise TypeError("image_input must be a PIL.Image.Image or numpy.ndarray")

        skew_angle = 0.0
        scale_factors = (1.0, 1.0)
        padding = (0, 0, 0, 0)

        # 1. Color Conversion
        if self.config.color_mode == "grayscale":
            if len(cv_img.shape) == 3:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        elif self.config.color_mode == "rgb":
            if len(cv_img.shape) == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)

        # 2. Deskewing
        if self.config.deskew:
            skew_angle = self.estimate_skew_angle(cv_img)
            cv_img = self.rotate_image(cv_img, skew_angle)

        # 3. Denoising
        if self.config.denoise:
            if self.config.denoise_method == "bilateral":
                if len(cv_img.shape) == 2:
                    cv_img = cv2.bilateralFilter(cv_img, 7, 50, 50)
                else:
                    cv_img = cv2.bilateralFilter(cv_img, 7, 50, 50)
            elif self.config.denoise_method == "gaussian":
                k = self.config.denoise_strength if self.config.denoise_strength % 2 == 1 else self.config.denoise_strength + 1
                cv_img = cv2.GaussianBlur(cv_img, (k, k), 0)
            elif self.config.denoise_method == "nlmeans":
                if len(cv_img.shape) == 2:
                    cv_img = cv2.fastNlMeansDenoising(cv_img, h=self.config.denoise_strength)
                else:
                    cv_img = cv2.fastNlMeansDenoisingColored(cv_img, h=self.config.denoise_strength)

        # 4. Contrast Enhancement
        if self.config.enhance_contrast:
            if self.config.contrast_method == "clahe":
                clahe = cv2.createCLAHE(
                    clipLimit=self.config.clahe_clip_limit,
                    tileGridSize=self.config.clahe_grid_size,
                )
                if len(cv_img.shape) == 2:
                    cv_img = clahe.apply(cv_img)
                else:
                    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    l2 = clahe.apply(l)
                    lab2 = cv2.merge((l2, a, b))
                    cv_img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
            elif self.config.contrast_method == "hist_eq":
                if len(cv_img.shape) == 2:
                    cv_img = cv2.equalizeHist(cv_img)
                else:
                    ycrcb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YCrCb)
                    y, cr, cb = cv2.split(ycrcb)
                    y = cv2.equalizeHist(y)
                    cv_img = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

        # 5. Thresholding / Binarization
        if self.config.threshold_mode is not None:
            gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            if self.config.threshold_mode == "otsu":
                _, cv_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif self.config.threshold_mode == "adaptive":
                block = self.config.adaptive_block_size
                block = block if block % 2 == 1 else block + 1
                cv_img = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, self.config.adaptive_c
                )

        # 6. Resizing
        if self.config.target_size is not None:
            target_w, target_h = self.config.target_size
            cv_img, scale_factors, padding = self.resize_with_aspect_ratio(cv_img, target_w, target_h)

        # 7. Normalization
        output_data: Union[np.ndarray, Image.Image]
        if self.config.normalize:
            float_img = cv_img.astype(np.float32)
            if self.config.normalization_type == "scale_0_1":
                output_data = float_img / 255.0
            elif self.config.normalization_type == "zscore":
                mean = np.mean(float_img)
                std = np.std(float_img) + 1e-7
                output_data = (float_img - mean) / std
            else:
                output_data = float_img / 255.0
            # Float normalized cannot be returned as PIL directly
            return_pil = False
        else:
            output_data = self.cv2_to_pil(cv_img) if return_pil else cv_img

        final_size = (
            (output_data.width, output_data.height)
            if isinstance(output_data, Image.Image)
            else (output_data.shape[1], output_data.shape[0])
        )

        return PreprocessedResult(
            image=output_data,
            original_size=orig_size,
            processed_size=final_size,
            scale_factors=scale_factors,
            padding=padding,
            skew_angle=skew_angle,
            config=self.config,
        )

    @staticmethod
    def transform_bounding_box(
        bbox: Dict[str, int],
        scale_factors: Tuple[float, float],
        padding: Tuple[int, int, int, int],
    ) -> Dict[str, int]:
        """Adjusts a bounding box given scaling factors and border padding."""
        scale_x, scale_y = scale_factors
        pad_left, pad_top, _, _ = padding

        return {
            "xmin": int(round(bbox["xmin"] * scale_x + pad_left)),
            "ymin": int(round(bbox["ymin"] * scale_y + pad_top)),
            "xmax": int(round(bbox["xmax"] * scale_x + pad_left)),
            "ymax": int(round(bbox["ymax"] * scale_y + pad_top)),
        }
