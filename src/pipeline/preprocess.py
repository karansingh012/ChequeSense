"""Input preprocessing stage for the ChequeSense end-to-end inference pipeline."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger("chequesense.pipeline.preprocess")


class PipelineImagePreprocessor:
    """Validates, decodes, and standardizes raw cheque inputs for downstream pipeline stages."""

    MIN_WIDTH = 200
    MIN_HEIGHT = 100

    def __init__(self, target_detection_size: Tuple[int, int] = (1000, 460)):
        self.target_detection_size = target_detection_size

    def load_and_validate(
        self, image_input: Union[str, Path, Image.Image, np.ndarray, bytes]
    ) -> Tuple[Image.Image, str]:
        """Loads and verifies input image, returning standardized PIL RGB Image and cheque ID."""
        cheque_id = "cheque_input"

        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Cheque image file not found: {p}")
            cheque_id = p.stem
            try:
                pil_img = Image.open(p).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to decode image from path {p}: {e}") from e

        elif isinstance(image_input, bytes):
            try:
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to decode image from raw bytes: {e}") from e

        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                pil_img = Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[2] == 4:
                pil_img = Image.fromarray(image_input[:, :, :3]).convert("RGB")
            else:
                # Assume OpenCV BGR format
                rgb = image_input[:, :, ::-1]
                pil_img = Image.fromarray(rgb).convert("RGB")

        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        w, h = pil_img.size
        if w < self.MIN_WIDTH or h < self.MIN_HEIGHT:
            raise ValueError(
                f"Image dimensions ({w}x{h}) below minimum required resolution ({self.MIN_WIDTH}x{self.MIN_HEIGHT})"
            )

        logger.debug("Successfully validated cheque input '%s' (Dimensions: %dx%d)", cheque_id, w, h)
        return pil_img, cheque_id
