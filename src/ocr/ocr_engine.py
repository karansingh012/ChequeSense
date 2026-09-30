"""OCR Engine wrapper for ChequeSense document and field text extraction.

Integrates Tesseract OCR via native sub-process TSV piping or pytesseract,
extracting raw text, word-level bounding boxes, and fine-grained confidence scores.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.ocr.preprocessing import OCRPreprocessor


@dataclass
class OCRWordToken:
    """Detailed token extracted by the OCR engine."""

    text: str
    confidence: float  # Normalized to [0.0, 1.0]
    left: int
    top: int
    width: int
    height: int
    line_num: int
    word_num: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "left": self.left,
                "top": self.top,
                "width": self.width,
                "height": self.height,
            },
            "line_num": self.line_num,
            "word_num": self.word_num,
        }


@dataclass
class OCRResult:
    """Full extraction result from the OCR engine."""

    raw_text: str
    tokens: List[OCRWordToken] = field(default_factory=list)
    mean_confidence: float = 0.0
    min_confidence: float = 0.0
    field_type: str = "general"
    engine_name: str = "tesseract"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "mean_confidence": round(self.mean_confidence, 4),
            "min_confidence": round(self.min_confidence, 4),
            "token_count": len(self.tokens),
            "tokens": [t.to_dict() for t in self.tokens],
            "field_type": self.field_type,
            "engine_name": self.engine_name,
        }


class OCREngine:
    """Production OCR interface for cheque field transcription."""

    # Common Tesseract Page Segmentation Modes (PSM)
    PSM_SINGLE_BLOCK = "6"      # Uniform block of text
    PSM_SINGLE_LINE = "7"       # Single line of text (IFSC, Account Number)
    PSM_SINGLE_WORD = "8"       # Single word (Code or amount)
    PSM_SPARSE_TEXT = "11"      # Sparse text finding
    PSM_RAW_LINE = "13"         # Raw line (bypasses internal hacks)

    # Field-to-PSM mapping
    FIELD_PSM_MAP = {
        "acno": PSM_SINGLE_LINE,
        "ifsc": PSM_SINGLE_LINE,
        "date": PSM_SINGLE_BLOCK,
        "amount": PSM_SINGLE_LINE,
        "name": PSM_SINGLE_LINE,
        "amt_in_words": PSM_SINGLE_BLOCK,
        "bank_name": PSM_SINGLE_LINE,
        "general": PSM_SINGLE_LINE,
    }

    # Field-specific character whitelists
    FIELD_WHITELISTS = {
        "acno": "0123456789",
        "date": "0123456789/-.",
        "amount": "0123456789,./-₹Rs",
        "ifsc": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    }

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        language: str = "eng",
        default_psm: str = PSM_SINGLE_LINE,
    ):
        self.language = language
        self.default_psm = default_psm
        self.preprocessor = OCRPreprocessor()
        self.tesseract_cmd = (
            tesseract_cmd
            or os.getenv("TESSERACT_CMD")
            or shutil.which("tesseract")
            or "/opt/homebrew/bin/tesseract"
        )

        if not Path(self.tesseract_cmd).exists() and not shutil.which(self.tesseract_cmd):
            raise RuntimeError(
                f"Tesseract binary not found at '{self.tesseract_cmd}'. Please install Tesseract."
            )

    def extract_text(
        self,
        image_input: Union[Image.Image, np.ndarray, str, Path],
        field_type: str = "general",
        psm: Optional[str] = None,
        apply_preprocessing: bool = True,
        binarize: bool = False,
        use_whitelist: bool = False,
    ) -> OCRResult:
        """Extracts text and word-level token confidences from a cropped field image."""
        # 1. Preprocessing
        if apply_preprocessing:
            processed_arr = self.preprocessor.preprocess_field(
                image_input, field_type=field_type, binarize=binarize
            )
        else:
            processed_arr = self.preprocessor.load_image(image_input)

        chosen_psm = psm or self.FIELD_PSM_MAP.get(field_type, self.default_psm)

        # 2. Configure CLI options
        extra_args = ["--psm", str(chosen_psm), "-l", self.language]

        # Config parameters
        config_params = []
        if use_whitelist and field_type in self.FIELD_WHITELISTS:
            config_params.extend(["-c", f"tessedit_char_whitelist={self.FIELD_WHITELISTS[field_type]}"])

        # 3. Execute Tesseract via temporary PNG image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_img:
            cv2.imwrite(tmp_img.name, processed_arr)

            # Note: in tesseract CLI, options and flags must precede the 'tsv' configfile
            cmd = [self.tesseract_cmd, tmp_img.name, "stdout"] + extra_args + config_params + ["tsv"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if proc.returncode != 0:
                # Fallback to simple stdout if tsv mode had any issue
                cmd_fallback = [self.tesseract_cmd, tmp_img.name, "stdout"] + extra_args
                proc_fallback = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                raw_text = proc_fallback.stdout.strip()
                return OCRResult(
                    raw_text=raw_text,
                    tokens=[],
                    mean_confidence=0.0,
                    min_confidence=0.0,
                    field_type=field_type,
                )

            tsv_output = proc.stdout

        # 4. Parse TSV Output
        tokens: List[OCRWordToken] = []
        raw_words: List[str] = []
        confidences: List[float] = []

        reader = csv.DictReader(io.StringIO(tsv_output), delimiter="\t")
        for row in reader:
            txt = row.get("text", "")
            if txt is None:
                continue
            txt = txt.strip()
            conf_str = row.get("conf", "-1")

            try:
                conf_val = float(conf_str)
            except ValueError:
                conf_val = -1.0

            # Level 5 corresponds to word tokens in Tesseract TSV
            if row.get("level") == "5" and txt:
                norm_conf = max(0.0, min(1.0, conf_val / 100.0)) if conf_val >= 0 else 0.0
                token = OCRWordToken(
                    text=txt,
                    confidence=norm_conf,
                    left=int(row.get("left", 0)),
                    top=int(row.get("top", 0)),
                    width=int(row.get("width", 0)),
                    height=int(row.get("height", 0)),
                    line_num=int(row.get("line_num", 1)),
                    word_num=int(row.get("word_num", 1)),
                )
                tokens.append(token)
                raw_words.append(txt)
                confidences.append(norm_conf)

        full_raw_text = " ".join(raw_words).strip()
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        min_conf = float(np.min(confidences)) if confidences else 0.0

        return OCRResult(
            raw_text=full_raw_text,
            tokens=tokens,
            mean_confidence=mean_conf,
            min_confidence=min_conf,
            field_type=field_type,
            engine_name="tesseract-5",
        )
