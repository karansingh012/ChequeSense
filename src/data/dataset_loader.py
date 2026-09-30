"""Dataset loader for ChequeSense.

Safely loads datasets from `Dataset/` (IDRBT, synthetic, handwritten & cheques)
without modifying any raw files. Supports in-memory decoding and lazy loading.
"""

from __future__ import annotations

import ast
import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


@dataclass
class ChequeSample:
    """Represents a single document or text line sample in ChequeSense."""

    sample_id: str
    dataset_name: str  # 'synthetic', 'idrbt_300', 'handwritten_ocr', 'cheque_vqa', 'idrbt_entity_vqa'
    source_dataset: str  # 'synthetic', 'IDRBT', 'handwritten_and_cheques_dataset'
    source_file: str
    source_split: Optional[str] = None
    width: int = 0
    height: int = 0
    color_mode: str = "RGB"
    image_hash: str = ""
    labels: List[str] = field(default_factory=list)
    annotation_type: str = "none"  # 'bounding_box', 'vqa', 'ocr_text', 'none'
    annotations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _image_bytes: Optional[bytes] = None
    _image_getter: Optional[Callable[[], Image.Image]] = None

    def get_image(self) -> Image.Image:
        """Returns the PIL Image instance, decoding lazily if necessary."""
        if self._image_getter is not None:
            return self._image_getter()
        if self._image_bytes is not None:
            img = Image.open(io.BytesIO(self._image_bytes))
            # Load into memory so the bytes buffer can be closed if needed
            img.load()
            return img
        raise ValueError(f"No image data or getter available for sample {self.sample_id}")

    @property
    def image_bytes(self) -> bytes:
        """Returns the raw binary image bytes."""
        if self._image_bytes is not None:
            return self._image_bytes
        if self._image_getter is not None:
            img = self._image_getter()
            buf = io.BytesIO()
            img.save(buf, format=img.format or "PNG")
            return buf.getvalue()
        raise ValueError(f"No image bytes available for sample {self.sample_id}")


class DatasetLoader:
    """Loader for the three ChequeSense datasets."""

    def __init__(self, base_dir: Union[str, Path] = "Dataset"):
        self.base_dir = Path(base_dir)
        self.idrbt_dir = self.base_dir / "IDRBT" / "300"
        self.synthetic_dir = self.base_dir / "synthetic" / "data"
        self.handwritten_dir = self.base_dir / "handwritten_and_cheques_dataset" / "data"

    @staticmethod
    def compute_sha256(data: bytes) -> str:
        """Compute SHA-256 hexadecimal digest of byte stream."""
        return hashlib.sha256(data).hexdigest()

    def load_idrbt(self, lazy: bool = True) -> Generator[ChequeSample, None, None]:
        """Loads samples from IDRBT/300 TIFF scans.

        Args:
            lazy: If True, delays full image reading until sample.get_image() is called.
        """
        if not self.idrbt_dir.exists():
            raise FileNotFoundError(f"IDRBT directory not found: {self.idrbt_dir}")

        for fname in sorted(os.listdir(self.idrbt_dir)):
            if fname.startswith(".") or not fname.lower().endswith((".tif", ".tiff")):
                continue

            file_path = self.idrbt_dir / fname
            sample_id = f"idrbt_{Path(fname).stem.replace(' ', '_').lower()}"

            # Open image header to read dimensions and mode quickly
            with open(file_path, "rb") as f:
                img_bytes = f.read()

            img_hash = self.compute_sha256(img_bytes)

            with Image.open(io.BytesIO(img_bytes)) as probe_img:
                width, height = probe_img.size
                color_mode = probe_img.mode

            def make_getter(path: Path) -> Callable[[], Image.Image]:
                def getter() -> Image.Image:
                    im = Image.open(path)
                    im.load()
                    return im
                return getter

            sample = ChequeSample(
                sample_id=sample_id,
                dataset_name="idrbt_300",
                source_dataset="IDRBT",
                source_file=str(file_path),
                source_split="unsplit",
                width=width,
                height=height,
                color_mode=color_mode,
                image_hash=img_hash,
                labels=["cheque_full_scan"],
                annotation_type="none",
                annotations={},
                metadata={"format": "TIFF", "dpi": (300, 300), "raw_filename": fname},
                _image_bytes=None if lazy else img_bytes,
                _image_getter=make_getter(file_path),
            )
            yield sample

    def load_synthetic(
        self, split: Optional[str] = None, lazy: bool = True
    ) -> Generator[ChequeSample, None, None]:
        """Loads samples from Dataset/synthetic Parquet files.

        Args:
            split: 'train', 'validation', 'test', or None for all splits.
            lazy: If True, stores bytes and decodes PIL Image on demand.
        """
        if not self.synthetic_dir.exists():
            raise FileNotFoundError(f"Synthetic directory not found: {self.synthetic_dir}")

        splits = [split] if split else ["train", "validation", "test"]

        for sp in splits:
            parquet_file = self.synthetic_dir / f"{sp}-00000-of-00001.parquet"
            if not parquet_file.exists():
                continue

            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            for idx, row in df.iterrows():
                image_id = str(row["image_id"])
                bank = str(row["bank"])
                filename = str(row["filename"])
                meta_w = int(row["image_width"])
                meta_h = int(row["image_height"])

                img_struct = row["image"]
                img_bytes = img_struct.get("bytes") if isinstance(img_struct, dict) else None
                if not img_bytes:
                    continue

                img_hash = self.compute_sha256(img_bytes)

                # Extract bounding boxes
                bboxes = {}
                for field_name in ["date", "amount", "ifsc", "acno", "sign", "name"]:
                    b_val = row[field_name]
                    if isinstance(b_val, dict):
                        bboxes[field_name] = {
                            "xmin": int(b_val["xmin"]),
                            "ymin": int(b_val["ymin"]),
                            "xmax": int(b_val["xmax"]),
                            "ymax": int(b_val["ymax"]),
                        }

                sample = ChequeSample(
                    sample_id=f"syn_{image_id}",
                    dataset_name="synthetic",
                    source_dataset="synthetic",
                    source_file=filename,
                    source_split=sp,
                    width=meta_w,
                    height=meta_h,
                    color_mode="RGB",
                    image_hash=img_hash,
                    labels=list(bboxes.keys()),
                    annotation_type="bounding_box",
                    annotations=bboxes,
                    metadata={"bank": bank, "original_index": idx},
                    _image_bytes=img_bytes,
                )
                yield sample

    def load_handwritten_and_cheques(
        self,
        split: Optional[str] = None,
        category: Optional[str] = None,
        lazy: bool = True,
    ) -> Generator[ChequeSample, None, None]:
        """Loads samples from Dataset/handwritten_and_cheques_dataset Parquet files.

        Args:
            split: 'train', 'test', or None for both.
            category: 'cheque_vqa', 'handwritten_ocr', 'idrbt_entity_vqa', or None for all.
            lazy: If True, decodes PIL Image on demand.
        """
        if not self.handwritten_dir.exists():
            raise FileNotFoundError(f"Handwritten directory not found: {self.handwritten_dir}")

        splits = [split] if split else ["train", "test"]

        for sp in splits:
            parquet_file = self.handwritten_dir / f"{sp}-00000-of-00001.parquet"
            if not parquet_file.exists():
                continue

            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            for idx, row in df.iterrows():
                query = str(row["query"])
                raw_answers = str(row["answers"])

                img_struct = row["image"]
                img_bytes = img_struct.get("bytes") if isinstance(img_struct, dict) else None
                if not img_bytes:
                    continue

                img_hash = self.compute_sha256(img_bytes)

                # Determine category
                if query.strip() == "Extract all entities from the image.":
                    cat = "idrbt_entity_vqa"
                    ann_type = "vqa_entity_dict"
                    try:
                        parsed_ans = ast.literal_eval(raw_answers)
                    except Exception:
                        parsed_ans = {"raw_text": raw_answers}
                    labels = list(parsed_ans.keys()) if isinstance(parsed_ans, dict) else []
                elif any(
                    k in query.lower()
                    for k in ["cheque", "payee", "amount", "entities", "recipient"]
                ):
                    cat = "cheque_vqa"
                    ann_type = "vqa_key_value"
                    try:
                        parsed_ans = ast.literal_eval(raw_answers)
                    except Exception:
                        parsed_ans = {"raw_text": raw_answers}
                    labels = ["amt_in_words", "amt_in_figures", "payee_name", "bank_name", "cheque_date"]
                else:
                    cat = "handwritten_ocr"
                    ann_type = "ocr_text"
                    parsed_ans = {"text": raw_answers}
                    labels = ["handwritten_line_text"]

                if category and cat != category:
                    continue

                # Probe dimensions from image bytes
                with Image.open(io.BytesIO(img_bytes)) as im:
                    w, h = im.size
                    color_mode = im.mode

                sample = ChequeSample(
                    sample_id=f"hw_{sp}_{idx:05d}",
                    dataset_name=cat,
                    source_dataset="handwritten_and_cheques_dataset",
                    source_file=f"{sp}_{idx:05d}.jpg",
                    source_split=sp,
                    width=w,
                    height=h,
                    color_mode=color_mode,
                    image_hash=img_hash,
                    labels=labels,
                    annotation_type=ann_type,
                    annotations={
                        "query": query,
                        "ground_truth": parsed_ans,
                        "raw_answers": raw_answers,
                    },
                    metadata={"category": cat, "original_index": idx},
                    _image_bytes=img_bytes,
                )
                yield sample
