"""PyTorch Dataset for handwritten and cheque digit recognition in ChequeSense."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from src.data.dataset_loader import DatasetLoader
from src.recognition.preprocessing import GlyphPreprocessor

DIGIT_CLASSES = [str(i) for i in range(10)]
CLASS_TO_IDX = {ch: idx for idx, ch in enumerate(DIGIT_CLASSES)}
IDX_TO_CLASS = {idx: ch for idx, ch in enumerate(DIGIT_CLASSES)}
NUM_CLASSES = len(DIGIT_CLASSES)  # 10 digits (0 to 9)


class ChequeDigitDataset(Dataset):
    """PyTorch Dataset for handwritten/cheque digit classification."""

    def __init__(
        self,
        manifest_path: Union[str, Path] = "data/manifests/train_manifest.json",
        target_size: Tuple[int, int] = (32, 32),
        augment: bool = False,
        cache_dir: Optional[str] = None,
        base_dir: str = "Dataset",
    ):
        self.manifest_path = Path(manifest_path)
        self.target_size = target_size
        self.augment = augment
        self.preprocessor = GlyphPreprocessor(target_size=target_size)
        self.loader = DatasetLoader(base_dir=base_dir)

        # Extract or load cached glyphs
        self.samples: List[Dict[str, Any]] = self._extract_glyphs()

    def _extract_glyphs(self) -> List[Dict[str, Any]]:
        """Extracts aligned digit glyphs from numerical fields in the manifest."""
        with open(self.manifest_path) as f:
            records = json.load(f)

        cheque_records = [r for r in records if r["dataset_name"] == "cheque_vqa"]

        # Cache sample objects from dataset loader
        sample_cache = {s.sample_id: s for s in self.loader.load_handwritten_and_cheques(split=None)}

        extracted: List[Dict[str, Any]] = []

        for r in cheque_records:
            sid = r["sample_id"]
            cheque_sample = sample_cache.get(sid)
            if not cheque_sample:
                continue

            im = cheque_sample.get_image()
            gt = r["annotations"].get("ground_truth", {})
            if not isinstance(gt, list):
                continue

            # 1. Extract from amount in figures
            fig_vals = [d.get("amt_in_figures") for d in gt if "amt_in_figures" in d]
            if fig_vals and fig_vals[0]:
                fig_str = str(fig_vals[0]).replace(",", "").strip()
                if fig_str.isdigit():
                    # Crop amount figures region [340:480, 80:140]
                    amt_crop = im.crop((340, 80, 480, 140))
                    arr = np.array(amt_crop)
                    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    boxes = []
                    for c in contours:
                        x, y, w, h = cv2.boundingRect(c)
                        if 4 <= w <= 40 and 8 <= h <= 45:
                            boxes.append((x, y, w, h))
                    boxes.sort(key=lambda b: b[0])

                    filtered = []
                    for b in boxes:
                        if not filtered or b[0] > filtered[-1][0] + 3:
                            filtered.append(b)

                    if len(filtered) == len(fig_str):
                        for ch, (x, y, w, h) in zip(fig_str, filtered):
                            pad = 2
                            x1 = max(0, x - pad)
                            y1 = max(0, y - pad)
                            x2 = min(amt_crop.width, x + w + pad)
                            y2 = min(amt_crop.height, y + h + pad)
                            glyph_crop = amt_crop.crop((x1, y1, x2, y2))
                            norm_glyph = self.preprocessor.normalize_glyph(glyph_crop)

                            extracted.append({
                                "glyph_array": norm_glyph,
                                "label_char": ch,
                                "label_idx": CLASS_TO_IDX[ch],
                                "source_sample_id": sid,
                                "field_type": "amt_in_figures",
                            })

            # 2. Extract from date digits
            date_vals = [d.get("cheque_date") for d in gt if "cheque_date" in d]
            if date_vals and date_vals[0]:
                dt_str = str(date_vals[0]).strip()
                digits_only = [c for c in dt_str if c.isdigit()]
                if len(digits_only) in (6, 8):
                    date_crop = im.crop((350, 10, 480, 50))
                    arr = np.array(date_crop)
                    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    boxes = []
                    for c in contours:
                        x, y, w, h = cv2.boundingRect(c)
                        if 3 <= w <= 25 and 8 <= h <= 35:
                            boxes.append((x, y, w, h))
                    boxes.sort(key=lambda b: b[0])

                    filtered = []
                    for b in boxes:
                        if not filtered or b[0] > filtered[-1][0] + 3:
                            filtered.append(b)

                    if len(filtered) == len(digits_only):
                        for ch, (x, y, w, h) in zip(digits_only, filtered):
                            pad = 2
                            x1 = max(0, x - pad)
                            y1 = max(0, y - pad)
                            x2 = min(date_crop.width, x + w + pad)
                            y2 = min(date_crop.height, y + h + pad)
                            glyph_crop = date_crop.crop((x1, y1, x2, y2))
                            norm_glyph = self.preprocessor.normalize_glyph(glyph_crop)

                            extracted.append({
                                "glyph_array": norm_glyph,
                                "label_char": ch,
                                "label_idx": CLASS_TO_IDX[ch],
                                "source_sample_id": sid,
                                "field_type": "cheque_date",
                            })

        return extracted

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        item = self.samples[idx]
        arr = item["glyph_array"]  # [32, 32] float32 in [0.0, 1.0]
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, 32, 32]

        if self.augment:
            # Random rotation ±10 degrees
            angle = (torch.rand(1).item() - 0.5) * 20.0
            tensor = TF.rotate(tensor, angle, fill=0.0)

            # Random small translation
            if torch.rand(1).item() > 0.5:
                translate = [int((torch.rand(1).item() - 0.5) * 4), int((torch.rand(1).item() - 0.5) * 4)]
                tensor = TF.affine(tensor, angle=0.0, translate=translate, scale=1.0, shear=[0.0, 0.0], fill=0.0)

        label = item["label_idx"]
        meta = {
            "char": item["label_char"],
            "source_sample_id": item["source_sample_id"],
            "field_type": item["field_type"],
        }
        return tensor, label, meta
