"""Dataset validation utility for ChequeSense.

Detects corrupted images, invalid annotations, duplicate files, and cross-split
data leakage.
"""

from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from PIL import Image

from src.data.dataset_loader import ChequeSample


@dataclass
class ValidationReport:
    """Contains comprehensive validation diagnostics for a dataset or split."""

    total_samples: int = 0
    valid_samples: int = 0
    corrupted_samples: int = 0
    duplicate_groups: int = 0
    duplicate_instances: int = 0
    duplicates_by_hash: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    leakage_by_pair: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    corrupted_details: List[Dict[str, Any]] = field(default_factory=list)
    annotation_errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return self.corrupted_samples > 0 or len(self.annotation_errors) > 0

    @property
    def has_leakage(self) -> bool:
        return any(len(v) > 0 for v in self.leakage_by_pair.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "corrupted_samples": self.corrupted_samples,
            "duplicate_groups": self.duplicate_groups,
            "duplicate_instances": self.duplicate_instances,
            "has_leakage": self.has_leakage,
            "leakage_pairs_count": {k: len(v) for k, v in self.leakage_by_pair.items()},
            "annotation_errors_count": len(self.annotation_errors),
            "warnings_count": len(self.warnings),
        }


class DatasetValidator:
    """Validates ChequeSense samples, annotations, and split integrity."""

    @staticmethod
    def validate_image_bytes(image_bytes: bytes) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
        """Checks if raw image bytes can be decoded properly.

        Returns:
            (is_valid, error_message, (width, height))
        """
        if not image_bytes or len(image_bytes) == 0:
            return False, "Empty image bytes", None

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()

            # Re-open for size (verify() can leave image unreadable for subsequent operations)
            with Image.open(io.BytesIO(image_bytes)) as img:
                return True, None, img.size
        except Exception as e:
            return False, f"Image decoding failed: {str(e)}", None

    @classmethod
    def validate_sample(cls, sample: ChequeSample) -> Tuple[bool, List[str]]:
        """Validates a single ChequeSample instance.

        Returns:
            (is_valid, list_of_issues)
        """
        issues: List[str] = []

        # 1. Image integrity check
        try:
            img = sample.get_image()
            if img.width <= 0 or img.height <= 0:
                issues.append(f"Invalid dimensions: ({img.width}, {img.height})")
            if sample.width > 0 and sample.height > 0:
                if (sample.width, sample.height) != img.size:
                    issues.append(
                        f"Dimension mismatch: metadata=({sample.width}, {sample.height}) vs decoded={img.size}"
                    )
        except Exception as e:
            issues.append(f"Failed to decode image: {str(e)}")

        # 2. Annotation check
        if sample.annotation_type == "bounding_box":
            bboxes = sample.annotations
            for field_name, coords in bboxes.items():
                if not isinstance(coords, dict):
                    issues.append(f"BBox '{field_name}' is not a dict")
                    continue
                xmin, ymin = coords.get("xmin", 0), coords.get("ymin", 0)
                xmax, ymax = coords.get("xmax", 0), coords.get("ymax", 0)
                if xmax <= xmin:
                    issues.append(f"BBox '{field_name}' invalid x-range: xmin={xmin}, xmax={xmax}")
                if ymax <= ymin:
                    issues.append(f"BBox '{field_name}' invalid y-range: ymin={ymin}, ymax={ymax}")
                if sample.width > 0 and xmax > sample.width:
                    issues.append(f"BBox '{field_name}' exceeds width: xmax={xmax} > {sample.width}")
                if sample.height > 0 and ymax > sample.height:
                    issues.append(f"BBox '{field_name}' exceeds height: ymax={ymax} > {sample.height}")

        elif sample.annotation_type in ("vqa_key_value", "vqa_entity_dict"):
            query = sample.annotations.get("query", "")
            if not query or len(query.strip()) == 0:
                issues.append("Empty VQA query string")
            gt = sample.annotations.get("ground_truth")
            if gt is None:
                issues.append("Missing VQA ground truth")

        elif sample.annotation_type == "ocr_text":
            gt = sample.annotations.get("ground_truth", {})
            text = gt.get("text", "")
            if not text or len(str(text).strip()) == 0:
                issues.append("Empty OCR ground truth text")

        return len(issues) == 0, issues

    @classmethod
    def audit_collection(cls, samples: Iterable[ChequeSample]) -> ValidationReport:
        """Audits a collection of samples for corruption, duplicates, and errors."""
        report = ValidationReport()
        hash_to_samples: Dict[str, List[str]] = defaultdict(list)

        for sample in samples:
            report.total_samples += 1
            is_valid, issues = cls.validate_sample(sample)

            if is_valid:
                report.valid_samples += 1
            else:
                report.corrupted_samples += 1
                report.corrupted_details.append({
                    "sample_id": sample.sample_id,
                    "dataset_name": sample.dataset_name,
                    "issues": issues,
                })

            if sample.image_hash:
                hash_to_samples[sample.image_hash].append(sample.sample_id)

        # Record duplicates
        for h, sids in hash_to_samples.items():
            if len(sids) > 1:
                report.duplicates_by_hash[h] = sids
                report.duplicate_groups += 1
                report.duplicate_instances += (len(sids) - 1)

        return report

    @classmethod
    def detect_leakage(
        cls, split_samples: Dict[str, Iterable[ChequeSample]]
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        """Detects identical images shared across different splits.

        Args:
            split_samples: e.g. {'train': [samples], 'val': [samples], 'test': [samples]}

        Returns:
            Dict mapping pair (e.g. 'train_val') to list of (hash, sample_id_a, sample_id_b)
        """
        split_hashes: Dict[str, Dict[str, str]] = {}
        for split_name, samples in split_samples.items():
            split_hashes[split_name] = {}
            for s in samples:
                if s.image_hash:
                    split_hashes[split_name][s.image_hash] = s.sample_id

        leakage: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        split_names = list(split_hashes.keys())

        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                pair_name = f"{s1}_{s2}_leakage"
                common_hashes = set(split_hashes[s1].keys()).intersection(set(split_hashes[s2].keys()))
                for h in common_hashes:
                    leakage[pair_name].append((h, split_hashes[s1][h], split_hashes[s2][h]))

        return dict(leakage)
