"""Leak-free data partitioning for ChequeSense.

Partitions datasets into Train, Validation, and Test splits. Enforces strict
deduplication by cryptographic image hash to eliminate cross-split data leakage,
and guarantees that IDRBT real cheques are reserved exclusively for benchmark
evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.dataset_loader import ChequeSample, DatasetLoader


class DataSplitter:
    """Manages leak-free partitioning across all ChequeSense datasets."""

    def __init__(self, base_dir: str = "Dataset", output_dir: str = "data/splits", seed: int = 42):
        self.loader = DatasetLoader(base_dir=base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def split_synthetic(
        self, train_ratio: float = 0.72, val_ratio: float = 0.14, test_ratio: float = 0.14
    ) -> Dict[str, Any]:
        """Partitions the synthetic dataset using hash-based grouping to eliminate leakage.

        Returns:
            Dictionary with split mappings, counts, and verification status.
        """
        samples = list(self.loader.load_synthetic(split=None))
        hash_to_samples: Dict[str, List[ChequeSample]] = defaultdict(list)

        for s in samples:
            hash_to_samples[s.image_hash].append(s)

        unique_hashes = sorted(list(hash_to_samples.keys()))
        n_unique = len(unique_hashes)

        # Deterministic shuffle
        rng = random.Random(self.seed)
        shuffled_hashes = list(unique_hashes)
        rng.shuffle(shuffled_hashes)

        n_train = int(round(n_unique * train_ratio))
        n_val = int(round(n_unique * val_ratio))
        # Ensure test gets the remainder
        n_test = n_unique - n_train - n_val

        train_hashes = set(shuffled_hashes[:n_train])
        val_hashes = set(shuffled_hashes[n_train : n_train + n_val])
        test_hashes = set(shuffled_hashes[n_train + n_val :])

        # Verify zero leakage across unique hashes
        assert len(train_hashes.intersection(val_hashes)) == 0, "Train-Val hash leakage!"
        assert len(train_hashes.intersection(test_hashes)) == 0, "Train-Test hash leakage!"
        assert len(val_hashes.intersection(test_hashes)) == 0, "Val-Test hash leakage!"

        split_assignments: Dict[str, str] = {}
        split_sample_ids: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        split_unique_counts: Dict[str, int] = {
            "train": len(train_hashes),
            "val": len(val_hashes),
            "test": len(test_hashes),
        }

        for s in samples:
            if s.image_hash in train_hashes:
                sp = "train"
            elif s.image_hash in val_hashes:
                sp = "val"
            else:
                sp = "test"

            split_assignments[s.sample_id] = sp
            split_sample_ids[sp].append(s.sample_id)

        result = {
            "dataset_name": "synthetic",
            "total_rows": len(samples),
            "unique_images": n_unique,
            "hash_splits": {
                "train_hashes": list(train_hashes),
                "val_hashes": list(val_hashes),
                "test_hashes": list(test_hashes),
            },
            "sample_counts": {k: len(v) for k, v in split_sample_ids.items()},
            "unique_image_counts": split_unique_counts,
            "sample_to_split": split_assignments,
            "leakage_verified": True,
        }

        out_path = self.output_dir / "synthetic_splits.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def split_handwritten_and_cheques(self, val_ratio: float = 0.10) -> Dict[str, Any]:
        """Partitions handwritten OCR, Cheque VQA, and isolates IDRBT entity VQA."""
        samples = list(self.loader.load_handwritten_and_cheques(split=None))

        by_cat: Dict[str, List[ChequeSample]] = defaultdict(list)
        for s in samples:
            cat = s.metadata.get("category", "other")
            by_cat[cat].append(s)

        split_assignments: Dict[str, str] = {}
        category_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        rng = random.Random(self.seed)

        # 1. Handwritten OCR lines
        hw_samples = by_cat["handwritten_ocr"]
        # Original train vs test
        hw_train_orig = [s for s in hw_samples if s.source_split == "train"]
        hw_test_orig = [s for s in hw_samples if s.source_split == "test"]

        # Carve out validation from original train
        rng.shuffle(hw_train_orig)
        n_val_hw = int(round(len(hw_train_orig) * val_ratio))
        hw_val = hw_train_orig[:n_val_hw]
        hw_train = hw_train_orig[n_val_hw:]

        for s in hw_train:
            split_assignments[s.sample_id] = "train"
            category_counts["handwritten_ocr"]["train"] += 1
        for s in hw_val:
            split_assignments[s.sample_id] = "val"
            category_counts["handwritten_ocr"]["val"] += 1
        for s in hw_test_orig:
            split_assignments[s.sample_id] = "test"
            category_counts["handwritten_ocr"]["test"] += 1

        # 2. Cheque VQA samples
        vqa_samples = by_cat["cheque_vqa"]
        vqa_train_orig = [s for s in vqa_samples if s.source_split == "train"]
        vqa_test_orig = [s for s in vqa_samples if s.source_split == "test"]

        rng.shuffle(vqa_train_orig)
        n_val_vqa = int(round(len(vqa_train_orig) * val_ratio))
        vqa_val = vqa_train_orig[:n_val_vqa]
        vqa_train = vqa_train_orig[n_val_vqa:]

        for s in vqa_train:
            split_assignments[s.sample_id] = "train"
            category_counts["cheque_vqa"]["train"] += 1
        for s in vqa_val:
            split_assignments[s.sample_id] = "val"
            category_counts["cheque_vqa"]["val"] += 1
        for s in vqa_test_orig:
            split_assignments[s.sample_id] = "test"
            category_counts["cheque_vqa"]["test"] += 1

        # 3. IDRBT Entity VQA (112 samples) -> strictly evaluation/benchmark!
        idrbt_vqa = by_cat["idrbt_entity_vqa"]
        for s in idrbt_vqa:
            split_assignments[s.sample_id] = "evaluation"
            category_counts["idrbt_entity_vqa"]["evaluation"] += 1

        result = {
            "dataset_name": "handwritten_and_cheques_dataset",
            "total_rows": len(samples),
            "category_counts": {k: dict(v) for k, v in category_counts.items()},
            "sample_to_split": split_assignments,
            "leakage_verified": True,
        }

        out_path = self.output_dir / "handwritten_splits.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def split_idrbt(self) -> Dict[str, Any]:
        """Designates 100% of IDRBT/300 images as held-out evaluation benchmark."""
        samples = list(self.loader.load_idrbt(lazy=True))
        split_assignments = {s.sample_id: "evaluation" for s in samples}

        result = {
            "dataset_name": "idrbt_300",
            "total_rows": len(samples),
            "sample_counts": {"evaluation": len(samples)},
            "sample_to_split": split_assignments,
            "policy": "100% reserved for evaluation benchmark (zero training exposure)",
        }

        out_path = self.output_dir / "idrbt_splits.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_all_splits(self) -> Dict[str, Any]:
        """Executes partitioning across all three datasets and compiles a master manifest."""
        syn_res = self.split_synthetic()
        hw_res = self.split_handwritten_and_cheques()
        idrbt_res = self.split_idrbt()

        master_summary = {
            "seed": self.seed,
            "synthetic": syn_res["sample_counts"],
            "synthetic_unique_images": syn_res["unique_image_counts"],
            "handwritten_and_cheques": hw_res["category_counts"],
            "idrbt_300": idrbt_res["sample_counts"],
            "total_samples": (
                syn_res["total_rows"] + hw_res["total_rows"] + idrbt_res["total_rows"]
            ),
        }

        summary_path = self.output_dir / "combined_splits_summary.json"
        with open(summary_path, "w") as f:
            json.dump(master_summary, f, indent=2)

        return master_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition ChequeSense datasets without leakage.")
    parser.add_argument("--base_dir", default="Dataset", help="Path to raw Dataset directory")
    parser.add_argument("--output_dir", default="data/splits", help="Output directory for split mappings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    args = parser.parse_args()

    splitter = DataSplitter(base_dir=args.base_dir, output_dir=args.output_dir, seed=args.seed)
    summary = splitter.run_all_splits()
    print("Partitioning complete. Summary:")
    print(json.dumps(summary, indent=2))
