"""Evaluation utilities and metrics for ChequeSense character/digit recognition."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.recognition.dataset import ChequeDigitDataset, DIGIT_CLASSES, IDX_TO_CLASS, NUM_CLASSES
from src.recognition.model import ChequeDigitCNN


@dataclass
class DigitClassMetric:
    """Per-class classification metrics."""

    digit: str
    class_id: int
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "digit": self.digit,
            "class_id": self.class_id,
            "support": self.support,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
        }


@dataclass
class RecognitionResult:
    """Comprehensive recognition performance evaluation output."""

    total_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: List[List[int]]
    per_class: Dict[str, DigitClassMetric] = field(default_factory=dict)
    confusing_pairs: List[Dict[str, Any]] = field(default_factory=list)
    low_confidence_cases: List[Dict[str, Any]] = field(default_factory=list)
    incorrect_cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "confusion_matrix": self.confusion_matrix,
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
            "confusing_pairs": self.confusing_pairs[:10],
            "low_confidence_count": len(self.low_confidence_cases),
            "incorrect_count": len(self.incorrect_cases),
        }


class RecognitionEvaluator:
    """Evaluates character/digit recognition models with full diagnostic analysis."""

    def __init__(self, low_confidence_threshold: float = 0.70):
        self.low_conf_threshold = low_confidence_threshold

    def evaluate(
        self,
        y_true: List[int],
        y_pred: List[int],
        confidences: List[float],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> RecognitionResult:
        """Computes complete multi-class metrics, confusion matrix, and error cases."""
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        confs_arr = np.array(confidences)
        n_samples = len(y_true)

        # 1. Confusion Matrix [NUM_CLASSES x NUM_CLASSES]
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
                cm[t, p] += 1

        # 2. Overall Accuracy
        accuracy = float(np.sum(y_true_arr == y_pred_arr) / max(1, n_samples))

        # 3. Per-class metrics
        per_class: Dict[str, DigitClassMetric] = {}
        precisions = []
        recalls = []
        f1s = []
        supports = []

        for cid in range(NUM_CLASSES):
            cname = IDX_TO_CLASS[cid]
            tp = int(cm[cid, cid])
            fp = int(np.sum(cm[:, cid]) - tp)
            fn = int(np.sum(cm[cid, :]) - tp)
            support = int(np.sum(cm[cid, :]))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class[cname] = DigitClassMetric(
                digit=cname,
                class_id=cid,
                support=support,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=prec,
                recall=rec,
                f1_score=f1,
            )

            if support > 0:
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)
                supports.append(support)

        macro_p = float(np.mean(precisions)) if precisions else 0.0
        macro_r = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        total_supp = sum(supports)
        weighted_f1 = (
            float(sum(f * s for f, s in zip(f1s, supports)) / total_supp)
            if total_supp > 0
            else 0.0
        )

        # 4. Identify Confusing Pairs (off-diagonal CM entries)
        confusing_pairs = []
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if i != j and cm[i, j] > 0:
                    confusing_pairs.append({
                        "true_digit": IDX_TO_CLASS[i],
                        "pred_digit": IDX_TO_CLASS[j],
                        "error_count": int(cm[i, j]),
                    })
        confusing_pairs.sort(key=lambda x: x["error_count"], reverse=True)

        # 5. Collect Error Cases and Low-Confidence Cases
        incorrect_cases = []
        low_confidence_cases = []

        for idx in range(n_samples):
            t = y_true[idx]
            p = y_pred[idx]
            conf = confidences[idx]
            meta = metadata[idx] if metadata else {}

            case_info = {
                "sample_index": idx,
                "true_digit": IDX_TO_CLASS[t],
                "pred_digit": IDX_TO_CLASS[p],
                "confidence": round(conf, 4),
                "is_correct": bool(t == p),
                **meta,
            }

            if t != p:
                incorrect_cases.append(case_info)
            if conf < self.low_conf_threshold:
                low_confidence_cases.append(case_info)

        return RecognitionResult(
            total_samples=n_samples,
            accuracy=accuracy,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            confusion_matrix=cm.tolist(),
            per_class=per_class,
            confusing_pairs=confusing_pairs,
            low_confidence_cases=low_confidence_cases,
            incorrect_cases=incorrect_cases,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ChequeSense digit recognition model.")
    parser.add_argument("--model_path", default="models/recognizer/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--manifest_path", default="data/manifests/test_manifest.json", help="Test manifest")
    parser.add_argument("--output_path", default="artifacts/recognition/test_metrics.json", help="Output path")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Loading recognition model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    model = ChequeDigitCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_dataset = ChequeDigitDataset(manifest_path=args.manifest_path, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Evaluating {len(test_dataset)} digit glyphs from {args.manifest_path}...")

    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []
    metas: List[Dict[str, Any]] = []

    with torch.no_grad():
        for tensors, labels, meta_batch in test_loader:
            tensors = tensors.to(device)
            preds, confs = model.predict_glyph(tensors)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
            confidences.extend(confs.cpu().tolist())

            # Convert meta batch
            batch_len = len(labels)
            for i in range(batch_len):
                metas.append({k: meta_batch[k][i] for k in meta_batch})

    evaluator = RecognitionEvaluator()
    res = evaluator.evaluate(y_true, y_pred, confidences, metas)

    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(res.to_dict(), f, indent=2)

    # Save raw confusion matrix
    cm_file = out_file.parent / "confusion_matrix.json"
    with open(cm_file, "w") as f:
        json.dump({
            "labels": DIGIT_CLASSES,
            "matrix": res.confusion_matrix,
        }, f, indent=2)

    # Save comprehensive error analysis
    err_file = out_file.parent / "error_analysis.json"
    with open(err_file, "w") as f:
        json.dump({
            "total_tested": res.total_samples,
            "total_incorrect": len(res.incorrect_cases),
            "total_low_confidence": len(res.low_confidence_cases),
            "confusing_pairs": res.confusing_pairs,
            "incorrect_cases": res.incorrect_cases,
            "low_confidence_cases": res.low_confidence_cases,
        }, f, indent=2)

    print("=" * 60)
    print("RECOGNITION TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Samples Tested: {res.total_samples}")
    print(f"Accuracy:             {res.accuracy * 100:.2f}%")
    print(f"Macro Precision:      {res.macro_precision:.4f}")
    print(f"Macro Recall:         {res.macro_recall:.4f}")
    print(f"Macro F1-Score:       {res.macro_f1:.4f}")
    print(f"Weighted F1-Score:    {res.weighted_f1:.4f}")
    print("-" * 60)
    print(f"{'Digit':<8} {'Support':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}")
    print("-" * 60)
    for cname, m in res.per_class.items():
        print(f"{cname:<8} {m.support:<10} {m.precision:<12.4f} {m.recall:<12.4f} {m.f1_score:<10.4f}")
    print("=" * 60)
    print(f"Top Confusing Digit Pairs: {res.confusing_pairs[:5]}")
    print(f"Metrics saved to: {out_file}")
