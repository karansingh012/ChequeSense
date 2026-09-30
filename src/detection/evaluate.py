"""Evaluation metrics and benchmarking for ChequeSense field detection.

Computes Precision, Recall, F1, AP@50, and mAP@[50:95] overall and per field class
(date, amount, ifsc, acno, sign, name).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.detection.dataset import FIELD_CLASSES, IDX_TO_CLASS, NUM_CLASSES


@dataclass
class ClassMetric:
    """Performance metrics for a single field class."""

    class_name: str
    class_id: int
    num_ground_truths: int = 0
    num_predictions: int = 0
    true_positives_50: int = 0
    false_positives_50: int = 0
    false_negatives_50: int = 0
    precision_50: float = 0.0
    recall_50: float = 0.0
    f1_50: float = 0.0
    ap_50: float = 0.0
    ap_50_95: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "gt_count": self.num_ground_truths,
            "pred_count": self.num_predictions,
            "precision_50": round(self.precision_50, 4),
            "recall_50": round(self.recall_50, 4),
            "f1_50": round(self.f1_50, 4),
            "ap_50": round(self.ap_50, 4),
            "ap_50_95": round(self.ap_50_95, 4),
        }


@dataclass
class EvaluationResult:
    """Comprehensive evaluation metrics across all classes."""

    mean_precision_50: float = 0.0
    mean_recall_50: float = 0.0
    mean_f1_50: float = 0.0
    map_50: float = 0.0
    map_50_95: float = 0.0
    total_images: int = 0
    total_ground_truths: int = 0
    total_predictions: int = 0
    per_class: Dict[str, ClassMetric] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_images": self.total_images,
            "total_ground_truths": self.total_ground_truths,
            "total_predictions": self.total_predictions,
            "mean_precision_50": round(self.mean_precision_50, 4),
            "mean_recall_50": round(self.mean_recall_50, 4),
            "mean_f1_50": round(self.mean_f1_50, 4),
            "map_50": round(self.map_50, 4),
            "map_50_95": round(self.map_50_95, 4),
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
        }


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise Intersection-over-Union (IoU) between two sets of boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Computes Average Precision (AP) using standard 101-point COCO interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute maximum precision for any recall >= current recall
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 101 recall points from 0 to 1.0
    recall_points = np.linspace(0.0, 1.0, 101)
    ap = np.trapz(mpre, mrec)
    return float(np.clip(ap, 0.0, 1.0))


class DetectionEvaluator:
    """Evaluates predicted bounding boxes against ground-truth targets."""

    def __init__(self, iou_thresholds: Optional[List[float]] = None, score_thresh: float = 0.3):
        self.iou_thresholds = iou_thresholds or list(np.arange(0.50, 1.0, 0.05))
        self.score_thresh = score_thresh

    def evaluate(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> EvaluationResult:
        """Evaluates batch of predictions against targets."""
        assert len(predictions) == len(targets), "Predictions and targets length mismatch!"

        # Group detections and GT by class
        # class_id -> list of predictions across all images: (image_idx, score, box)
        preds_by_class: Dict[int, List[Tuple[int, float, np.ndarray]]] = {c: [] for c in range(1, NUM_CLASSES)}
        # class_id -> dict of image_idx -> list of gt_boxes
        gts_by_class: Dict[int, Dict[int, List[np.ndarray]]] = {c: {i: [] for i in range(len(targets))} for c in range(1, NUM_CLASSES)}

        total_gt = 0
        total_pred = 0

        for img_idx, (pred, tgt) in enumerate(zip(predictions, targets)):
            # Target boxes
            tgt_boxes = tgt["boxes"].detach().cpu().numpy()
            tgt_labels = tgt["labels"].detach().cpu().numpy()
            for b, l in zip(tgt_boxes, tgt_labels):
                if int(l) in gts_by_class:
                    gts_by_class[int(l)][img_idx].append(b)
                    total_gt += 1

            # Predicted boxes
            p_boxes = pred["boxes"].detach().cpu().numpy()
            p_scores = pred["scores"].detach().cpu().numpy()
            p_labels = pred["labels"].detach().cpu().numpy()

            for b, s, l in zip(p_boxes, p_scores, p_labels):
                if s >= self.score_thresh and int(l) in preds_by_class:
                    preds_by_class[int(l)].append((img_idx, float(s), b))
                    total_pred += 1

        result = EvaluationResult(
            total_images=len(targets),
            total_ground_truths=total_gt,
            total_predictions=total_pred,
        )

        ap_50_list = []
        ap_50_95_list = []
        p_50_list = []
        r_50_list = []
        f1_50_list = []

        for cid in range(1, NUM_CLASSES):
            cname = IDX_TO_CLASS[cid]
            class_preds = preds_by_class[cid]
            class_gts = gts_by_class[cid]

            n_gt = sum(len(boxes) for boxes in class_gts.values())
            n_pred = len(class_preds)

            metric = ClassMetric(
                class_name=cname,
                class_id=cid,
                num_ground_truths=n_gt,
                num_predictions=n_pred,
            )

            if n_gt == 0:
                result.per_class[cname] = metric
                continue

            # Sort predictions by score descending
            class_preds.sort(key=lambda x: x[1], reverse=True)

            # Evaluate across all IoU thresholds
            aps_across_ious = []
            for iou_thresh in self.iou_thresholds:
                tp = np.zeros(n_pred)
                fp = np.zeros(n_pred)
                detected_gts = {img_i: [False] * len(boxes) for img_i, boxes in class_gts.items()}

                for p_idx, (img_i, score, p_box) in enumerate(class_preds):
                    img_gt_boxes = class_gts[img_i]
                    if len(img_gt_boxes) == 0:
                        fp[p_idx] = 1.0
                        continue

                    ious = box_iou(np.array([p_box]), np.array(img_gt_boxes))[0]
                    best_match_idx = int(np.argmax(ious))
                    best_iou = ious[best_match_idx]

                    if best_iou >= iou_thresh and not detected_gts[img_i][best_match_idx]:
                        tp[p_idx] = 1.0
                        detected_gts[img_i][best_match_idx] = True
                    else:
                        fp[p_idx] = 1.0

                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(fp)
                recalls = cum_tp / n_gt
                precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

                ap = compute_ap(recalls, precisions)
                aps_across_ious.append(ap)

                # Store metrics at IoU=0.50
                if abs(iou_thresh - 0.50) < 1e-4:
                    n_tp = int(np.sum(tp))
                    n_fp = int(np.sum(fp))
                    n_fn = n_gt - n_tp
                    p_50 = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
                    r_50 = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
                    f1_50 = 2 * p_50 * r_50 / (p_50 + r_50) if (p_50 + r_50) > 0 else 0.0

                    metric.true_positives_50 = n_tp
                    metric.false_positives_50 = n_fp
                    metric.false_negatives_50 = n_fn
                    metric.precision_50 = p_50
                    metric.recall_50 = r_50
                    metric.f1_50 = f1_50
                    metric.ap_50 = ap

            metric.ap_50_95 = float(np.mean(aps_across_ious)) if aps_across_ious else 0.0
            result.per_class[cname] = metric

            ap_50_list.append(metric.ap_50)
            ap_50_95_list.append(metric.ap_50_95)
            p_50_list.append(metric.precision_50)
            r_50_list.append(metric.recall_50)
            f1_50_list.append(metric.f1_50)

        # Macro averages
        result.map_50 = float(np.mean(ap_50_list)) if ap_50_list else 0.0
        result.map_50_95 = float(np.mean(ap_50_95_list)) if ap_50_95_list else 0.0
        result.mean_precision_50 = float(np.mean(p_50_list)) if p_50_list else 0.0
        result.mean_recall_50 = float(np.mean(r_50_list)) if r_50_list else 0.0
        result.mean_f1_50 = float(np.mean(f1_50_list)) if f1_50_list else 0.0

        return result


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from torch.utils.data import DataLoader
    from src.detection.dataset import ChequeDetectionDataset, NUM_CLASSES, get_detection_collate_fn
    from src.detection.train import create_detector_model

    parser = argparse.ArgumentParser(description="Evaluate ChequeSense field detector.")
    parser.add_argument("--model_path", default="models/field_detector/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--manifest_path", default="data/manifests/test_manifest.json", help="Manifest to evaluate")
    parser.add_argument("--output_path", default="artifacts/field_detection/test_metrics.json", help="Output path")
    parser.add_argument("--score_thresh", type=float, default=0.35, help="Detection score threshold")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Loading model checkpoint from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    target_size = checkpoint.get("target_size", (1000, 460))

    model = create_detector_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = ChequeDetectionDataset(
        manifest_path=args.manifest_path,
        target_size=target_size,
        augment=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=get_detection_collate_fn(),
    )

    print(f"Evaluating {len(dataset)} samples from {args.manifest_path}...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [im.to(device) for im in images]
            preds = model(images)
            for p in preds:
                all_preds.append({k: v.cpu() for k, v in p.items()})
            for t in targets:
                all_targets.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()})

    evaluator = DetectionEvaluator(score_thresh=args.score_thresh)
    res = evaluator.evaluate(all_preds, all_targets)

    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(res.to_dict(), f, indent=2)

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Images: {res.total_images}")
    print(f"mAP@50:       {res.map_50:.4f}")
    print(f"mAP@50-95:    {res.map_50_95:.4f}")
    print(f"Mean Prec@50: {res.mean_precision_50:.4f}")
    print(f"Mean Rec@50:  {res.mean_recall_50:.4f}")
    print(f"Mean F1@50:   {res.mean_f1_50:.4f}")
    print("-" * 60)
    print(f"{'Class':<12} {'GT':<6} {'Pred':<6} {'Prec@50':<10} {'Rec@50':<10} {'F1@50':<10} {'AP@50':<10} {'AP@50-95':<10}")
    print("-" * 60)
    for cname, m in res.per_class.items():
        print(f"{cname:<12} {m.num_ground_truths:<6} {m.num_predictions:<6} {m.precision_50:<10.4f} {m.recall_50:<10.4f} {m.f1_50:<10.4f} {m.ap_50:<10.4f} {m.ap_50_95:<10.4f}")
    print("=" * 60)
    print(f"Metrics written to: {out_file}")

