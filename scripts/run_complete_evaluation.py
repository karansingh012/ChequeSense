"""Comprehensive ML and System Evaluation Script for ChequeSense (Phase 12).

Evaluates separately:
1. Field detection (precision, recall, F1, AP@50, mAP@[50:95] overall and per class)
2. Handwritten recognition (accuracy, precision, recall, F1, 10x10 confusion matrix, error pairs)
3. OCR & text processing (character accuracy, CER, WER, digit normalization)
4. Field extraction (field-level presence, exact-match accuracy, character-level edit distance)
5. Validation layer (compliance pass rates, confidence tier distribution, review queue routing)
6. End-to-end pipeline & System performance (average latency per stage, peak memory RSS, failure rate)
7. Error analysis & failure mode taxonomy (handwriting, image quality, detection, OCR, low-confidence)

Outputs:
- reports/final_metrics.json
- reports/figures/*.png
- reports/error_analysis_data.json
"""

from __future__ import annotations

import gc
import json
import logging
import os
import resource
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset_loader import DatasetLoader
from src.detection.dataset import ChequeDetectionDataset, NUM_CLASSES, get_detection_collate_fn
from src.detection.evaluate import DetectionEvaluator
from src.detection.train import create_detector_model
from src.pipeline.pipeline import ChequeInferencePipeline
from src.recognition.dataset import ChequeDigitDataset
from src.recognition.evaluate import RecognitionEvaluator
from src.recognition.model import ChequeDigitCNN
from src.validation.field_validator import FieldValidator
from src.validation.review_queue import ReviewQueueManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("chequesense.evaluation")


def calculate_levenshtein_similarity(s1: str, s2: str) -> float:
    """Computes normalized character-level Levenshtein similarity in [0, 1]."""
    s1, s2 = str(s1 or "").strip().lower(), str(s2 or "").strip().lower()
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    dist = dp[len1][len2]
    max_len = max(len1, len2)
    return max(0.0, 1.0 - (dist / max_len))


def evaluate_field_detection(device: torch.device) -> Dict[str, Any]:
    """Evaluates Faster R-CNN field detector on held-out synthetic test set."""
    logger.info("Evaluating Field Detection Module...")
    model_path = Path("models/field_detector/best_model.pt")
    manifest_path = Path("data/manifests/test_manifest.json")

    checkpoint = torch.load(model_path, map_location=device)
    target_size = checkpoint.get("target_size", (1000, 460))

    model = create_detector_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = ChequeDetectionDataset(
        manifest_path=manifest_path,
        target_size=target_size,
        augment=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=get_detection_collate_fn(),
    )

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

    evaluator = DetectionEvaluator(score_thresh=0.35)
    det_res = evaluator.evaluate(all_preds, all_targets)
    return det_res.to_dict()


def evaluate_handwritten_recognition(device: torch.device) -> Dict[str, Any]:
    """Evaluates ChequeDigitCNN on held-out test digit crops."""
    logger.info("Evaluating Handwritten Recognition Module...")
    model_path = Path("models/recognizer/best_model.pt")
    manifest_path = Path("data/manifests/test_manifest.json")

    checkpoint = torch.load(model_path, map_location=device)
    model = ChequeDigitCNN(num_classes=10, in_channels=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_dataset = ChequeDigitDataset(
        manifest_path=manifest_path,
        target_size=(32, 32),
        augment=False,
    )
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []

    with torch.no_grad():
        for tensors, labels, _ in dataloader:
            tensors = tensors.to(device)
            outputs = model(tensors)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            confidences.extend(confs.cpu().tolist())

    evaluator = RecognitionEvaluator(low_confidence_threshold=0.70)
    rec_res = evaluator.evaluate(y_true, y_pred, confidences)
    return rec_res.to_dict()


def evaluate_pipeline_and_systems(
    pipeline: ChequeInferencePipeline,
    sample_cache: Dict[str, Any],
    all_test_records: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Single-pass integrated evaluation across test cheques for extraction, system, and validation."""
    logger.info("Executing Pipeline and System Evaluation across %d test cheques...", len(all_test_records))
    review_queue = ReviewQueueManager()

    latencies_overall = []
    latencies_prep = []
    latencies_det = []
    latencies_ext = []

    validation_statuses = Counter()
    review_reasons = Counter()
    confidence_tiers = Counter()

    field_presence_counts = defaultdict(int)
    field_exact_match_counts = defaultdict(int)
    field_similarity_sums = defaultdict(float)
    field_total_counts = defaultdict(int)

    ocr_token_eval = {"total_tokens": 0, "correct_tokens": 0, "char_errors": 0, "total_chars": 0}

    # Error analysis buckets
    error_cases = []
    failure_mode_counts = Counter({
        "handwriting_recognition_error": 0,
        "poor_image_quality_low_resolution": 0,
        "field_detection_miss": 0,
        "ocr_background_interference": 0,
        "low_confidence_review_trigger": 0,
    })

    failures = 0
    total_samples = len(all_test_records)

    # Start memory tracing
    gc.collect()
    mem_before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()

    for idx, rec in enumerate(all_test_records):
        sid = rec["sample_id"]
        sample = sample_cache.get(sid)
        if not sample:
            continue

        try:
            img = sample.get_image()
            t0 = time.perf_counter()
            result = pipeline.process(img, cheque_id=sid)
            total_elapsed = (time.perf_counter() - t0) * 1000.0
            latencies_overall.append(total_elapsed)

            diag = result.diagnostics
            if diag and diag.stage_latencies_ms:
                latencies_prep.append(diag.stage_latencies_ms.get("preprocessing_ms", 0.0))
                latencies_det.append(diag.stage_latencies_ms.get("detection_ms", 0.0))
                latencies_ext.append(diag.stage_latencies_ms.get("extraction_ms", 0.0))

            # Run compliance review gating
            review_item = review_queue.evaluate_cheque(result)
            validation_statuses[review_item.status] += 1
            for r in review_item.reasons_for_review:
                review_reasons[r] += 1

            for fname, fval in result.fields.items():
                confidence_tiers[fval.confidence_tier] += 1

            # Check ground truth for field extraction (VQA records)
            gt_list = rec.get("annotations", {}).get("ground_truth", [])
            gt_map: Dict[str, str] = {}
            for item in gt_list:
                if isinstance(item, dict):
                    gt_map.update(item)

            if gt_map:
                field_eval_pairs = [
                    ("amount", gt_map.get("amt_in_figures")),
                    ("date", gt_map.get("cheque_date")),
                    ("name", gt_map.get("payee_name")),
                ]

                for field_key, gt_val in field_eval_pairs:
                    if not gt_val:
                        continue

                    field_total_counts[field_key] += 1
                    pred_field = result.fields.get(field_key)
                    pred_val = pred_field.value if pred_field else ""

                    if pred_val and pred_val.strip():
                        field_presence_counts[field_key] += 1

                    clean_gt = "".join(c for c in str(gt_val).lower() if c.isalnum() or c in "./-")
                    clean_pred = "".join(c for c in str(pred_val).lower() if c.isalnum() or c in "./-")

                    exact_match = (clean_gt == clean_pred) and bool(clean_gt)
                    if exact_match:
                        field_exact_match_counts[field_key] += 1

                    sim = calculate_levenshtein_similarity(clean_gt, clean_pred)
                    field_similarity_sums[field_key] += sim

                    ocr_token_eval["total_chars"] += max(len(clean_gt), 1)
                    char_diff = abs(len(clean_gt) - len(clean_pred)) + sum(1 for c1, c2 in zip(clean_gt, clean_pred) if c1 != c2)
                    ocr_token_eval["char_errors"] += char_diff
                    if exact_match:
                        ocr_token_eval["correct_tokens"] += 1
                    ocr_token_eval["total_tokens"] += 1

                    if not exact_match:
                        # Categorize error
                        if img.width <= 600 or img.height <= 300:
                            failure_mode_counts["poor_image_quality_low_resolution"] += 1
                            cat = "poor_image_quality_low_resolution"
                        elif pred_field and pred_field.method == "recognizer":
                            failure_mode_counts["handwriting_recognition_error"] += 1
                            cat = "handwriting_recognition_error"
                        elif not pred_val:
                            failure_mode_counts["field_detection_miss"] += 1
                            cat = "field_detection_miss"
                        else:
                            failure_mode_counts["ocr_background_interference"] += 1
                            cat = "ocr_background_interference"

                        if pred_field and pred_field.confidence < 0.60:
                            failure_mode_counts["low_confidence_review_trigger"] += 1

                        error_cases.append({
                            "sample_id": sid,
                            "field": field_key,
                            "ground_truth": str(gt_val),
                            "predicted": pred_val,
                            "confidence": round(pred_field.confidence, 4) if pred_field else 0.0,
                            "method": pred_field.method if pred_field else "none",
                            "similarity": round(sim, 4),
                            "resolution": f"{img.width}x{img.height}",
                            "failure_category": cat,
                        })

        except Exception as e:
            failures += 1
            logger.error("Unhandled pipeline exception on sample %s: %s", sid, e)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # macOS ru_maxrss is bytes, Linux is KB
    rss_mb = mem_after_rss / (1024 * 1024) if mem_after_rss > 10000000 else mem_after_rss / 1024
    peak_trace_mb = peak_mem / (1024 * 1024)

    # Compile Extraction Metrics
    extraction_metrics = {}
    for fkey, tot in field_total_counts.items():
        extraction_metrics[fkey] = {
            "total_evaluated": tot,
            "presence_rate": round(field_presence_counts[fkey] / max(1, tot), 4),
            "exact_match_accuracy": round(field_exact_match_counts[fkey] / max(1, tot), 4),
            "character_accuracy": round(field_similarity_sums[fkey] / max(1, tot), 4),
        }

    # Compile OCR Metrics
    tot_tokens = max(1, ocr_token_eval["total_tokens"])
    tot_chars = max(1, ocr_token_eval["total_chars"])
    ocr_metrics = {
        "word_exact_match_accuracy": round(ocr_token_eval["correct_tokens"] / tot_tokens, 4),
        "word_error_rate_wer": round(1.0 - (ocr_token_eval["correct_tokens"] / tot_tokens), 4),
        "character_error_rate_cer": round(min(1.0, ocr_token_eval["char_errors"] / tot_chars), 4),
        "character_level_accuracy": round(max(0.0, 1.0 - (ocr_token_eval["char_errors"] / tot_chars)), 4),
    }

    # Compile System Metrics
    system_metrics = {
        "total_processed": total_samples,
        "successful_runs": total_samples - failures,
        "failed_runs": failures,
        "failure_rate": round(failures / max(1, total_samples), 4),
        "latency_ms": {
            "mean_total": round(float(np.mean(latencies_overall)), 2) if latencies_overall else 0.0,
            "median_total": round(float(np.median(latencies_overall)), 2) if latencies_overall else 0.0,
            "p95_total": round(float(np.percentile(latencies_overall, 95)), 2) if latencies_overall else 0.0,
            "preprocessing_mean": round(float(np.mean(latencies_prep)), 2) if latencies_prep else 0.0,
            "detection_mean": round(float(np.mean(latencies_det)), 2) if latencies_det else 0.0,
            "extraction_mean": round(float(np.mean(latencies_ext)), 2) if latencies_ext else 0.0,
        },
        "memory_mb": {
            "peak_rss_mb": round(rss_mb, 2),
            "peak_traced_heap_mb": round(peak_trace_mb, 2),
        },
        "failure_modes": dict(failure_mode_counts),
    }

    # Compile Validation Metrics
    validation_metrics = {
        "status_distribution": {k: v for k, v in validation_statuses.items()},
        "verification_rate": round(validation_statuses.get("VERIFIED", 0) / max(1, total_samples), 4),
        "review_required_rate": round(validation_statuses.get("REVIEW_REQUIRED", 0) / max(1, total_samples), 4),
        "invalid_rate": round(validation_statuses.get("INVALID", 0) / max(1, total_samples), 4),
        "error_state_rate": round(validation_statuses.get("ERROR", 0) / max(1, total_samples), 4),
        "confidence_tier_counts": dict(confidence_tiers),
        "top_review_reasons": review_reasons.most_common(8),
    }

    return ocr_metrics, extraction_metrics, system_metrics, validation_metrics, error_cases


def generate_evaluation_visualizations(
    det_metrics: Dict[str, Any],
    rec_metrics: Dict[str, Any],
    ext_metrics: Dict[str, Any],
    sys_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    figures_dir: Path,
):
    """Generates publication-quality charts for the evaluation report."""
    logger.info("Generating evaluation charts in %s...", figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Field Detection AP Chart
    plt.figure(figsize=(9, 5))
    classes = list(det_metrics["per_class"].keys())
    ap50 = [det_metrics["per_class"][c]["ap_50"] for c in classes]
    ap50_95 = [det_metrics["per_class"][c]["ap_50_95"] for c in classes]

    x = np.arange(len(classes))
    width = 0.35
    plt.bar(x - width/2, ap50, width, label="AP@50", color="#1f77b4")
    plt.bar(x + width/2, ap50_95, width, label="AP@[50:95]", color="#ff7f0e")
    plt.xlabel("Cheque Field Category", fontsize=11, fontweight="bold")
    plt.ylabel("Average Precision (AP)", fontsize=11, fontweight="bold")
    plt.title("Faster R-CNN Field Detection Performance (Held-out Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, [c.upper() for c in classes], fontsize=10)
    plt.ylim(0.0, 1.1)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "detection_metrics.png", dpi=300)
    plt.close()

    # 2. Recognition Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6.5))
    cm = np.array(rec_metrics["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)],
        cbar=True,
    )
    plt.title("Handwritten Digit Recognition Confusion Matrix (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Digit", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Digit", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "recognition_confusion_matrix.png", dpi=300)
    plt.close()

    # 3. Field Extraction Accuracy Chart
    plt.figure(figsize=(8, 4.8))
    ext_fields = list(ext_metrics.keys())
    exact_accs = [ext_metrics[f]["exact_match_accuracy"] for f in ext_fields]
    char_accs = [ext_metrics[f]["character_accuracy"] for f in ext_fields]

    x = np.arange(len(ext_fields))
    width = 0.35
    plt.bar(x - width/2, exact_accs, width, label="Exact Match Accuracy", color="#2ca02c")
    plt.bar(x + width/2, char_accs, width, label="Character Levenshtein Similarity", color="#17becf")
    plt.xlabel("Extracted Cheque Field", fontsize=11, fontweight="bold")
    plt.ylabel("Metric Score [0.0 - 1.0]", fontsize=11, fontweight="bold")
    plt.title("Field Extraction Accuracy across Test Cheques", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, [f.upper() for f in ext_fields], fontsize=10)
    plt.ylim(0.0, 1.1)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "field_extraction_accuracy.png", dpi=300)
    plt.close()

    # 4. Latency Breakdown Chart
    plt.figure(figsize=(8, 4.5))
    stages = [
        "1. Preprocessing",
        "2. Field Detection",
        "3. Extraction & OCR",
    ]
    lat_values = [
        sys_metrics["latency_ms"]["preprocessing_mean"],
        sys_metrics["latency_ms"]["detection_mean"],
        sys_metrics["latency_ms"]["extraction_mean"],
    ]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
    bars = plt.bar(stages, lat_values, color=colors, width=0.50)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.1f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.title("Pipeline Stage Mean Latency Breakdown", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Mean Latency (ms)", fontsize=11, fontweight="bold")
    plt.ylim(0, max(lat_values) * 1.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "latency_waterfall.png", dpi=300)
    plt.close()

    # 5. Review Queue Routing Distribution
    plt.figure(figsize=(6, 6))
    statuses = list(val_metrics["status_distribution"].keys())
    counts = [val_metrics["status_distribution"][s] for s in statuses]
    palette = {"VERIFIED": "#2ca02c", "REVIEW_REQUIRED": "#ff7f0e", "INVALID": "#d62728", "PARTIAL": "#9467bd", "ERROR": "#7f7f7f"}
    c_list = [palette.get(s, "#bcbd22") for s in statuses]
    plt.pie(
        counts,
        labels=statuses,
        autopct="%1.1f%%",
        startangle=140,
        colors=c_list,
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    plt.title("Compliance Review Routing Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "review_queue_routing.png", dpi=300)
    plt.close()

    # 6. Failure Modes Taxonomy Bar Chart
    plt.figure(figsize=(9, 4.5))
    f_modes = sys_metrics.get("failure_modes", {})
    labels = [
        "Low Resolution",
        "Handwriting Ambiguity",
        "Field Detection Miss",
        "OCR Mesh Interference",
        "Low Confidence Trigger",
    ]
    counts = [
        f_modes.get("poor_image_quality_low_resolution", 0),
        f_modes.get("handwriting_recognition_error", 0),
        f_modes.get("field_detection_miss", 0),
        f_modes.get("ocr_background_interference", 0),
        f_modes.get("low_confidence_review_trigger", 0),
    ]
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, counts, color="#d62728", alpha=0.85, height=0.55)
    plt.yticks(y_pos, labels, fontsize=10, fontweight="bold")
    plt.xlabel("Occurrences in Evaluation Set", fontsize=11, fontweight="bold")
    plt.title("Error Analysis: Taxonomy of Failure Modes", fontsize=13, fontweight="bold", pad=12)
    for i, v in enumerate(counts):
        plt.text(v + 1, i, str(v), va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_distribution.png", dpi=300)
    plt.close()


def main():
    logger.info("Initializing ChequeSense System Evaluation Benchmark...")
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # 1. Field Detection Evaluation
    det_metrics = evaluate_field_detection(device)

    # 2. Handwritten Recognition Evaluation
    rec_metrics = evaluate_handwritten_recognition(device)

    # 3. Load Datasets for Pipeline & Extraction Evaluation
    loader = DatasetLoader(base_dir="Dataset")
    with open("data/manifests/test_manifest.json") as f:
        manifest_records = json.load(f)

    # Cache synthetic and handwritten test samples
    syn_cache = {s.sample_id: s for s in loader.load_synthetic(split=None)}
    hw_cache = {s.sample_id: s for s in loader.load_handwritten_and_cheques(split=None)}
    combined_cache = {**syn_cache, **hw_cache}

    all_test_records = [
        r for r in manifest_records
        if r.get("dataset_name") in ("synthetic", "cheque_vqa") and r.get("sample_id") in combined_cache
    ]

    pipeline = ChequeInferencePipeline()

    # 4. Integrated Pipeline, Extraction, OCR, and Validation Evaluation
    ocr_metrics, ext_metrics, sys_metrics, val_metrics, error_cases = evaluate_pipeline_and_systems(
        pipeline=pipeline,
        sample_cache=combined_cache,
        all_test_records=all_test_records,
    )

    # 5. Generate Figures
    generate_evaluation_visualizations(
        det_metrics=det_metrics,
        rec_metrics=rec_metrics,
        ext_metrics=ext_metrics,
        sys_metrics=sys_metrics,
        val_metrics=val_metrics,
        figures_dir=figures_dir,
    )

    # 6. Consolidate Final Metrics JSON
    final_metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_dataset_sizes": {
            "synthetic_detection_test_samples": det_metrics["total_images"],
            "handwritten_recognition_test_samples": rec_metrics["total_samples"],
            "pipeline_and_system_test_samples": len(all_test_records),
        },
        "field_detection": det_metrics,
        "handwritten_recognition": rec_metrics,
        "ocr_performance": ocr_metrics,
        "field_extraction": ext_metrics,
        "system_performance": sys_metrics,
        "validation_and_review": val_metrics,
    }

    metrics_file = reports_dir / "final_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info("Saved final metrics to %s", metrics_file)

    # Save detailed error cases for deep qualitative analysis
    error_dump_file = reports_dir / "error_analysis_data.json"
    with open(error_dump_file, "w") as f:
        json.dump({
            "extraction_errors_sample": error_cases[:60],
            "failure_mode_counts": sys_metrics.get("failure_modes", {}),
            "confusing_digit_pairs": rec_metrics["confusing_pairs"],
        }, f, indent=2)
    logger.info("Saved raw error data to %s", error_dump_file)


if __name__ == "__main__":
    main()
