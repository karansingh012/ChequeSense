"""Reporting utilities and export generators for ChequeSense banking analytics."""

from __future__ import annotations

import csv
import datetime
import io
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.analytics.metrics import (
    ExecutiveSummary,
    calculate_executive_dashboard,
)
from src.database.models import Cheque, ExtractedField, ProcessingRun


def generate_executive_report_markdown(summary: ExecutiveSummary) -> str:
    """Renders a comprehensive, executive-ready Markdown briefing for bank operations."""
    gen_time = summary.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    p_start = summary.period_start.strftime("%Y-%m-%d") if summary.period_start else "All Time"
    p_end = summary.period_end.strftime("%Y-%m-%d") if summary.period_end else "Present"

    md = []
    md.append("# ChequeSense Banking Analytics & Operations Report")
    md.append(f"**Generated:** {gen_time} | **Period:** {p_start} to {p_end}\n")

    # 1. Executive Summary KPIs
    md.append("## 1. Executive KPI Overview")
    md.append("| Metric | Value | Target / Benchmark |")
    md.append("|---|---|---|")
    md.append(f"| **Total Cheques Processed** | {summary.volume.total_cheques:,} | — |")
    md.append(f"| **Verified (Auto-Cleared)** | {summary.volume.verified_count:,} | — |")
    md.append(f"| **Manual Review Required** | {summary.volume.review_required_count:,} ({summary.reviews.manual_review_rate_pct:.1f}%) | < 15.0% |")
    md.append(f"| **Total Recognized Amount** | Rs. {summary.amounts.total_recognized_amount:,.2f} | — |")
    md.append(f"| **Average Cheque Amount** | Rs. {summary.amounts.average_amount:,.2f} | — |")
    md.append(f"| **Overall Average Confidence** | {summary.confidence.mean_confidence:.4f} | > 0.8500 |")
    md.append(f"| **Validation Pass Rate** | {summary.validation.pass_rate_pct:.1f}% | > 90.0% |")
    md.append(f"| **Median Latency (p50)** | {summary.latency.p50_latency_ms:.1f} ms | < 200.0 ms |")
    md.append(f"| **Tail Latency (p99)** | {summary.latency.p99_latency_ms:.1f} ms | < 500.0 ms |\n")

    # 2. Volume & Status Breakdown
    md.append("## 2. Operational Status Breakdown")
    for status, count in summary.volume.status_breakdown.items():
        pct = (count / summary.volume.total_cheques * 100.0) if summary.volume.total_cheques > 0 else 0.0
        md.append(f"- **{status}**: {count:,} cheques ({pct:.1f}%)")
    md.append("")

    # 3. Monetary Clearing Statistics
    md.append("## 3. Financial Clearing & Liquidity")
    md.append(f"- **Total Value Cleared**: Rs. {summary.amounts.total_recognized_amount:,.2f}")
    md.append(f"- **Mean Cheque Amount**: Rs. {summary.amounts.average_amount:,.2f}")
    md.append(f"- **Median Cheque Amount**: Rs. {summary.amounts.median_amount:,.2f}")
    md.append(f"- **Amount Range**: Rs. {summary.amounts.min_amount:,.2f} to Rs. {summary.amounts.max_amount:,.2f}")
    md.append(f"- **Standard Deviation**: Rs. {summary.amounts.std_amount:,.2f}\n")

    # 4. Confidence Distribution
    md.append("## 4. Model Confidence Distribution")
    md.append("| Tier / Interval | Count | Percentage |")
    md.append("|---|---|---|")
    total_eval = summary.confidence.total_evaluated or 1
    for bucket, b_count in summary.confidence.distribution_buckets.items():
        b_pct = (b_count / total_eval) * 100.0
        md.append(f"| `{bucket}` | {b_count:,} | {b_pct:.1f}% |")
    md.append("")

    # 5. Field Recognition Performance
    md.append("## 5. Field-Level Recognition & OCR Performance")
    md.append("| Field Name | Total Extractions | Avg Confidence | High Tier | Med Tier | Low Tier | Primary Method |")
    md.append("|---|---|---|---|---|---|---|")
    for f in summary.fields:
        method = "OCR" if f.ocr_method_count >= f.recognizer_method_count else "CNN Recognizer"
        md.append(
            f"| `{f.field_name}` | {f.total_extractions:,} | {f.avg_confidence:.4f} | "
            f"{f.high_tier_count:,} | {f.medium_tier_count:,} | {f.low_tier_count:,} | {method} |"
        )
    md.append("")

    # 6. Validation Rule Audits
    md.append("## 6. Compliance & Validation Failure Audit")
    md.append(f"- **Total Gate Evaluations**: {summary.validation.total_checks:,}")
    md.append(f"- **Passed Checks**: {summary.validation.valid_count:,}")
    md.append(f"- **Failed / Invalid Checks**: {summary.validation.invalid_count:,}")
    md.append(f"- **Warnings (Manual Review Triggers)**: {summary.validation.warning_count:,}\n")

    if summary.validation.top_review_reasons:
        md.append("### Top Review Triggers")
        for idx, item in enumerate(summary.validation.top_review_reasons, 1):
            md.append(f"{idx}. {item['reason']} ({item['count']} instances)")
        md.append("")

    # 7. Regulatory Disclaimer
    md.append("## 7. Regulatory & Compliance Notice")
    md.append(
        "> **Note on Anomaly Detection:** Any statistical outliers identified by ChequeSense "
        "function strictly as operational REVIEW SIGNALS for secondary teller review. "
        "Statistical anomalies are never classified as fraud without independent human and legal investigation."
    )

    return "\n".join(md)


def generate_operational_sla_report(
    session: Session,
    target_latency_ms: float = 250.0,
    target_review_pct: float = 15.0,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """Evaluates operations against bank Service Level Agreements (SLAs)."""
    summary = calculate_executive_dashboard(session, start_date, end_date)

    latency_compliant = summary.latency.p90_latency_ms <= target_latency_ms
    review_compliant = summary.reviews.manual_review_rate_pct <= target_review_pct

    return {
        "evaluation_timestamp": datetime.datetime.utcnow().isoformat(),
        "sla_overall_pass": latency_compliant and review_compliant,
        "latency_sla": {
            "target_ms": target_latency_ms,
            "actual_p90_ms": summary.latency.p90_latency_ms,
            "actual_avg_ms": summary.latency.avg_latency_ms,
            "compliant": latency_compliant,
        },
        "straight_through_processing_sla": {
            "target_review_rate_pct": target_review_pct,
            "actual_review_rate_pct": summary.reviews.manual_review_rate_pct,
            "auto_cleared_rate_pct": round(100.0 - summary.reviews.manual_review_rate_pct, 2),
            "compliant": review_compliant,
        },
    }


def generate_cheque_reconciliation_csv(session: Session, limit: int = 500) -> str:
    """Exports structured cheque reconciliation data to CSV format for bank ledger systems."""
    stmt = (
        select(Cheque)
        .order_by(desc(Cheque.created_at))
        .limit(limit)
    )
    cheques = list(session.scalars(stmt).all())

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Cheque ID",
        "Created At",
        "Status",
        "Overall Confidence",
        "Review Required",
        "Cheque Number",
        "Amount",
        "Date",
        "Payee Name",
        "Account Number",
    ])

    for chq in cheques:
        # Fetch fields for cheque
        fields_stmt = select(ExtractedField).where(ExtractedField.cheque_id == chq.id)
        fields = list(session.scalars(fields_stmt).all())
        field_map = {f.field_name: f.normalized_value for f in fields}

        writer.writerow([
            chq.cheque_identifier,
            chq.created_at.strftime("%Y-%m-%d %H:%M:%S") if chq.created_at else "",
            chq.status,
            f"{chq.overall_confidence:.4f}",
            "YES" if chq.review_required else "NO",
            field_map.get("cheque_number", ""),
            field_map.get("amount", ""),
            field_map.get("date", ""),
            field_map.get("payee_name", ""),
            field_map.get("account_number", ""),
        ])

    return output.getvalue()


def generate_field_performance_csv(session: Session) -> str:
    """Exports field-level performance metrics to CSV format."""
    summary = calculate_executive_dashboard(session)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Field Name",
        "Total Extractions",
        "Average Confidence",
        "Average Detection Confidence",
        "Average Extraction Confidence",
        "High Tier Count",
        "Medium Tier Count",
        "Low Tier Count",
        "OCR Method Count",
        "Recognizer Method Count",
    ])

    for f in summary.fields:
        writer.writerow([
            f.field_name,
            f.total_extractions,
            f"{f.avg_confidence:.4f}",
            f"{f.avg_detection_confidence:.4f}",
            f"{f.avg_extraction_confidence:.4f}",
            f.high_tier_count,
            f.medium_tier_count,
            f.low_tier_count,
            f.ocr_method_count,
            f.recognizer_method_count,
        ])

    return output.getvalue()
