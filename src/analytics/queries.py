"""Reusable SQL queries and SQLAlchemy query builders for ChequeSense banking analytics."""

from __future__ import annotations

import datetime
import math
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, case, desc, func, select
from sqlalchemy.orm import Session

from src.database.models import (
    Cheque,
    ExtractedField,
    Prediction,
    ProcessingRun,
    User,
    ValidationResult,
)

# ------------------------------------------------------------------------------
# Standard Reusable Raw SQL Statements (for BI, dashboards & reporting tools)
# ------------------------------------------------------------------------------

SQL_QUERIES: Dict[str, str] = {
    "total_volume_and_review_rate": """
        SELECT
            COUNT(*) AS total_cheques,
            COUNT(CASE WHEN review_required THEN 1 END) AS review_required_count,
            ROUND(COUNT(CASE WHEN review_required THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS review_rate_pct,
            ROUND(AVG(overall_confidence)::numeric, 4) AS avg_confidence
        FROM cheques
        WHERE (:start_date IS NULL OR created_at >= :start_date)
          AND (:end_date IS NULL OR created_at <= :end_date);
    """,
    "volume_by_status": """
        SELECT
            status,
            COUNT(*) AS cheque_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
        FROM cheques
        WHERE (:start_date IS NULL OR created_at >= :start_date)
          AND (:end_date IS NULL OR created_at <= :end_date)
        GROUP BY status
        ORDER BY cheque_count DESC;
    """,
    "daily_processing_volume": """
        SELECT
            DATE_TRUNC('day', created_at) AS date_bucket,
            COUNT(*) AS total_processed,
            COUNT(CASE WHEN status = 'VERIFIED' THEN 1 END) AS verified_count,
            COUNT(CASE WHEN review_required THEN 1 END) AS review_required_count,
            ROUND(AVG(overall_confidence)::numeric, 4) AS avg_confidence
        FROM cheques
        WHERE (:start_date IS NULL OR created_at >= :start_date)
          AND (:end_date IS NULL OR created_at <= :end_date)
        GROUP BY DATE_TRUNC('day', created_at)
        ORDER BY date_bucket ASC;
    """,
    "amount_statistics": r"""
        SELECT
            COUNT(*) AS amount_field_count,
            ROUND(SUM(NULLIF(regexp_replace(normalized_value, '[^0-9.]', '', 'g'), '')::numeric), 2) AS total_amount,
            ROUND(AVG(NULLIF(regexp_replace(normalized_value, '[^0-9.]', '', 'g'), '')::numeric), 2) AS avg_amount,
            MIN(NULLIF(regexp_replace(normalized_value, '[^0-9.]', '', 'g'), '')::numeric) AS min_amount,
            MAX(NULLIF(regexp_replace(normalized_value, '[^0-9.]', '', 'g'), '')::numeric) AS max_amount
        FROM extracted_fields
        WHERE field_name = 'amount'
          AND normalized_value ~ '^[0-9]+(\.[0-9]+)?$';
    """,
    "field_level_performance": """
        SELECT
            field_name,
            COUNT(*) AS occurrences,
            ROUND(AVG(confidence)::numeric, 4) AS avg_confidence,
            ROUND(AVG(detection_confidence)::numeric, 4) AS avg_detection_confidence,
            ROUND(AVG(extraction_confidence)::numeric, 4) AS avg_extraction_confidence,
            COUNT(CASE WHEN confidence_tier = 'HIGH' THEN 1 END) AS high_tier_count,
            COUNT(CASE WHEN confidence_tier = 'MEDIUM' THEN 1 END) AS medium_tier_count,
            COUNT(CASE WHEN confidence_tier = 'LOW' THEN 1 END) AS low_tier_count,
            COUNT(CASE WHEN extraction_method = 'ocr' THEN 1 END) AS ocr_count,
            COUNT(CASE WHEN extraction_method = 'recognizer' THEN 1 END) AS recognizer_count
        FROM extracted_fields
        GROUP BY field_name
        ORDER BY occurrences DESC;
    """,
    "validation_failure_analysis": """
        SELECT
            check_type,
            COUNT(*) AS total_checks,
            COUNT(CASE WHEN validation_status = 'VALID' THEN 1 END) AS valid_count,
            COUNT(CASE WHEN validation_status = 'INVALID' THEN 1 END) AS invalid_count,
            COUNT(CASE WHEN validation_status = 'WARNING' THEN 1 END) AS warning_count,
            ROUND(COUNT(CASE WHEN validation_status = 'VALID' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS pass_rate_pct
        FROM validation_results
        GROUP BY check_type
        ORDER BY total_checks DESC;
    """,
    "latency_percentiles": """
        SELECT
            COUNT(*) AS total_runs,
            ROUND(AVG(total_latency_ms)::numeric, 2) AS avg_latency_ms,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_latency_ms)::numeric, 2) AS p50_latency_ms,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY total_latency_ms)::numeric, 2) AS p90_latency_ms,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_latency_ms)::numeric, 2) AS p99_latency_ms,
            ROUND(MIN(total_latency_ms)::numeric, 2) AS min_latency_ms,
            ROUND(MAX(total_latency_ms)::numeric, 2) AS max_latency_ms
        FROM processing_runs;
    """,
}


# ------------------------------------------------------------------------------
# SQLAlchemy Analytical Query Functions
# ------------------------------------------------------------------------------

def get_total_cheques_count(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> int:
    """Returns total number of cheques processed within an optional date range."""
    stmt = select(func.count(Cheque.id))
    if start_date:
        stmt = stmt.where(Cheque.created_at >= start_date)
    if end_date:
        stmt = stmt.where(Cheque.created_at <= end_date)
    return session.scalar(stmt) or 0


def get_cheque_status_counts(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> Dict[str, int]:
    """Returns breakdown of cheques by operational status (PROCESSED, VERIFIED, etc.)."""
    stmt = select(Cheque.status, func.count(Cheque.id))
    if start_date:
        stmt = stmt.where(Cheque.created_at >= start_date)
    if end_date:
        stmt = stmt.where(Cheque.created_at <= end_date)
    stmt = stmt.group_by(Cheque.status)
    rows = session.execute(stmt).all()
    return {row[0]: int(row[1]) for row in rows}


def get_review_requirement_stats(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """Returns total volume, review required count, and percentage requiring manual review."""
    stmt = select(
        func.count(Cheque.id).label("total"),
        func.sum(case((Cheque.review_required == True, 1), else_=0)).label("review_required_count"),
        func.avg(Cheque.overall_confidence).label("avg_confidence"),
    )
    if start_date:
        stmt = stmt.where(Cheque.created_at >= start_date)
    if end_date:
        stmt = stmt.where(Cheque.created_at <= end_date)

    row = session.execute(stmt).one()
    total = int(row.total or 0)
    review_count = int(row.review_required_count or 0)
    rate = round((review_count / total * 100.0), 2) if total > 0 else 0.0
    avg_conf = round(float(row.avg_confidence or 0.0), 4)

    return {
        "total_cheques": total,
        "review_required_count": review_count,
        "auto_cleared_count": total - review_count,
        "manual_review_rate_pct": rate,
        "average_confidence": avg_conf,
    }


def get_amount_statistics(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """Calculates total recognized monetary amount, average amount, min, and max."""
    stmt = select(ExtractedField.normalized_value).where(
        ExtractedField.field_name == "amount"
    )
    if start_date or end_date:
        stmt = stmt.join(Cheque, ExtractedField.cheque_id == Cheque.id)
        if start_date:
            stmt = stmt.where(Cheque.created_at >= start_date)
        if end_date:
            stmt = stmt.where(Cheque.created_at <= end_date)

    values = session.scalars(stmt).all()
    numeric_amounts: List[float] = []
    for v in values:
        if not v:
            continue
        try:
            # Strip currency symbols and commas if any
            clean_str = "".join(ch for ch in str(v) if ch.isdigit() or ch == ".")
            if clean_str:
                numeric_amounts.append(float(clean_str))
        except ValueError:
            continue

    if not numeric_amounts:
        return {
            "count": 0,
            "total_recognized_amount": 0.0,
            "average_amount": 0.0,
            "min_amount": 0.0,
            "max_amount": 0.0,
            "median_amount": 0.0,
            "std_amount": 0.0,
        }

    numeric_amounts.sort()
    count = len(numeric_amounts)
    total_val = sum(numeric_amounts)
    avg_val = total_val / count
    min_val = numeric_amounts[0]
    max_val = numeric_amounts[-1]

    # Median
    mid = count // 2
    if count % 2 == 1:
        median_val = numeric_amounts[mid]
    else:
        median_val = (numeric_amounts[mid - 1] + numeric_amounts[mid]) / 2.0

    # Variance / std
    variance = sum((x - avg_val) ** 2 for x in numeric_amounts) / count
    std_val = math.sqrt(variance)

    return {
        "count": count,
        "total_recognized_amount": round(total_val, 2),
        "average_amount": round(avg_val, 2),
        "min_amount": round(min_val, 2),
        "max_amount": round(max_val, 2),
        "median_amount": round(median_val, 2),
        "std_amount": round(std_val, 2),
    }


def get_confidence_distribution(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """Calculates summary statistics and histogram distribution for cheque overall confidence."""
    stmt = select(Cheque.overall_confidence)
    if start_date:
        stmt = stmt.where(Cheque.created_at >= start_date)
    if end_date:
        stmt = stmt.where(Cheque.created_at <= end_date)

    conf_scores = [float(c) for c in session.scalars(stmt).all() if c is not None]
    if not conf_scores:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "buckets": {
                "< 0.50": 0,
                "0.50 - 0.70": 0,
                "0.70 - 0.85": 0,
                "0.85 - 0.95": 0,
                ">= 0.95": 0,
            },
        }

    conf_scores.sort()
    count = len(conf_scores)
    mean_val = sum(conf_scores) / count
    min_val = conf_scores[0]
    max_val = conf_scores[-1]
    mid = count // 2
    median_val = conf_scores[mid] if count % 2 == 1 else (conf_scores[mid - 1] + conf_scores[mid]) / 2.0
    variance = sum((x - mean_val) ** 2 for x in conf_scores) / count
    std_val = math.sqrt(variance)

    buckets = {
        "< 0.50": 0,
        "0.50 - 0.70": 0,
        "0.70 - 0.85": 0,
        "0.85 - 0.95": 0,
        ">= 0.95": 0,
    }
    for c in conf_scores:
        if c < 0.50:
            buckets["< 0.50"] += 1
        elif c < 0.70:
            buckets["0.50 - 0.70"] += 1
        elif c < 0.85:
            buckets["0.70 - 0.85"] += 1
        elif c < 0.95:
            buckets["0.85 - 0.95"] += 1
        else:
            buckets[">= 0.95"] += 1

    return {
        "count": count,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "std": round(std_val, 4),
        "buckets": buckets,
    }


def get_field_level_performance(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> List[Dict[str, Any]]:
    """Analyzes recognition and extraction accuracy per field type."""
    stmt = (
        select(
            ExtractedField.field_name,
            func.count(ExtractedField.id).label("total_extractions"),
            func.avg(ExtractedField.confidence).label("avg_confidence"),
            func.avg(ExtractedField.detection_confidence).label("avg_detection_confidence"),
            func.avg(ExtractedField.extraction_confidence).label("avg_extraction_confidence"),
            func.sum(case((ExtractedField.confidence_tier == "HIGH", 1), else_=0)).label("high_tier"),
            func.sum(case((ExtractedField.confidence_tier == "MEDIUM", 1), else_=0)).label("med_tier"),
            func.sum(case((ExtractedField.confidence_tier == "LOW", 1), else_=0)).label("low_tier"),
            func.sum(case((ExtractedField.extraction_method == "ocr", 1), else_=0)).label("ocr_count"),
            func.sum(case((ExtractedField.extraction_method == "recognizer", 1), else_=0)).label("recognizer_count"),
        )
        .group_by(ExtractedField.field_name)
        .order_by(desc("total_extractions"))
    )
    if start_date or end_date:
        stmt = stmt.join(Cheque, ExtractedField.cheque_id == Cheque.id)
        if start_date:
            stmt = stmt.where(Cheque.created_at >= start_date)
        if end_date:
            stmt = stmt.where(Cheque.created_at <= end_date)

    rows = session.execute(stmt).all()
    results = []
    for r in rows:
        results.append(
            {
                "field_name": r.field_name,
                "total_extractions": int(r.total_extractions or 0),
                "avg_confidence": round(float(r.avg_confidence or 0.0), 4),
                "avg_detection_confidence": round(float(r.avg_detection_confidence or 0.0), 4),
                "avg_extraction_confidence": round(float(r.avg_extraction_confidence or 0.0), 4),
                "high_tier_count": int(r.high_tier or 0),
                "medium_tier_count": int(r.med_tier or 0),
                "low_tier_count": int(r.low_tier or 0),
                "ocr_count": int(r.ocr_count or 0),
                "recognizer_count": int(r.recognizer_count or 0),
            }
        )
    return results


def get_validation_summary(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """Calculates validation pass/fail rates, status breakdowns, and top review triggers."""
    stmt_overall = select(
        func.count(ValidationResult.id).label("total"),
        func.sum(case((ValidationResult.validation_status == "VALID", 1), else_=0)).label("valid_count"),
        func.sum(case((ValidationResult.validation_status == "INVALID", 1), else_=0)).label("invalid_count"),
        func.sum(case((ValidationResult.validation_status == "WARNING", 1), else_=0)).label("warning_count"),
    )
    if start_date:
        stmt_overall = stmt_overall.where(ValidationResult.created_at >= start_date)
    if end_date:
        stmt_overall = stmt_overall.where(ValidationResult.created_at <= end_date)

    row = session.execute(stmt_overall).one()
    total = int(row.total or 0)
    valid_count = int(row.valid_count or 0)
    invalid_count = int(row.invalid_count or 0)
    warning_count = int(row.warning_count or 0)
    pass_rate = round((valid_count / total * 100.0), 2) if total > 0 else 0.0

    # Breakdown by check_type
    stmt_types = (
        select(
            ValidationResult.check_type,
            func.count(ValidationResult.id).label("total_checks"),
            func.sum(case((ValidationResult.validation_status == "VALID", 1), else_=0)).label("valid"),
            func.sum(case((ValidationResult.validation_status == "INVALID", 1), else_=0)).label("invalid"),
            func.sum(case((ValidationResult.validation_status == "WARNING", 1), else_=0)).label("warning"),
        )
        .group_by(ValidationResult.check_type)
        .order_by(desc("total_checks"))
    )
    if start_date:
        stmt_types = stmt_types.where(ValidationResult.created_at >= start_date)
    if end_date:
        stmt_types = stmt_types.where(ValidationResult.created_at <= end_date)

    type_rows = session.execute(stmt_types).all()
    check_breakdown = {}
    for tr in type_rows:
        tot_c = int(tr.total_checks or 0)
        v_c = int(tr.valid or 0)
        check_breakdown[tr.check_type] = {
            "total_checks": tot_c,
            "valid_count": v_c,
            "invalid_count": int(tr.invalid or 0),
            "warning_count": int(tr.warning or 0),
            "pass_rate_pct": round(v_c / tot_c * 100.0, 2) if tot_c > 0 else 0.0,
        }

    # Top reasons for review
    stmt_reasons = (
        select(ValidationResult.validation_reason, func.count(ValidationResult.id).label("freq"))
        .where(ValidationResult.validation_status.in_(["INVALID", "WARNING"]))
        .group_by(ValidationResult.validation_reason)
        .order_by(desc("freq"))
        .limit(10)
    )
    if start_date:
        stmt_reasons = stmt_reasons.where(ValidationResult.created_at >= start_date)
    if end_date:
        stmt_reasons = stmt_reasons.where(ValidationResult.created_at <= end_date)

    reason_rows = session.execute(stmt_reasons).all()
    top_reasons = [{"reason": rr[0], "count": int(rr[1])} for rr in reason_rows]

    return {
        "total_checks": total,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "warning_count": warning_count,
        "pass_rate_pct": pass_rate,
        "checks_by_type": check_breakdown,
        "top_review_reasons": top_reasons,
    }


def get_processing_latency_statistics(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """Calculates latency metrics (average, min, max, p50, p90, p99) from processing runs."""
    stmt = select(ProcessingRun.total_latency_ms)
    if start_date:
        stmt = stmt.where(ProcessingRun.run_timestamp >= start_date)
    if end_date:
        stmt = stmt.where(ProcessingRun.run_timestamp <= end_date)

    latencies = [float(val) for val in session.scalars(stmt).all() if val is not None]
    if not latencies:
        return {
            "total_runs": 0,
            "avg_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p90_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
        }

    latencies.sort()
    count = len(latencies)
    avg_lat = sum(latencies) / count
    min_lat = latencies[0]
    max_lat = latencies[-1]

    def percentile(p: float) -> float:
        idx = int(math.ceil(p * count)) - 1
        return latencies[max(0, min(idx, count - 1))]

    return {
        "total_runs": count,
        "avg_latency_ms": round(avg_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "p50_latency_ms": round(percentile(0.50), 2),
        "p90_latency_ms": round(percentile(0.90), 2),
        "p99_latency_ms": round(percentile(0.99), 2),
    }
