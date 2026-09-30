"""Time series trends and statistical anomaly detection review signals for ChequeSense."""

from __future__ import annotations

import datetime
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import Cheque, ExtractedField, ProcessingRun


# ------------------------------------------------------------------------------
# Trend Point Schemas
# ------------------------------------------------------------------------------

class VolumeTrendPoint(BaseModel):
    """Time-bucketed cheque processing volume and operational breakdown."""

    period: str = Field(description="ISO formatted timestamp or date bucket string")
    total_processed: int = Field(description="Total cheques processed in this interval")
    verified_count: int = Field(description="Cheques verified in this interval")
    review_required_count: int = Field(description="Cheques requiring review in this interval")
    review_rate_pct: float = Field(description="Percentage requiring review")
    avg_confidence: float = Field(description="Mean confidence across cheques in interval")

    model_config = ConfigDict(from_attributes=True)


class AmountTrendPoint(BaseModel):
    """Time-bucketed monetary clearing values."""

    period: str = Field(description="Date bucket string")
    cheque_count: int = Field(description="Cheques with recognized amount in bucket")
    total_amount: float = Field(description="Sum of recognized amounts in bucket")
    avg_amount: float = Field(description="Average cheque amount in bucket")

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Anomaly Detection Review Signal Schemas
# ------------------------------------------------------------------------------

class ReviewSignal(BaseModel):
    """Statistical review trigger for human verification.
    
    IMPORTANT COMPLIANCE INVARIANT:
    Anomaly detection produces REVIEW SIGNALS ONLY.
    Under NO circumstances must an anomaly signal be labeled as fraud or counterfeit.
    """

    signal_type: str = Field(description="Category of review signal: AMOUNT_OUTLIER, LATENCY_OUTLIER, etc.")
    severity: str = Field(description="Operational review priority: HIGH, MEDIUM, LOW")
    anomaly_score: float = Field(description="Statistical deviation score (e.g. Z-score or IQR distance)")
    observed_value: float = Field(description="Observed value of the evaluated metric")
    historical_benchmark: Dict[str, float] = Field(description="Historical reference statistics (median, mean, etc.)")
    message: str = Field(description="Operational advisory message for reviewing teller")
    is_fraud_claim: bool = Field(
        default=False,
        description="ALWAYS FALSE. Statistical anomaly is NOT a fraud accusation.",
    )

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Trend Analysis Calculations
# ------------------------------------------------------------------------------

def calculate_volume_trend(
    session: Session,
    interval: str = "day",  # 'day', 'hour', 'week'
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> List[VolumeTrendPoint]:
    """Calculates time series volume, review rates, and confidence over time."""
    stmt = select(Cheque.created_at, Cheque.status, Cheque.review_required, Cheque.overall_confidence)
    if start_date:
        stmt = stmt.where(Cheque.created_at >= start_date)
    if end_date:
        stmt = stmt.where(Cheque.created_at <= end_date)
    stmt = stmt.order_by(Cheque.created_at.asc())

    rows = session.execute(stmt).all()
    if not rows:
        return []

    # Bucket grouping in Python for cross-database (SQLite + PostgreSQL) compatibility
    buckets: Dict[str, List[Tuple[Any, ...]]] = {}
    for created_at, status, review_req, conf in rows:
        if not created_at:
            continue
        if interval == "hour":
            key = created_at.strftime("%Y-%m-%d %H:00")
        elif interval == "week":
            # Monday of the week
            start_of_week = created_at - datetime.timedelta(days=created_at.weekday())
            key = start_of_week.strftime("%Y-%m-%d (W%W)")
        else:  # default 'day'
            key = created_at.strftime("%Y-%m-%d")

        if key not in buckets:
            buckets[key] = []
        buckets[key].append((status, review_req, conf))

    trend_points: List[VolumeTrendPoint] = []
    for period, items in sorted(buckets.items()):
        total = len(items)
        verified = sum(1 for st, _, _ in items if st == "VERIFIED")
        review_req = sum(1 for _, rr, _ in items if rr)
        review_rate = round((review_req / total * 100.0), 2) if total > 0 else 0.0
        conf_scores = [c for _, _, c in items if c is not None]
        avg_conf = round(sum(conf_scores) / len(conf_scores), 4) if conf_scores else 0.0

        trend_points.append(
            VolumeTrendPoint(
                period=period,
                total_processed=total,
                verified_count=verified,
                review_required_count=review_req,
                review_rate_pct=review_rate,
                avg_confidence=avg_conf,
            )
        )

    return trend_points


def calculate_amount_trend(
    session: Session,
    interval: str = "day",
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> List[AmountTrendPoint]:
    """Calculates monetary clearing volume and average amount per time bucket."""
    stmt = (
        select(Cheque.created_at, ExtractedField.normalized_value)
        .join(ExtractedField, ExtractedField.cheque_id == Cheque.id)
        .where(ExtractedField.field_name == "amount")
    )
    if start_date:
        stmt = stmt.where(Cheque.created_at >= start_date)
    if end_date:
        stmt = stmt.where(Cheque.created_at <= end_date)
    stmt = stmt.order_by(Cheque.created_at.asc())

    rows = session.execute(stmt).all()
    if not rows:
        return []

    buckets: Dict[str, List[float]] = {}
    for created_at, val in rows:
        if not created_at or not val:
            continue
        try:
            clean_str = "".join(ch for ch in str(val) if ch.isdigit() or ch == ".")
            if not clean_str:
                continue
            amt = float(clean_str)
        except ValueError:
            continue

        if interval == "hour":
            key = created_at.strftime("%Y-%m-%d %H:00")
        elif interval == "week":
            start_of_week = created_at - datetime.timedelta(days=created_at.weekday())
            key = start_of_week.strftime("%Y-%m-%d (W%W)")
        else:
            key = created_at.strftime("%Y-%m-%d")

        if key not in buckets:
            buckets[key] = []
        buckets[key].append(amt)

    trend_points: List[AmountTrendPoint] = []
    for period, amounts in sorted(buckets.items()):
        total_amt = sum(amounts)
        count = len(amounts)
        avg_amt = total_amt / count if count > 0 else 0.0

        trend_points.append(
            AmountTrendPoint(
                period=period,
                cheque_count=count,
                total_amount=round(total_amt, 2),
                avg_amount=round(avg_amt, 2),
            )
        )

    return trend_points


# ------------------------------------------------------------------------------
# Statistical Anomaly Detection (Review Signal Only)
# ------------------------------------------------------------------------------

MIN_HISTORICAL_SAMPLES_FOR_ANOMALY = 10


def detect_amount_anomaly_signal(
    session: Session,
    candidate_amount: float,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 2.5,
) -> Optional[ReviewSignal]:
    """Evaluates whether a candidate cheque amount is a statistical outlier relative to history.
    
    CRITICAL POLICY:
    Returns a REVIEW SIGNAL to alert tellers of high-value or unusual disbursements.
    Does NOT declare fraud.
    """
    stmt = select(ExtractedField.normalized_value).where(
        ExtractedField.field_name == "amount"
    )
    all_values = session.scalars(stmt).all()

    historical_amounts: List[float] = []
    for v in all_values:
        if not v:
            continue
        try:
            clean_str = "".join(ch for ch in str(v) if ch.isdigit() or ch == ".")
            if clean_str:
                historical_amounts.append(float(clean_str))
        except ValueError:
            continue

    # Need sufficient historical sample size for reliable distribution metrics
    if len(historical_amounts) < MIN_HISTORICAL_SAMPLES_FOR_ANOMALY:
        return None

    historical_amounts.sort()
    n = len(historical_amounts)
    mean_val = sum(historical_amounts) / n
    variance = sum((x - mean_val) ** 2 for x in historical_amounts) / n
    std_val = math.sqrt(variance)

    # Quartiles for robust IQR
    q1_idx = int(0.25 * n)
    q3_idx = int(0.75 * n)
    q1 = historical_amounts[q1_idx]
    q3 = historical_amounts[q3_idx]
    iqr = q3 - q1
    iqr_upper = q3 + (iqr_multiplier * iqr)

    # Z-score calculation
    z_score = (candidate_amount - mean_val) / std_val if std_val > 0 else 0.0

    # Determine if either robust IQR or Z-score threshold is exceeded
    is_outlier = (candidate_amount > iqr_upper and iqr > 0) or (z_score >= z_threshold)

    if not is_outlier:
        return None

    severity = "HIGH" if z_score >= 4.0 or candidate_amount > (q3 + 4.0 * iqr) else "MEDIUM"
    benchmark = {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "iqr_upper_bound": round(iqr_upper, 2),
    }

    message = (
        f"Amount (Rs. {candidate_amount:,.2f}) exceeds normal historical distribution "
        f"(Mean: Rs. {mean_val:,.2f}, Q3: Rs. {q3:,.2f}, Z-score: {z_score:.2f}). "
        f"Flagged for secondary teller verification. Note: High-value transactions are valid business operations and not indicative of fraud."
    )

    return ReviewSignal(
        signal_type="AMOUNT_OUTLIER_REVIEW_SIGNAL",
        severity=severity,
        anomaly_score=round(z_score, 2),
        observed_value=candidate_amount,
        historical_benchmark=benchmark,
        message=message,
        is_fraud_claim=False,
    )


def detect_latency_anomaly_signal(
    session: Session,
    candidate_latency_ms: float,
    z_threshold: float = 2.5,
) -> Optional[ReviewSignal]:
    """Detects unusual pipeline execution delays compared to historical performance."""
    stmt = select(ProcessingRun.total_latency_ms)
    latencies = [float(v) for v in session.scalars(stmt).all() if v is not None]

    if len(latencies) < MIN_HISTORICAL_SAMPLES_FOR_ANOMALY:
        return None

    n = len(latencies)
    mean_lat = sum(latencies) / n
    variance = sum((x - mean_lat) ** 2 for x in latencies) / n
    std_lat = math.sqrt(variance)

    z_score = (candidate_latency_ms - mean_lat) / std_lat if std_lat > 0 else 0.0

    if z_score < z_threshold:
        return None

    benchmark = {
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
    }

    message = (
        f"Inference latency ({candidate_latency_ms:.1f}ms) significantly exceeds historical average "
        f"({mean_lat:.1f}ms ± {std_lat:.1f}ms, Z-score: {z_score:.2f}). "
        f"Flagged for pipeline telemetry monitoring."
    )

    return ReviewSignal(
        signal_type="LATENCY_SPIKE_REVIEW_SIGNAL",
        severity="LOW" if z_score < 3.5 else "MEDIUM",
        anomaly_score=round(z_score, 2),
        observed_value=candidate_latency_ms,
        historical_benchmark=benchmark,
        message=message,
        is_fraud_claim=False,
    )
