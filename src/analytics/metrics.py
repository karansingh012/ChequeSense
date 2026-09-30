"""Metric models and computation engine for ChequeSense banking analytics."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from sqlalchemy.orm import Session

from src.analytics.queries import (
    get_amount_statistics,
    get_cheque_status_counts,
    get_confidence_distribution,
    get_field_level_performance,
    get_processing_latency_statistics,
    get_review_requirement_stats,
    get_total_cheques_count,
    get_validation_summary,
)


# ------------------------------------------------------------------------------
# Pydantic Metric Schemas
# ------------------------------------------------------------------------------

class VolumeMetrics(BaseModel):
    """Overall volume and operational statuses."""

    total_cheques: int = Field(description="Total count of cheques processed")
    status_breakdown: Dict[str, int] = Field(
        default_factory=dict, description="Counts by operational status (PROCESSED, VERIFIED, etc.)"
    )
    verified_count: int = Field(default=0, description="Cheques verified without intervention")
    review_required_count: int = Field(default=0, description="Cheques sent to teller review")
    invalid_count: int = Field(default=0, description="Cheques rejected or invalid")

    model_config = ConfigDict(from_attributes=True)


class AmountMetrics(BaseModel):
    """Monetary clearing amounts recognized from cheque fields."""

    cheques_with_amount_count: int = Field(description="Count of cheques with recognized amount")
    total_recognized_amount: float = Field(description="Sum of all recognized cheque amounts")
    average_amount: float = Field(description="Mean amount recognized")
    median_amount: float = Field(description="Median amount recognized")
    min_amount: float = Field(description="Minimum recognized cheque amount")
    max_amount: float = Field(description="Maximum recognized cheque amount")
    std_amount: float = Field(description="Standard deviation of recognized amounts")

    model_config = ConfigDict(from_attributes=True)


class ReviewMetrics(BaseModel):
    """Human review rates and queue efficiency."""

    total_cheques: int = Field(description="Total cheques evaluated")
    review_required_count: int = Field(description="Cheques requiring manual teller review")
    auto_cleared_count: int = Field(description="Cheques cleared automatically without review")
    manual_review_rate_pct: float = Field(description="Percentage of volume flagged for review")
    average_confidence: float = Field(description="Average overall cheque confidence")

    model_config = ConfigDict(from_attributes=True)


class ConfidenceMetrics(BaseModel):
    """Statistical distribution of model confidence across cheques."""

    total_evaluated: int = Field(description="Number of cheques evaluated")
    mean_confidence: float = Field(description="Mean overall confidence")
    median_confidence: float = Field(description="Median overall confidence")
    min_confidence: float = Field(description="Minimum observed confidence")
    max_confidence: float = Field(description="Maximum observed confidence")
    std_confidence: float = Field(description="Standard deviation of confidence")
    distribution_buckets: Dict[str, int] = Field(
        default_factory=dict, description="Histogram frequency counts by confidence interval"
    )

    model_config = ConfigDict(from_attributes=True)


class FieldPerformanceMetrics(BaseModel):
    """Per-field extraction accuracy and confidence metrics."""

    field_name: str = Field(description="Name of the cheque field (amount, cheque_number, date, etc.)")
    total_extractions: int = Field(description="Number of times this field was extracted")
    avg_confidence: float = Field(description="Average composite confidence")
    avg_detection_confidence: float = Field(description="Average bounding box detector confidence")
    avg_extraction_confidence: float = Field(description="Average OCR/recognizer confidence")
    high_tier_count: int = Field(default=0, description="Count of extractions in HIGH confidence tier")
    medium_tier_count: int = Field(default=0, description="Count of extractions in MEDIUM confidence tier")
    low_tier_count: int = Field(default=0, description="Count of extractions in LOW confidence tier")
    ocr_method_count: int = Field(default=0, description="Extractions performed by OCR engine")
    recognizer_method_count: int = Field(default=0, description="Extractions performed by digit CNN")

    model_config = ConfigDict(from_attributes=True)


class ValidationMetrics(BaseModel):
    """Validation rule outcomes and failure gate performance."""

    total_checks: int = Field(description="Total individual validation checks executed")
    valid_count: int = Field(description="Checks passing without violation")
    invalid_count: int = Field(description="Hard validation failures")
    warning_count: int = Field(description="Soft warnings requiring verification")
    pass_rate_pct: float = Field(description="Percentage of checks evaluated as VALID")
    checks_by_type: Dict[str, Any] = Field(
        default_factory=dict, description="Performance breakdown per check type"
    )
    top_review_reasons: List[Dict[str, Any]] = Field(
        default_factory=list, description="Most frequent reasons cheques were flagged for review"
    )

    model_config = ConfigDict(from_attributes=True)


class LatencyMetrics(BaseModel):
    """End-to-end inference and pipeline processing speed."""

    total_runs: int = Field(description="Total pipeline execution passes logged")
    avg_latency_ms: float = Field(description="Average execution latency in milliseconds")
    p50_latency_ms: float = Field(description="Median (50th percentile) latency in milliseconds")
    p90_latency_ms: float = Field(description="90th percentile latency in milliseconds")
    p99_latency_ms: float = Field(description="99th percentile latency in milliseconds")
    min_latency_ms: float = Field(description="Fastest execution latency in milliseconds")
    max_latency_ms: float = Field(description="Slowest execution latency in milliseconds")

    model_config = ConfigDict(from_attributes=True)


class ExecutiveSummary(BaseModel):
    """Consolidated KPI report across all operational facets."""

    generated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    period_start: Optional[datetime.datetime] = None
    period_end: Optional[datetime.datetime] = None
    volume: VolumeMetrics
    amounts: AmountMetrics
    reviews: ReviewMetrics
    confidence: ConfidenceMetrics
    validation: ValidationMetrics
    latency: LatencyMetrics
    fields: List[FieldPerformanceMetrics]

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Metric Calculator Functions
# ------------------------------------------------------------------------------

def calculate_volume_metrics(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> VolumeMetrics:
    """Computes total volume and cheque status breakdown."""
    total = get_total_cheques_count(session, start_date, end_date)
    status_counts = get_cheque_status_counts(session, start_date, end_date)
    return VolumeMetrics(
        total_cheques=total,
        status_breakdown=status_counts,
        verified_count=status_counts.get("VERIFIED", 0),
        review_required_count=status_counts.get("REVIEW_REQUIRED", 0),
        invalid_count=status_counts.get("INVALID", 0),
    )


def calculate_amount_metrics(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> AmountMetrics:
    """Computes monetary clearing amounts, totals, averages, and statistical dispersion."""
    raw = get_amount_statistics(session, start_date, end_date)
    return AmountMetrics(
        cheques_with_amount_count=raw["count"],
        total_recognized_amount=raw["total_recognized_amount"],
        average_amount=raw["average_amount"],
        median_amount=raw["median_amount"],
        min_amount=raw["min_amount"],
        max_amount=raw["max_amount"],
        std_amount=raw["std_amount"],
    )


def calculate_review_metrics(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> ReviewMetrics:
    """Computes manual review requirements and percentages."""
    raw = get_review_requirement_stats(session, start_date, end_date)
    return ReviewMetrics(
        total_cheques=raw["total_cheques"],
        review_required_count=raw["review_required_count"],
        auto_cleared_count=raw["auto_cleared_count"],
        manual_review_rate_pct=raw["manual_review_rate_pct"],
        average_confidence=raw["average_confidence"],
    )


def calculate_confidence_metrics(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> ConfidenceMetrics:
    """Computes confidence distribution and summary statistics."""
    raw = get_confidence_distribution(session, start_date, end_date)
    return ConfidenceMetrics(
        total_evaluated=raw["count"],
        mean_confidence=raw["mean"],
        median_confidence=raw["median"],
        min_confidence=raw["min"],
        max_confidence=raw["max"],
        std_confidence=raw["std"],
        distribution_buckets=raw["buckets"],
    )


def calculate_validation_metrics(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> ValidationMetrics:
    """Computes validation pass rates, breakdown by check type, and top failure reasons."""
    raw = get_validation_summary(session, start_date, end_date)
    return ValidationMetrics(
        total_checks=raw["total_checks"],
        valid_count=raw["valid_count"],
        invalid_count=raw["invalid_count"],
        warning_count=raw["warning_count"],
        pass_rate_pct=raw["pass_rate_pct"],
        checks_by_type=raw["checks_by_type"],
        top_review_reasons=raw["top_review_reasons"],
    )


def calculate_field_metrics(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> List[FieldPerformanceMetrics]:
    """Computes recognition and extraction performance for every supported field."""
    raw_list = get_field_level_performance(session, start_date, end_date)
    return [
        FieldPerformanceMetrics(
            field_name=item["field_name"],
            total_extractions=item["total_extractions"],
            avg_confidence=item["avg_confidence"],
            avg_detection_confidence=item["avg_detection_confidence"],
            avg_extraction_confidence=item["avg_extraction_confidence"],
            high_tier_count=item["high_tier_count"],
            medium_tier_count=item["medium_tier_count"],
            low_tier_count=item["low_tier_count"],
            ocr_method_count=item["ocr_count"],
            recognizer_method_count=item["recognizer_count"],
        )
        for item in raw_list
    ]


def calculate_latency_metrics(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> LatencyMetrics:
    """Computes execution speed percentiles."""
    raw = get_processing_latency_statistics(session, start_date, end_date)
    return LatencyMetrics(
        total_runs=raw["total_runs"],
        avg_latency_ms=raw["avg_latency_ms"],
        p50_latency_ms=raw["p50_latency_ms"],
        p90_latency_ms=raw["p90_latency_ms"],
        p99_latency_ms=raw["p99_latency_ms"],
        min_latency_ms=raw["min_latency_ms"],
        max_latency_ms=raw["max_latency_ms"],
    )


def calculate_executive_dashboard(
    session: Session,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> ExecutiveSummary:
    """Aggregates all KPI modules into an executive analytical payload."""
    return ExecutiveSummary(
        period_start=start_date,
        period_end=end_date,
        volume=calculate_volume_metrics(session, start_date, end_date),
        amounts=calculate_amount_metrics(session, start_date, end_date),
        reviews=calculate_review_metrics(session, start_date, end_date),
        confidence=calculate_confidence_metrics(session, start_date, end_date),
        validation=calculate_validation_metrics(session, start_date, end_date),
        latency=calculate_latency_metrics(session, start_date, end_date),
        fields=calculate_field_metrics(session, start_date, end_date),
    )
