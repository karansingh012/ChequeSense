"""Banking analytics, KPI reporting, and trend analysis package for ChequeSense."""

from src.analytics.metrics import (
    AmountMetrics,
    ConfidenceMetrics,
    ExecutiveSummary,
    FieldPerformanceMetrics,
    LatencyMetrics,
    ReviewMetrics,
    ValidationMetrics,
    VolumeMetrics,
    calculate_amount_metrics,
    calculate_confidence_metrics,
    calculate_executive_dashboard,
    calculate_field_metrics,
    calculate_latency_metrics,
    calculate_review_metrics,
    calculate_validation_metrics,
    calculate_volume_metrics,
)
from src.analytics.queries import (
    SQL_QUERIES,
    get_amount_statistics,
    get_cheque_status_counts,
    get_confidence_distribution,
    get_field_level_performance,
    get_processing_latency_statistics,
    get_review_requirement_stats,
    get_total_cheques_count,
    get_validation_summary,
)
from src.analytics.reports import (
    generate_cheque_reconciliation_csv,
    generate_executive_report_markdown,
    generate_field_performance_csv,
    generate_operational_sla_report,
)
from src.analytics.trends import (
    AmountTrendPoint,
    ReviewSignal,
    VolumeTrendPoint,
    calculate_amount_trend,
    calculate_volume_trend,
    detect_amount_anomaly_signal,
    detect_latency_anomaly_signal,
)

__all__ = [
    # Queries & SQL
    "SQL_QUERIES",
    "get_total_cheques_count",
    "get_cheque_status_counts",
    "get_review_requirement_stats",
    "get_amount_statistics",
    "get_confidence_distribution",
    "get_field_level_performance",
    "get_validation_summary",
    "get_processing_latency_statistics",
    # Metrics
    "VolumeMetrics",
    "AmountMetrics",
    "ReviewMetrics",
    "ConfidenceMetrics",
    "FieldPerformanceMetrics",
    "ValidationMetrics",
    "LatencyMetrics",
    "ExecutiveSummary",
    "calculate_volume_metrics",
    "calculate_amount_metrics",
    "calculate_review_metrics",
    "calculate_confidence_metrics",
    "calculate_field_metrics",
    "calculate_validation_metrics",
    "calculate_latency_metrics",
    "calculate_executive_dashboard",
    # Trends & Review Signals
    "VolumeTrendPoint",
    "AmountTrendPoint",
    "ReviewSignal",
    "calculate_volume_trend",
    "calculate_amount_trend",
    "detect_amount_anomaly_signal",
    "detect_latency_anomaly_signal",
    # Reports
    "generate_executive_report_markdown",
    "generate_operational_sla_report",
    "generate_cheque_reconciliation_csv",
    "generate_field_performance_csv",
]
