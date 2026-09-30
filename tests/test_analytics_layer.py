"""Comprehensive unit and integration tests for ChequeSense banking analytics layer."""

import datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.analytics import (
    SQL_QUERIES,
    calculate_amount_metrics,
    calculate_confidence_metrics,
    calculate_executive_dashboard,
    calculate_field_metrics,
    calculate_latency_metrics,
    calculate_review_metrics,
    calculate_validation_metrics,
    calculate_volume_metrics,
    calculate_amount_trend,
    calculate_volume_trend,
    detect_amount_anomaly_signal,
    detect_latency_anomaly_signal,
    generate_cheque_reconciliation_csv,
    generate_executive_report_markdown,
    generate_field_performance_csv,
    generate_operational_sla_report,
)
from src.database.connection import Base
from src.database.models import (
    Cheque,
    ExtractedField,
    ProcessingRun,
    ValidationResult,
)


@pytest.fixture(scope="function")
def analytics_db_session():
    """Seeds an isolated in-memory SQLite database with 20 historical cheques and analytics data."""
    test_engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()

    base_time = datetime.datetime(2026, 9, 20, 10, 0, 0)

    # 1. Seed 20 cheques across 4 days
    # Amounts: 15 typical cheques (around Rs. 10,000 - 30,000)
    typical_amounts = [
        10000.0, 12500.0, 15000.0, 18000.0, 20000.0,
        22000.0, 25000.0, 14000.0, 16500.0, 19000.0,
        21000.0, 24000.0, 26000.0, 15500.0, 17500.0,
    ]

    for i in range(20):
        day_offset = i // 5  # 0 to 3 days
        chq_time = base_time + datetime.timedelta(days=day_offset, hours=i % 5)
        is_review = (i % 4 == 0)  # 5 cheques require review
        status = "REVIEW_REQUIRED" if is_review else ("VERIFIED" if i % 2 == 0 else "PROCESSED")
        conf = 0.75 + (i * 0.01)  # 0.75 to 0.94

        chq = Cheque(
            cheque_identifier=f"CHQ_ANALYTICS_{i:03d}",
            image_path=f"/storage/chq_{i:03d}.png",
            image_width=1200,
            image_height=600,
            image_hash=f"hash_{i:03d}",
            status=status,
            overall_confidence=round(conf, 4),
            review_required=is_review,
            created_at=chq_time,
        )
        session.add(chq)
        session.flush()

        # Processing run
        latency = 120.0 + (i * 5.0)  # 120ms to 215ms
        run = ProcessingRun(
            cheque_id=chq.id,
            run_timestamp=chq_time,
            status="SUCCESS",
            detector_model_version="fasterrcnn:v1.0",
            recognizer_model_version="cnn:v1.0",
            ocr_engine_version="tesseract:v5.5",
            total_latency_ms=latency,
        )
        session.add(run)
        session.flush()

        # Extracted fields
        # Amount field (for the first 15 cheques)
        if i < len(typical_amounts):
            amt_val = typical_amounts[i]
            session.add(
                ExtractedField(
                    cheque_id=chq.id,
                    processing_run_id=run.id,
                    field_name="amount",
                    raw_value=f"Rs. {amt_val:,.0f}/-",
                    normalized_value=f"{int(amt_val)}",
                    confidence=0.92,
                    detection_confidence=0.95,
                    extraction_confidence=0.90,
                    extraction_method="ocr",
                    confidence_tier="HIGH",
                    created_at=chq_time,
                )
            )

        # Cheque number field (all 20 cheques)
        session.add(
            ExtractedField(
                cheque_id=chq.id,
                processing_run_id=run.id,
                field_name="cheque_number",
                raw_value=f"{100000 + i}",
                normalized_value=f"{100000 + i}",
                confidence=0.96,
                detection_confidence=0.98,
                extraction_confidence=0.95,
                extraction_method="recognizer" if i % 2 == 0 else "ocr",
                confidence_tier="HIGH",
                created_at=chq_time,
            )
        )

        # Validation results
        if is_review:
            session.add(
                ValidationResult(
                    cheque_id=chq.id,
                    processing_run_id=run.id,
                    field_name="date",
                    check_type="DATE_VALIDITY",
                    validation_status="INVALID",
                    is_valid=False,
                    validation_reason="Cheque date is older than 90 days",
                    created_at=chq_time,
                )
            )
        else:
            session.add(
                ValidationResult(
                    cheque_id=chq.id,
                    processing_run_id=run.id,
                    field_name=None,
                    check_type="ALL_GATES",
                    validation_status="VALID",
                    is_valid=True,
                    validation_reason="All gates passed successfully",
                    created_at=chq_time,
                )
            )

    session.commit()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


# ------------------------------------------------------------------------------
# 1. SQL Query Repository Tests
# ------------------------------------------------------------------------------

def test_sql_queries_dictionary():
    """Verifies that all required reusable SQL queries are defined and non-empty."""
    required_keys = [
        "total_volume_and_review_rate",
        "volume_by_status",
        "daily_processing_volume",
        "amount_statistics",
        "field_level_performance",
        "validation_failure_analysis",
        "latency_percentiles",
    ]
    for key in required_keys:
        assert key in SQL_QUERIES
        assert len(SQL_QUERIES[key].strip()) > 0
        assert "SELECT" in SQL_QUERIES[key].upper()


# ------------------------------------------------------------------------------
# 2. Volume & Review Metric Tests
# ------------------------------------------------------------------------------

def test_calculate_volume_metrics(analytics_db_session):
    """Verifies total volume and status breakdown calculations."""
    vol = calculate_volume_metrics(analytics_db_session)
    assert vol.total_cheques == 20
    assert "REVIEW_REQUIRED" in vol.status_breakdown
    assert vol.review_required_count == 5
    assert vol.verified_count + vol.review_required_count + vol.status_breakdown.get("PROCESSED", 0) == 20


def test_calculate_review_metrics(analytics_db_session):
    """Verifies manual review percentage and auto-cleared counts."""
    rev = calculate_review_metrics(analytics_db_session)
    assert rev.total_cheques == 20
    assert rev.review_required_count == 5
    assert rev.auto_cleared_count == 15
    assert rev.manual_review_rate_pct == 25.0
    assert 0.70 <= rev.average_confidence <= 0.95


# ------------------------------------------------------------------------------
# 3. Amount Statistics Tests
# ------------------------------------------------------------------------------

def test_calculate_amount_metrics(analytics_db_session):
    """Verifies sum, average, min, max, and median for recognized amounts."""
    amt = calculate_amount_metrics(analytics_db_session)
    assert amt.cheques_with_amount_count == 15
    assert amt.min_amount == 10000.0
    assert amt.max_amount == 26000.0
    assert amt.total_recognized_amount > 200000.0
    assert 17000.0 <= amt.average_amount <= 20000.0
    assert amt.median_amount > 0.0
    assert amt.std_amount > 0.0


# ------------------------------------------------------------------------------
# 4. Confidence Distribution Tests
# ------------------------------------------------------------------------------

def test_calculate_confidence_metrics(analytics_db_session):
    """Verifies summary statistics and histogram buckets for confidence scores."""
    conf = calculate_confidence_metrics(analytics_db_session)
    assert conf.total_evaluated == 20
    assert 0.75 <= conf.min_confidence <= 0.76
    assert 0.93 <= conf.max_confidence <= 0.95
    assert 0.80 <= conf.mean_confidence <= 0.90
    assert conf.distribution_buckets["0.70 - 0.85"] > 0
    assert conf.distribution_buckets["0.85 - 0.95"] > 0


# ------------------------------------------------------------------------------
# 5. Field-Level Recognition Tests
# ------------------------------------------------------------------------------

def test_calculate_field_metrics(analytics_db_session):
    """Verifies per-field recognition performance, counts, and methods."""
    fields = calculate_field_metrics(analytics_db_session)
    f_map = {f.field_name: f for f in fields}
    assert "cheque_number" in f_map
    assert "amount" in f_map

    # Cheque number is present on all 20 cheques
    assert f_map["cheque_number"].total_extractions == 20
    assert f_map["cheque_number"].avg_confidence >= 0.90
    assert f_map["cheque_number"].ocr_method_count == 10
    assert f_map["cheque_number"].recognizer_method_count == 10

    # Amount is present on 15 cheques
    assert f_map["amount"].total_extractions == 15
    assert f_map["amount"].ocr_method_count == 15


# ------------------------------------------------------------------------------
# 6. Validation Rule Tests
# ------------------------------------------------------------------------------

def test_calculate_validation_metrics(analytics_db_session):
    """Verifies validation pass rates, breakdown, and top review reasons."""
    val = calculate_validation_metrics(analytics_db_session)
    assert val.total_checks == 20
    assert val.valid_count == 15
    assert val.invalid_count == 5
    assert val.pass_rate_pct == 75.0
    assert "DATE_VALIDITY" in val.checks_by_type
    assert len(val.top_review_reasons) > 0
    assert "90 days" in val.top_review_reasons[0]["reason"]


# ------------------------------------------------------------------------------
# 7. Latency Metric Tests
# ------------------------------------------------------------------------------

def test_calculate_latency_metrics(analytics_db_session):
    """Verifies latency percentiles (p50, p90, p99, min, max, avg)."""
    lat = calculate_latency_metrics(analytics_db_session)
    assert lat.total_runs == 20
    assert lat.min_latency_ms == 120.0
    assert lat.max_latency_ms == 215.0
    assert lat.min_latency_ms <= lat.p50_latency_ms <= lat.p90_latency_ms <= lat.p99_latency_ms <= lat.max_latency_ms


# ------------------------------------------------------------------------------
# 8. Time-Series Trends Tests
# ------------------------------------------------------------------------------

def test_calculate_volume_and_amount_trends(analytics_db_session):
    """Verifies temporal aggregation into daily volume and amount trend points."""
    vol_trends = calculate_volume_trend(analytics_db_session, interval="day")
    assert len(vol_trends) == 4  # 4 days
    for pt in vol_trends:
        assert pt.total_processed == 5
        assert pt.avg_confidence > 0.0

    amt_trends = calculate_amount_trend(analytics_db_session, interval="day")
    assert len(amt_trends) >= 3
    for apt in amt_trends:
        assert apt.total_amount > 0.0
        assert apt.avg_amount > 0.0


# ------------------------------------------------------------------------------
# 9. Statistical Anomaly Detection (Review Signal Only) Tests
# ------------------------------------------------------------------------------

def test_anomaly_detection_amount_signal(analytics_db_session):
    """Verifies that statistical anomaly detection triggers REVIEW SIGNALS only, never fraud claims."""
    # 1. Normal amount within distribution (Rs. 18,000) -> No signal
    normal_signal = detect_amount_anomaly_signal(analytics_db_session, candidate_amount=18000.0)
    assert normal_signal is None

    # 2. Extreme outlier amount (Rs. 500,000 when normal is ~Rs. 18,000)
    outlier_signal = detect_amount_anomaly_signal(analytics_db_session, candidate_amount=500000.0)
    assert outlier_signal is not None
    assert outlier_signal.signal_type == "AMOUNT_OUTLIER_REVIEW_SIGNAL"
    assert outlier_signal.severity == "HIGH"
    assert outlier_signal.anomaly_score > 3.0
    assert outlier_signal.observed_value == 500000.0

    # CRITICAL INVARIANT: Must NOT claim fraud!
    assert outlier_signal.is_fraud_claim is False
    assert "not indicative of fraud" in outlier_signal.message
    assert "verification" in outlier_signal.message


def test_anomaly_detection_latency_signal(analytics_db_session):
    """Verifies that execution latency spikes trigger an operational review signal."""
    # Normal latency (150ms) -> No signal
    normal_lat = detect_latency_anomaly_signal(analytics_db_session, candidate_latency_ms=150.0)
    assert normal_lat is None

    # Latency spike (800ms when normal is ~167ms)
    spike_lat = detect_latency_anomaly_signal(analytics_db_session, candidate_latency_ms=800.0)
    assert spike_lat is not None
    assert spike_lat.signal_type == "LATENCY_SPIKE_REVIEW_SIGNAL"
    assert spike_lat.is_fraud_claim is False
    assert spike_lat.anomaly_score >= 2.5


# ------------------------------------------------------------------------------
# 10. Executive Report & Export Tests
# ------------------------------------------------------------------------------

def test_executive_summary_and_markdown_report(analytics_db_session):
    """Verifies full executive dashboard generation and Markdown briefing rendering."""
    summary = calculate_executive_dashboard(analytics_db_session)
    assert summary.volume.total_cheques == 20
    assert summary.amounts.total_recognized_amount > 0.0

    md_report = generate_executive_report_markdown(summary)
    assert "# ChequeSense Banking Analytics & Operations Report" in md_report
    assert "Executive KPI Overview" in md_report
    assert "Total Cheques Processed" in md_report
    assert "Regulatory & Compliance Notice" in md_report
    assert "never classified as fraud" in md_report


def test_operational_sla_report(analytics_db_session):
    """Verifies SLA compliance evaluation against bank targets."""
    sla = generate_operational_sla_report(
        analytics_db_session,
        target_latency_ms=300.0,
        target_review_pct=30.0,
    )
    assert sla["sla_overall_pass"] is True
    assert sla["latency_sla"]["compliant"] is True
    assert sla["straight_through_processing_sla"]["compliant"] is True


def test_csv_export_generators(analytics_db_session):
    """Verifies CSV export for ledger reconciliation and field performance."""
    recon_csv = generate_cheque_reconciliation_csv(analytics_db_session, limit=10)
    assert "Cheque ID,Created At,Status" in recon_csv
    assert "CHQ_ANALYTICS_019" in recon_csv

    field_csv = generate_field_performance_csv(analytics_db_session)
    assert "Field Name,Total Extractions,Average Confidence" in field_csv
    assert "cheque_number" in field_csv
