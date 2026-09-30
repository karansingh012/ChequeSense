"""Unit and integration tests for ChequeSense Streamlit dashboard layer."""

import compileall
import io
import tempfile
from pathlib import Path
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from dashboard.components.api_client import BackendClient
from dashboard.components.styles import (
    get_confidence_badge_html,
    get_status_badge_html,
)
from dashboard.components.visualizations import (
    draw_bounding_boxes_on_cheque,
    format_currency_inr,
)
from src.database.connection import Base
from src.database.models import Cheque, ExtractedField, ValidationResult


@pytest.fixture(scope="function")
def dashboard_test_client(monkeypatch):
    """Sets up an isolated in-memory test database and temp upload directory for BackendClient."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    temp_dir = tempfile.TemporaryDirectory()
    temp_upload_path = Path(temp_dir.name)

    # Monkeypatch get_engine & upload directory
    import src.database.connection
    monkeypatch.setattr(src.database.connection, "_default_engine", test_engine)
    monkeypatch.setattr(src.database.connection, "SessionLocal", TestingSessionLocal)
    monkeypatch.setenv("UPLOAD_DIR", str(temp_upload_path))

    client = BackendClient(base_url="http://127.0.0.1:9999/api/v1")
    # Force service-layer fallback mode for isolated test
    client._http_available = False

    yield client, TestingSessionLocal

    temp_dir.cleanup()
    Base.metadata.drop_all(bind=test_engine)


# ------------------------------------------------------------------------------
# 1. Dashboard Code Syntax & Compilation Tests
# ------------------------------------------------------------------------------

def test_dashboard_files_compile():
    """Verifies that all Streamlit app and page files compile cleanly without syntax errors."""
    dashboard_dir = Path("dashboard")
    assert dashboard_dir.exists()
    compiled = compileall.compile_dir(str(dashboard_dir), quiet=True)
    assert compiled is True


# ------------------------------------------------------------------------------
# 2. Visualizations & Formatting Tests
# ------------------------------------------------------------------------------

def test_format_currency_inr():
    """Verifies Indian Rupee currency formatting."""
    assert format_currency_inr(25000) == "₹ 25,000.00"
    assert format_currency_inr(1500000.5) == "₹ 1,500,000.50"
    assert format_currency_inr("50000") == "₹ 50,000.00"
    assert "₹" in format_currency_inr("invalid")


def test_draw_bounding_boxes_on_cheque():
    """Verifies bounding box rendering onto a cheque image."""
    base_img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    fields = [
        {
            "field_name": "cheque_number",
            "confidence": 0.95,
            "bounding_box_json": {"xmin": 200, "ymin": 250, "xmax": 350, "ymax": 280},
        },
        {
            "field_name": "amount",
            "confidence": 0.65,  # Low confidence
            "bounding_box_json": {"xmin": 400, "ymin": 100, "xmax": 550, "ymax": 140},
        },
    ]

    annotated = draw_bounding_boxes_on_cheque(base_img, fields, highlight_low_confidence=True)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (600, 300)
    # Ensure drawing modified the image pixels
    assert annotated.tobytes() != base_img.tobytes()


# ------------------------------------------------------------------------------
# 3. Styling & Badge Helpers Tests
# ------------------------------------------------------------------------------

def test_status_badges():
    """Verifies status badge HTML generation."""
    v_badge = get_status_badge_html("VERIFIED")
    assert "badge-verified" in v_badge
    assert "VERIFIED" in v_badge

    r_badge = get_status_badge_html("REVIEW_REQUIRED")
    assert "badge-review" in r_badge
    assert "REVIEW REQUIRED" in r_badge

    i_badge = get_status_badge_html("INVALID")
    assert "badge-invalid" in i_badge


def test_confidence_badges():
    """Verifies confidence tier badge HTML generation."""
    high_badge = get_confidence_badge_html(0.96, "HIGH")
    assert "badge-high" in high_badge
    assert "96.0%" in high_badge

    low_badge = get_confidence_badge_html(0.55, "LOW")
    assert "badge-low" in low_badge
    assert "55.0%" in low_badge


# ------------------------------------------------------------------------------
# 4. BackendClient Tests
# ------------------------------------------------------------------------------

def test_client_upload_and_get_cheque(dashboard_test_client):
    """Verifies BackendClient upload and retrieval."""
    client, _ = dashboard_test_client
    img = Image.new("RGB", (800, 400), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    upload_res = client.upload_cheque("test_chq.png", raw_bytes)
    assert upload_res["id"] is not None
    assert upload_res["cheque_identifier"].startswith("CHQ_")
    assert Path(upload_res["image_path"]).exists()

    # Get cheque
    retrieved = client.get_cheque(upload_res["id"])
    assert retrieved is not None
    assert retrieved["id"] == upload_res["id"]
    assert retrieved["image_width"] == 800


def test_client_submit_review_resolution(dashboard_test_client):
    """Verifies human reviewer resolution submission."""
    client, session_local = dashboard_test_client

    # Seed cheque with validation check
    with session_local() as session:
        chq = Cheque(
            cheque_identifier="CHQ_REV_TEST",
            image_path="/dummy.png",
            image_width=1000,
            image_height=500,
            image_hash="hash123",
            status="REVIEW_REQUIRED",
            review_required=True,
        )
        session.add(chq)
        session.flush()

        field = ExtractedField(
            cheque_id=chq.id,
            processing_run_id=1,
            field_name="amount",
            raw_value="15000",
            normalized_value="15000",
            confidence=0.60,
        )
        session.add(field)

        val = ValidationResult(
            cheque_id=chq.id,
            processing_run_id=1,
            check_type="FORMAT",
            validation_status="WARNING",
            is_valid=True,
            validation_reason="Low confidence score",
        )
        session.add(val)
        session.commit()
        val_id = val.id
        chq_id = chq.id

    # Submit correction
    success = client.submit_review_resolution(
        validation_id=val_id,
        resolution="CORRECTED",
        reviewer_id=1,
        corrected_fields={"amount": "20000"},
    )
    assert success is True

    # Verify updated in DB
    updated_chq = client.get_cheque(chq_id)
    assert updated_chq["status"] == "VERIFIED"
    assert updated_chq["review_required"] is False
    amt_f = next(f for f in updated_chq["extracted_fields"] if f["field_name"] == "amount")
    assert amt_f["normalized_value"] == "20000"
    assert amt_f["confidence"] == 1.0


def test_client_analytics(dashboard_test_client):
    """Verifies BackendClient analytics and trend retrieval."""
    client, _ = dashboard_test_client
    summary = client.get_analytics_summary()
    assert "volume" in summary
    assert "amounts" in summary
    assert "reviews" in summary

    trends = client.get_analytics_trends(interval="day")
    assert trends["interval"] == "day"
    assert "volume_trends" in trends
    assert "amount_trends" in trends
