"""Comprehensive API tests for ChequeSense FastAPI backend."""

import io
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.dependencies import MAX_FILE_SIZE_BYTES, get_db, get_upload_dir
from api.main import app
from src.database.connection import Base
from src.database.models import Cheque, ExtractedField, ProcessingRun, ValidationResult


@pytest.fixture(scope="function")
def api_test_client():
    """Sets up an isolated database and test client with dependency overrides."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    temp_dir = tempfile.TemporaryDirectory()
    temp_upload_path = Path(temp_dir.name)

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    def override_get_upload_dir():
        return temp_upload_path

    from api.dependencies import get_current_user
    from src.database.models import User


    def override_get_current_user():
        return User(
            id=1,
            username="admin_operator",
            email="admin@bank.com",
            role="ADMIN",
            is_active=True,
            hashed_password="hashed_secret",
        )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_upload_dir] = override_get_upload_dir
    app.dependency_overrides[get_current_user] = override_get_current_user

    with TestClient(app) as client:
        yield client, TestingSessionLocal


    app.dependency_overrides.clear()
    temp_dir.cleanup()
    Base.metadata.drop_all(bind=test_engine)


def create_synthetic_cheque_bytes(width: int = 800, height: int = 400, color: tuple = (255, 255, 255)) -> bytes:
    """Generates valid PNG image bytes for upload testing."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------------------------
# 1. Health Endpoint Tests
# ------------------------------------------------------------------------------

def test_health_endpoint(api_test_client):
    """Verifies GET /api/v1/health returns 200 OK with complete diagnostic status."""
    client, _ = api_test_client
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["version"] == "1.0.0"
    assert data["database"] == "connected"
    assert "pipeline" in data


def test_root_endpoint(api_test_client):
    """Verifies GET / returns service information."""
    client, _ = api_test_client
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ChequeSense API"
    assert data["health_check"] == "/api/v1/health"


# ------------------------------------------------------------------------------
# 2. Cheque Upload Endpoint Tests
# ------------------------------------------------------------------------------

def test_upload_cheque_success(api_test_client):
    """Verifies uploading a valid cheque image creates a database record and stores the file."""
    client, _ = api_test_client
    img_bytes = create_synthetic_cheque_bytes(width=1000, height=500)

    response = client.post(
        "/api/v1/cheques/upload",
        files={"file": ("cheque_sample.png", img_bytes, "image/png")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["id"] is not None
    assert data["cheque_identifier"].startswith("CHQ_")
    assert data["image_width"] == 1000
    assert data["image_height"] == 500
    assert len(data["image_hash"]) == 64
    assert data["status"] == "PROCESSED"
    assert Path(data["image_path"]).exists()


def test_upload_cheque_duplicate_detection(api_test_client):
    """Verifies that uploading the identical image bytes returns the existing cheque record."""
    client, _ = api_test_client
    img_bytes = create_synthetic_cheque_bytes()

    res1 = client.post(
        "/api/v1/cheques/upload",
        files={"file": ("first_upload.png", img_bytes, "image/png")},
    )
    assert res1.status_code == 201
    id1 = res1.json()["id"]

    # Upload exact duplicate
    res2 = client.post(
        "/api/v1/cheques/upload",
        files={"file": ("duplicate_upload.png", img_bytes, "image/png")},
    )
    assert res2.status_code == 201
    id2 = res2.json()["id"]
    assert id1 == id2


def test_upload_cheque_empty_file(api_test_client):
    """Verifies that an empty file is rejected with 400 Bad Request."""
    client, _ = api_test_client
    response = client.post(
        "/api/v1/cheques/upload",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_cheque_invalid_extension(api_test_client):
    """Verifies that unsupported file extensions are rejected with 400 Bad Request."""
    client, _ = api_test_client
    response = client.post(
        "/api/v1/cheques/upload",
        files={"file": ("malicious.exe", b"fake binary data", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "unsupported file extension" in response.json()["detail"].lower()


def test_upload_cheque_corrupted_image(api_test_client):
    """Verifies that non-image bytes named .png are rejected with 400 Bad Request."""
    client, _ = api_test_client
    response = client.post(
        "/api/v1/cheques/upload",
        files={"file": ("corrupted.png", b"this is not a valid image", "image/png")},
    )
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


# ------------------------------------------------------------------------------
# 3. Cheque Processing Endpoint Tests
# ------------------------------------------------------------------------------

def test_process_cheque_not_found(api_test_client):
    """Verifies processing a non-existent cheque ID returns 404 Not Found."""
    client, _ = api_test_client
    response = client.post("/api/v1/cheques/99999/process")
    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]


def test_process_cheque_success(api_test_client):
    """Verifies executing the full pipeline and review queue on an uploaded cheque."""
    client, _ = api_test_client

    # 1. Use an actual synthetic cheque image from Dataset or artifacts if available, or generate one
    test_img_path = Path("artifacts/pipeline/test_cheques/syn_canara_syn_0091.png")
    if test_img_path.exists():
        with open(test_img_path, "rb") as f:
            img_bytes = f.read()
    else:
        img_bytes = create_synthetic_cheque_bytes(1200, 600)

    upload_res = client.post(
        "/api/v1/cheques/upload",
        files={"file": ("test_cheque.png", img_bytes, "image/png")},
    )
    assert upload_res.status_code == 201
    cheque_id = upload_res.json()["id"]

    # 2. Invoke processing endpoint
    process_res = client.post(f"/api/v1/cheques/{cheque_id}/process")
    assert process_res.status_code == 200
    pdata = process_res.json()

    assert pdata["cheque_id"] == cheque_id
    assert pdata["status"] in ["PROCESSED", "VERIFIED", "REVIEW_REQUIRED", "INVALID"]
    assert "overall_confidence" in pdata
    assert "fields" in pdata
    assert "processing_time_ms" in pdata
    assert pdata["processing_time_ms"] >= 0.0


# ------------------------------------------------------------------------------
# 4. Cheque Details and Listing Tests
# ------------------------------------------------------------------------------

def test_get_cheque_by_id(api_test_client):
    """Verifies retrieving cheque details by primary key."""
    client, _ = api_test_client
    img_bytes = create_synthetic_cheque_bytes()
    upload_res = client.post(
        "/api/v1/cheques/upload",
        files={"file": ("sample.png", img_bytes, "image/png")},
    )
    cheque_id = upload_res.json()["id"]

    get_res = client.get(f"/api/v1/cheques/{cheque_id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["id"] == cheque_id
    assert "extracted_fields" in data
    assert "validation_results" in data
    assert "processing_runs" in data


def test_get_cheque_by_id_not_found(api_test_client):
    """Verifies GET /api/v1/cheques/{id} returns 404 for unknown IDs."""
    client, _ = api_test_client
    response = client.get("/api/v1/cheques/999999")
    assert response.status_code == 404


def test_list_cheques_pagination_and_filtering(api_test_client):
    """Verifies listing cheques with pagination and status filtering."""
    client, _ = api_test_client

    # Upload two distinct cheques
    for i in range(2):
        img_bytes = create_synthetic_cheque_bytes(color=(i * 10, i * 10, i * 10))
        client.post(
            "/api/v1/cheques/upload",
            files={"file": (f"sample_{i}.png", img_bytes, "image/png")},
        )

    list_res = client.get("/api/v1/cheques?skip=0&limit=10")
    assert list_res.status_code == 200
    data = list_res.json()
    assert data["total"] >= 2
    assert len(data["items"]) >= 2

    # Filter with non-existent status
    empty_res = client.get("/api/v1/cheques?status=NON_EXISTENT_STATUS")
    assert empty_res.status_code == 200
    assert empty_res.json()["total"] == 0


# ------------------------------------------------------------------------------
# 5. Analytics Endpoints Tests
# ------------------------------------------------------------------------------

def test_analytics_summary_endpoint(api_test_client):
    """Verifies GET /api/v1/analytics/summary returns executive metrics."""
    client, _ = api_test_client
    response = client.get("/api/v1/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "volume" in data
    assert "amounts" in data
    assert "reviews" in data
    assert "confidence" in data
    assert "validation" in data
    assert "latency" in data
    assert "fields" in data


def test_analytics_trends_endpoint(api_test_client):
    """Verifies GET /api/v1/analytics/trends returns temporal volume and amount trends."""
    client, _ = api_test_client
    response = client.get("/api/v1/analytics/trends?interval=day")
    assert response.status_code == 200
    data = response.json()
    assert data["interval"] == "day"
    assert "volume_trends" in data
    assert "amount_trends" in data

    # Invalid interval regex validation
    bad_interval = client.get("/api/v1/analytics/trends?interval=invalid_interval")
    assert bad_interval.status_code == 422
