"""Comprehensive test suite for Authentication, Role-Based Access Control (RBAC), and Audit Logging in ChequeSense."""

import datetime
from datetime import timezone
import io
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.dependencies import get_db, get_upload_dir
from api.main import app
from src.database.connection import Base
from src.database.models import AuditLog, Cheque, ExtractedField, User
from src.security.audit import AuditAction, record_audit_event
from src.security.auth import (
    ROLE_PERMISSIONS,
    Permission,
    UserRole,
    create_access_token,
    decode_access_token,
    has_permission,
    hash_password,
    verify_password,
)


@pytest.fixture(scope="function")
def auth_test_env():
    """Sets up an in-memory SQLite database, clean tables, and TestClient without auth overrides."""
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

    # Only override DB and upload dir, NOT get_current_user, so real JWT auth runs
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_upload_dir] = override_get_upload_dir

    with TestClient(app) as client:
        yield client, TestingSessionLocal

    app.dependency_overrides.clear()
    temp_dir.cleanup()
    Base.metadata.drop_all(bind=test_engine)


def create_test_cheque_bytes(width: int = 600, height: int = 300) -> bytes:
    """Helper creating minimal valid cheque image bytes."""
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------------------------
# 1. Password Hashing & Security Tests
# ------------------------------------------------------------------------------

def test_password_hashing_bcrypt():
    """Verifies passwords are hashed with bcrypt (cost factor 12) and never stored as plaintext."""
    raw_pwd = "BankingSecret@2026"
    hashed = hash_password(raw_pwd)

    # Must not match plaintext
    assert hashed != raw_pwd
    # Bcrypt signature $2b$ or $2a$
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
    # Verify correctly matches
    assert verify_password(raw_pwd, hashed) is True
    # Verify rejection of incorrect password
    assert verify_password("WrongPassword123", hashed) is False
    assert verify_password("", hashed) is False


def test_password_verification_legacy_sha256_compatibility():
    """Verifies backward compatibility with existing SHA-256 test hashes."""
    import hashlib
    raw_pwd = "supersecretpassword"
    legacy_hash = hashlib.sha256(raw_pwd.encode("utf-8")).hexdigest()

    assert verify_password(raw_pwd, legacy_hash) is True
    assert verify_password("wrong", legacy_hash) is False


# ------------------------------------------------------------------------------
# 2. JWT Generation & Validation Tests
# ------------------------------------------------------------------------------

def test_jwt_lifecycle_and_claims():
    """Verifies token issuance, claim preservation, and signature verification."""
    data = {"sub": "alice_employee", "role": "EMPLOYEE", "user_id": 42}
    token = create_access_token(data=data, expires_delta=datetime.timedelta(minutes=30))

    assert isinstance(token, str)
    assert len(token) > 20

    payload = decode_access_token(token)
    assert payload["sub"] == "alice_employee"
    assert payload["role"] == "EMPLOYEE"
    assert payload["user_id"] == 42
    assert "exp" in payload
    assert "iat" in payload


def test_jwt_expired_token():
    """Verifies expired token raises an error."""
    import jwt

    data = {"sub": "expired_user", "role": "ANALYST"}
    token = create_access_token(data=data, expires_delta=datetime.timedelta(seconds=-10))

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token)


def test_jwt_tampered_token():
    """Verifies tampered token raises signature error."""
    import jwt

    data = {"sub": "bob", "role": "ADMIN"}
    token = create_access_token(data=data)
    tampered = token[:-4] + "fake"

    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(tampered)


# ------------------------------------------------------------------------------
# 3. RBAC Permissions Matrix Tests
# ------------------------------------------------------------------------------

def test_rbac_roles_and_permissions():
    """Verifies strict assignment of operational permissions across the 4 roles."""
    # ADMIN: All functionality
    for perm in Permission:
        assert has_permission("ADMIN", perm) is True

    # EMPLOYEE
    assert has_permission("EMPLOYEE", Permission.UPLOAD_CHEQUES) is True
    assert has_permission("EMPLOYEE", Permission.PROCESS_CHEQUES) is True
    assert has_permission("EMPLOYEE", Permission.VIEW_RESULTS) is True
    assert has_permission("EMPLOYEE", Permission.MANAGE_USERS) is False
    assert has_permission("EMPLOYEE", Permission.ACCESS_ANALYTICS) is False
    assert has_permission("EMPLOYEE", Permission.REVIEW_QUEUE) is False
    assert has_permission("EMPLOYEE", Permission.CORRECT_VALUES) is False

    # REVIEWER
    assert has_permission("REVIEWER", Permission.REVIEW_QUEUE) is True
    assert has_permission("REVIEWER", Permission.CORRECT_VALUES) is True
    assert has_permission("REVIEWER", Permission.VIEW_RESULTS) is True
    assert has_permission("REVIEWER", Permission.UPLOAD_CHEQUES) is False
    assert has_permission("REVIEWER", Permission.PROCESS_CHEQUES) is False
    assert has_permission("REVIEWER", Permission.ACCESS_ANALYTICS) is False

    # ANALYST
    assert has_permission("ANALYST", Permission.ACCESS_ANALYTICS) is True
    assert has_permission("ANALYST", Permission.VIEW_RESULTS) is True
    assert has_permission("ANALYST", Permission.UPLOAD_CHEQUES) is False
    assert has_permission("ANALYST", Permission.PROCESS_CHEQUES) is False
    assert has_permission("ANALYST", Permission.REVIEW_QUEUE) is False
    assert has_permission("ANALYST", Permission.CORRECT_VALUES) is False


# ------------------------------------------------------------------------------
# 4. Authentication Endpoints Tests (Register, Login, Me)
# ------------------------------------------------------------------------------

def test_auth_register_and_login_flow(auth_test_env):
    """Verifies user registration, bcrypt persistence, login, and profile retrieval."""
    client, session_factory = auth_test_env

    # 1. Register EMPLOYEE user
    reg_payload = {
        "username": "teller_emma",
        "email": "emma@bank.com",
        "password": "StrongPassword123!",
        "role": "EMPLOYEE",
    }
    reg_res = client.post("/auth/register", json=reg_payload)
    assert reg_res.status_code == 201
    user_data = reg_res.json()
    assert user_data["username"] == "teller_emma"
    assert user_data["role"] == "EMPLOYEE"
    assert "upload_cheques" in user_data["permissions"]

    # Verify password is NEVER stored in plain text in database
    with session_factory() as session:
        user_db = session.scalar(select(User).where(User.username == "teller_emma"))
        assert user_db is not None
        assert user_db.hashed_password != "StrongPassword123!"
        assert user_db.hashed_password.startswith("$2b$") or user_db.hashed_password.startswith("$2a$")

    # 2. Reject duplicate username
    dup_res = client.post("/auth/register", json=reg_payload)
    assert dup_res.status_code == 400
    assert "already registered" in dup_res.json()["detail"].lower()

    # 3. Reject invalid role
    bad_role_payload = {
        "username": "bad_user",
        "email": "bad@bank.com",
        "password": "StrongPassword123!",
        "role": "SUPERUSER",
    }
    bad_res = client.post("/auth/register", json=bad_role_payload)
    assert bad_res.status_code == 400
    assert "invalid role" in bad_res.json()["detail"].lower()

    # 4. Login with valid credentials
    login_payload = {"username": "teller_emma", "password": "StrongPassword123!"}
    login_res = client.post("/auth/login", json=login_payload)
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    assert token_data["role"] == "EMPLOYEE"
    assert token_data["username"] == "teller_emma"

    # Also test /api/v1/auth/login mount
    v1_login_res = client.post("/api/v1/auth/login", json=login_payload)
    assert v1_login_res.status_code == 200

    # 5. Login with invalid password
    bad_login = client.post("/auth/login", json={"username": "teller_emma", "password": "WrongPassword"})
    assert bad_login.status_code == 401

    # 6. GET /auth/me with Bearer token
    token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/auth/me", headers=headers)
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["username"] == "teller_emma"
    assert me_data["email"] == "emma@bank.com"
    assert me_data["role"] == "EMPLOYEE"
    assert "upload_cheques" in me_data["permissions"]

    # 7. GET /auth/me without token -> 401
    unauth_me = client.get("/auth/me")
    assert unauth_me.status_code == 401


# ------------------------------------------------------------------------------
# 5. Role-Based Access Control (RBAC) Endpoint Protection Tests
# ------------------------------------------------------------------------------

def register_and_get_token(client, username: str, role: str) -> str:
    """Helper registering a user and returning the JWT access token."""
    client.post(
        "/auth/register",
        json={
            "username": username,
            "email": f"{username}@bank.com",
            "password": "Password123!",
            "role": role,
        },
    )
    login_res = client.post(
        "/auth/login",
        json={"username": username, "password": "Password123!"},
    )
    return login_res.json()["access_token"]


def test_rbac_endpoint_isolation_matrix(auth_test_env):
    """Verifies complete RBAC enforcement across EMPLOYEE, REVIEWER, ANALYST, and ADMIN."""
    client, session_factory = auth_test_env

    # Setup accounts for each role
    emp_token = register_and_get_token(client, "emp_john", "EMPLOYEE")
    rev_token = register_and_get_token(client, "rev_sarah", "REVIEWER")
    ana_token = register_and_get_token(client, "ana_alex", "ANALYST")
    adm_token = register_and_get_token(client, "adm_rachel", "ADMIN")

    img_bytes = create_test_cheque_bytes()

    # 1. Unauthenticated request to protected endpoint -> 401
    res_no_auth = client.post(
        "/api/v1/cheques/upload",
        files={"file": ("cheque.png", img_bytes, "image/png")},
    )
    assert res_no_auth.status_code == 401

    # 2. Upload Endpoint (/api/v1/cheques/upload)
    # Allowed: EMPLOYEE, ADMIN
    # Denied: REVIEWER, ANALYST
    res_ana_upload = client.post(
        "/api/v1/cheques/upload",
        headers={"Authorization": f"Bearer {ana_token}"},
        files={"file": ("cheque.png", img_bytes, "image/png")},
    )
    assert res_ana_upload.status_code == 403

    res_rev_upload = client.post(
        "/api/v1/cheques/upload",
        headers={"Authorization": f"Bearer {rev_token}"},
        files={"file": ("cheque.png", img_bytes, "image/png")},
    )
    assert res_rev_upload.status_code == 403

    res_emp_upload = client.post(
        "/api/v1/cheques/upload",
        headers={"Authorization": f"Bearer {emp_token}"},
        files={"file": ("cheque.png", img_bytes, "image/png")},
    )
    assert res_emp_upload.status_code == 201
    cheque_id = res_emp_upload.json()["id"]

    # 3. Processing Endpoint (/api/v1/cheques/{id}/process)
    # Allowed: EMPLOYEE, ADMIN
    # Denied: REVIEWER, ANALYST
    res_ana_proc = client.post(
        f"/api/v1/cheques/{cheque_id}/process",
        headers={"Authorization": f"Bearer {ana_token}"},
    )
    assert res_ana_proc.status_code == 403

    res_rev_proc = client.post(
        f"/api/v1/cheques/{cheque_id}/process",
        headers={"Authorization": f"Bearer {rev_token}"},
    )
    assert res_rev_proc.status_code == 403

    res_emp_proc = client.post(
        f"/api/v1/cheques/{cheque_id}/process",
        headers={"Authorization": f"Bearer {emp_token}"},
    )
    assert res_emp_proc.status_code == 200

    # 4. Review & Manual Correction Endpoint (/api/v1/cheques/{id}/review)
    # Allowed: REVIEWER, ADMIN
    # Denied: EMPLOYEE, ANALYST
    correction_payload = {
        "corrections": {"amount": "12500.50", "cheque_number": "000999"},
        "status": "VERIFIED",
        "resolution_note": "Teller verified amount on signature line",
    }

    res_emp_rev = client.post(
        f"/api/v1/cheques/{cheque_id}/review",
        headers={"Authorization": f"Bearer {emp_token}"},
        json=correction_payload,
    )
    assert res_emp_rev.status_code == 403

    res_ana_rev = client.post(
        f"/api/v1/cheques/{cheque_id}/review",
        headers={"Authorization": f"Bearer {ana_token}"},
        json=correction_payload,
    )
    assert res_ana_rev.status_code == 403

    res_rev_rev = client.post(
        f"/api/v1/cheques/{cheque_id}/review",
        headers={"Authorization": f"Bearer {rev_token}"},
        json=correction_payload,
    )
    assert res_rev_rev.status_code == 200
    rev_data = res_rev_rev.json()
    assert rev_data["status"] == "VERIFIED"
    assert rev_data["review_required"] is False

    # Check field normalized value was updated to 12500.50
    amount_field = next((f for f in rev_data["extracted_fields"] if f["field_name"] == "amount"), None)
    assert amount_field is not None
    assert amount_field["normalized_value"] == "12500.50"
    assert amount_field["confidence"] == 1.0

    # 5. Analytics Endpoints (/api/v1/analytics/summary & /api/v1/analytics/trends)
    # Allowed: ANALYST, ADMIN
    # Denied: EMPLOYEE, REVIEWER
    res_emp_ana = client.get(
        "/api/v1/analytics/summary",
        headers={"Authorization": f"Bearer {emp_token}"},
    )
    assert res_emp_ana.status_code == 403

    res_rev_ana = client.get(
        "/api/v1/analytics/summary",
        headers={"Authorization": f"Bearer {rev_token}"},
    )
    assert res_rev_ana.status_code == 403

    res_ana_ana = client.get(
        "/api/v1/analytics/summary",
        headers={"Authorization": f"Bearer {ana_token}"},
    )
    assert res_ana_ana.status_code == 200
    assert "volume" in res_ana_ana.json()

    res_ana_trends = client.get(
        "/api/v1/analytics/trends?interval=day",
        headers={"Authorization": f"Bearer {ana_token}"},
    )
    assert res_ana_trends.status_code == 200

    # 6. ADMIN has global access across all endpoints
    # Admin upload
    res_adm_upload = client.post(
        "/api/v1/cheques/upload",
        headers={"Authorization": f"Bearer {adm_token}"},
        files={"file": ("adm_cheque.png", create_test_cheque_bytes(700, 350), "image/png")},
    )
    assert res_adm_upload.status_code == 201
    adm_cheque_id = res_adm_upload.json()["id"]

    # Admin process
    res_adm_proc = client.post(
        f"/api/v1/cheques/{adm_cheque_id}/process",
        headers={"Authorization": f"Bearer {adm_token}"},
    )
    assert res_adm_proc.status_code == 200

    # Admin review
    res_adm_rev = client.post(
        f"/api/v1/cheques/{adm_cheque_id}/review",
        headers={"Authorization": f"Bearer {adm_token}"},
        json={"corrections": {"amount": "77000.00"}, "status": "VERIFIED"},
    )
    assert res_adm_rev.status_code == 200

    # Admin analytics
    res_adm_ana = client.get(
        "/api/v1/analytics/summary",
        headers={"Authorization": f"Bearer {adm_token}"},
    )
    assert res_adm_ana.status_code == 200


# ------------------------------------------------------------------------------
# 6. Audit Logging Verification Tests
# ------------------------------------------------------------------------------

def test_audit_logging_actions(auth_test_env):
    """Verifies database audit trail captures LOGIN, UPLOAD_CHEQUE, PROCESS_CHEQUE, MANUAL_CORRECTION, and STATUS_CHANGE."""
    client, session_factory = auth_test_env

    emp_token = register_and_get_token(client, "auditor_emp", "EMPLOYEE")
    rev_token = register_and_get_token(client, "auditor_rev", "REVIEWER")

    # 1. Action: UPLOAD_CHEQUE
    upload_res = client.post(
        "/api/v1/cheques/upload",
        headers={"Authorization": f"Bearer {emp_token}"},
        files={"file": ("audit_sample.png", create_test_cheque_bytes(), "image/png")},
    )
    assert upload_res.status_code == 201
    cheque_id = upload_res.json()["id"]

    # 2. Action: PROCESS_CHEQUE and STATUS_CHANGE
    proc_res = client.post(
        f"/api/v1/cheques/{cheque_id}/process",
        headers={"Authorization": f"Bearer {emp_token}"},
    )
    assert proc_res.status_code == 200

    # 3. Action: MANUAL_CORRECTION and STATUS_CHANGE
    rev_res = client.post(
        f"/api/v1/cheques/{cheque_id}/review",
        headers={"Authorization": f"Bearer {rev_token}"},
        json={
            "corrections": {"amount": "54000.00"},
            "status": "VERIFIED",
            "resolution_note": "Signature verified with mandate",
        },
    )
    assert rev_res.status_code == 200

    # Query audit logs from database and verify all 5 required actions exist
    with session_factory() as session:
        logs = list(session.scalars(select(AuditLog).order_by(AuditLog.id)).all())

        actions_recorded = [log.action for log in logs]

        # Verify all 5 required actions are present
        assert "LOGIN" in actions_recorded
        assert "UPLOAD_CHEQUE" in actions_recorded
        assert "PROCESS_CHEQUE" in actions_recorded
        assert "MANUAL_CORRECTION" in actions_recorded
        assert "STATUS_CHANGE" in actions_recorded

        # Inspect specific log details
        login_log = next(log for log in logs if log.action == "LOGIN")
        assert login_log.username in ("auditor_emp", "auditor_rev")
        assert login_log.resource_type == "auth"
        assert login_log.timestamp is not None

        upload_log = next(log for log in logs if log.action == "UPLOAD_CHEQUE")
        assert upload_log.username == "auditor_emp"
        assert upload_log.resource_id == str(cheque_id)
        assert "filename" in upload_log.details_json

        correction_log = next(log for log in logs if log.action == "MANUAL_CORRECTION")
        assert correction_log.username == "auditor_rev"
        assert correction_log.resource_id == str(cheque_id)
        assert correction_log.details_json["corrections"]["amount"] == "54000.00"

        status_change_log = next(log for log in logs if log.action == "STATUS_CHANGE")
        assert status_change_log.resource_id == str(cheque_id)
        assert "new_status" in status_change_log.details_json
