# ChequeSense Security Architecture & Role-Based Access Control (RBAC)

## 1. Executive Summary

ChequeSense is an enterprise AI-powered cheque processing and verification system designed for automated clearinghouse (CTS/NACH) operations. Financial transactions and negotiable instruments require strict adherence to the **Principle of Least Privilege (PoLP)**, zero-trust perimeter enforcement, immutable compliance audit trails, and cryptographic data protection.

This document details the security model, cryptographic password hashing, JSON Web Token (JWT) lifecycle, Role-Based Access Control (RBAC) matrix, and audit logging standards implemented in the system.

---

## 2. Authentication Architecture

### 2.1 Password Security & Bcrypt Hashing
ChequeSense **never stores plaintext passwords**. All user credentials submitted during registration are salted and hashed using **bcrypt** with an adaptive cost factor (work factor) of 12 rounds:

- **Salt Generation**: Cryptographically secure pseudo-random salt generated via `bcrypt.gensalt(rounds=12)`.
- **Pre-image & Rainbow Table Resistance**: Unique 128-bit salt prevents pre-computation and dictionary attacks.
- **Timing Attack Resistance**: Constant-time comparison via `bcrypt.checkpw()`.
- **Legacy Migration Support**: The authentication layer transparently supports backward-compatible verification for existing test fixtures while enforcing bcrypt for all new and updated user credentials.

```python
# src/security/auth.py
def hash_password(plain_password: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if hashed_password.startswith("$2b$") or hashed_password.startswith("$2a$"):
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    # Fallback for legacy SHA-256 fixture hashes
    if len(hashed_password) == 64:
        return hashlib.sha256(plain_password.encode("utf-8")).hexdigest() == hashed_password
    return False
```

### 2.2 JWT-Based Stateless Authentication
Session management is handled through signed **JSON Web Tokens (JWT)** compliant with RFC 7519:

- **Signature Algorithm**: HMAC-SHA256 (`HS256`).
- **Secret Key**: Injected at runtime via the `JWT_SECRET_KEY` environment variable (minimum 256-bit entropy).
- **Expiration Policy**: Configurable via `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` (defaults to 1440 minutes / 24 hours).
- **Token Claims**:
  - `sub`: Username subject identifier.
  - `user_id`: Database primary key ID.
  - `role`: Assigned banking operational role.
  - `iat`: Timestamp of issuance (epoch seconds).
  - `exp`: Timestamp of token expiration (epoch seconds).

```json
{
  "sub": "sarah_reviewer",
  "user_id": 14,
  "role": "REVIEWER",
  "iat": 1759276800,
  "exp": 1759363200
}
```

---

## 3. Role-Based Access Control (RBAC)

### 3.1 Operational Banking Roles

ChequeSense partitions system capabilities across four specialized operational roles:

| Role | Operational Scope | Description |
| :--- | :--- | :--- |
| **`ADMIN`** | System Administration & Governance | Complete administrative authority. Manages user provisioning, configures system thresholds, reviews queues, and accesses all system endpoints. |
| **`EMPLOYEE`** | Front-Office Teller / Ingestion | Ingests branch or batch cheque images, triggers inference processing pipelines, and views standard processing outputs. |
| **`REVIEWER`** | Back-Office Compliance & Verification | Specializes in the human review queue. Inspects low-confidence or validation-flagged cheques, performs manual corrections, and updates ledger statuses. |
| **`ANALYST`** | Operations Analytics & MIS | Accesses business intelligence dashboards, clearing volume statistics, trend distributions, latency metrics, and executive reporting. |

### 3.2 Granular Permissions Matrix

| Permission | Description | `ADMIN` | `EMPLOYEE` | `REVIEWER` | `ANALYST` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `manage_users` | Provision, suspend, or update banking user accounts | ✅ | ❌ | ❌ | ❌ |
| `upload_cheques` | Ingest cheque image files into the staging storage | ✅ | ✅ | ❌ | ❌ |
| `process_cheques` | Execute field detection, digit recognition, and OCR pipelines | ✅ | ✅ | ❌ | ❌ |
| `view_results` | Query cheque records, extracted fields, and processing runs | ✅ | ✅ | ✅ | ✅ |
| `review_queue` | Access cheques gated for manual human teller inspection | ✅ | ❌ | ✅ | ❌ |
| `correct_values` | Apply manual overrides to extracted field values & verify cheques | ✅ | ❌ | ✅ | ❌ |
| `access_analytics`| Query executive dashboards, volume, and monetary trends | ✅ | ❌ | ❌ | ✅ |

### 3.3 Endpoint Authorization Mapping

| HTTP Method | Route | Required Role(s) | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/auth/register` | Public (or `ADMIN`) | Creates new user account with bcrypt password |
| `POST` | `/auth/login` | Public | Authenticates credentials, issues JWT token, logs `LOGIN` |
| `GET` | `/auth/me` | Authenticated (Any) | Returns current user profile and resolved permissions |
| `POST` | `/api/v1/cheques/upload` | `EMPLOYEE`, `ADMIN` | Ingests cheque image, validates format/size, logs `UPLOAD_CHEQUE` |
| `POST` | `/api/v1/cheques/{id}/process` | `EMPLOYEE`, `ADMIN` | Runs ML inference pipeline, logs `PROCESS_CHEQUE`, `STATUS_CHANGE` |
| `GET` | `/api/v1/cheques/{id}` | Authenticated (Any) | Retrieves cheque details, extracted fields, validation gates |
| `GET` | `/api/v1/cheques` | Authenticated (Any) | Lists paginated cheques with status & review filtering |
| `POST` | `/api/v1/cheques/{id}/review` | `REVIEWER`, `ADMIN` | Corrects field values, resolves review, logs `MANUAL_CORRECTION` |
| `GET` | `/api/v1/analytics/summary` | `ANALYST`, `ADMIN` | Returns clearing volume, confidence, and latency summary |
| `GET` | `/api/v1/analytics/trends` | `ANALYST`, `ADMIN` | Returns time-series volume and monetary clearing trends |
| `GET` | `/api/v1/health` | Public | System liveness, database, and pipeline status |

---

## 4. Audit Logging Architecture

Banking regulations (e.g., PCI-DSS, RBI Information Security Guidelines, SOX) mandate an immutable, time-stamped audit trail for all security-sensitive actions and operational modifications.

### 4.1 Tracked Audit Events

1. **`LOGIN`**: Authenticated user session initiation with IP tracking and timestamp.
2. **`UPLOAD_CHEQUE`**: Physical image ingestion with SHA-256 cryptographic digest, file size, and filename.
3. **`PROCESS_CHEQUE`**: Machine learning pipeline execution pass, model latencies, and gating decisions.
4. **`MANUAL_CORRECTION`**: Human teller intervention updating extracted field values (record before/after).
5. **`STATUS_CHANGE`**: Transitions across lifecycle states (`PROCESSED` → `VERIFIED`, `REVIEW_REQUIRED`, `INVALID`).

### 4.2 Database Schema: `audit_logs`

```sql
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    username VARCHAR(64) NOT NULL,
    action VARCHAR(64) NOT NULL,
    resource_type VARCHAR(64) NOT NULL,
    resource_id VARCHAR(128),
    details_json JSONB,
    ip_address VARCHAR(45)
);

CREATE INDEX idx_audit_logs_action_timestamp ON audit_logs (action, timestamp);
CREATE INDEX ix_audit_logs_username ON audit_logs (username);
```

### 4.3 SIEM Integration
In addition to relational storage, every audit event is emitted as a structured, log-forwarding friendly string designed for ingestion by Security Information and Event Management (SIEM) systems (e.g., Splunk, Datadog, ELK, AWS CloudWatch):

```
2026-10-01 02:00:35,120 [INFO] chequesense.audit: AUDIT_EVENT: action=MANUAL_CORRECTION user=auditor_rev resource=cheque/12 ip=127.0.0.1 details={"corrections": {"amount": "54000.00"}, "resolution_note": "Signature verified with mandate"}
```

---

## 5. Defense-in-Depth & System Hardening

1. **Denial of Service (DoS) Prevention on Uploads**:
   - Strict 15 MB payload ceiling (`HTTP 413 Payload Too Large`).
   - Allowed file MIME types (`image/png`, `image/jpeg`, `image/tiff`, `image/webp`).
   - In-memory PIL header verification (`img.verify()`) to prevent image decompression bombs and corrupted file parsing.
2. **Path Traversal Protection**:
   - Client filenames are never used directly on disk.
   - Files are stored using cryptographic hashes and UUIDs: `{uuid}_{hash[:16]}.{ext}`.
3. **Deduplication by Cryptographic Digest**:
   - SHA-256 hashes detect re-uploads of identical cheques and prevent duplicate clearing transactions.
4. **Environment Secrets Isolation**:
   - Secrets are loaded exclusively from system environment variables (`JWT_SECRET_KEY`, `POSTGRES_PASSWORD`).
   - `.env` files are strictly excluded from source control via `.gitignore`.
   - `.env.example` provides template variables without exposing credentials.

---

## 6. Verification and Testing

Automated security and authorization tests are maintained under [`tests/test_auth_layer.py`](file:///Users/karansingh/ChequeSense/tests/test_auth_layer.py):

```bash
# Execute security and authentication test suite
python -m pytest tests/test_auth_layer.py -v

# Execute entire test suite
python -m pytest
```

### Test Coverage Highlights:
- **`test_password_hashing_bcrypt`**: Verifies bcrypt salting, round count, and rejection of invalid passwords.
- **`test_jwt_lifecycle_and_claims`**: Validates signature generation, expiration enforcement, and tampered token detection.
- **`test_rbac_roles_and_permissions`**: Validates the permission matrix across all 4 roles.
- **`test_auth_register_and_login_flow`**: Validates `/auth/register`, `/auth/login`, duplicate checks, and `/auth/me`.
- **`test_rbac_endpoint_isolation_matrix`**: Tests that unauthorized roles receive `HTTP 403 Forbidden` and unauthenticated calls receive `HTTP 401 Unauthorized` across upload, process, review, and analytics endpoints.
- **`test_audit_logging_actions`**: Confirms database records for `LOGIN`, `UPLOAD_CHEQUE`, `PROCESS_CHEQUE`, `MANUAL_CORRECTION`, and `STATUS_CHANGE`.
