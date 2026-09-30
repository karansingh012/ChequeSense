# ChequeSense — Security Documentation

## Overview

ChequeSense implements defence-in-depth security appropriate for an internal banking tool. This document describes the authentication mechanism, authorisation model, audit logging, credential management, and security constraints.

> **Note:** This system is designed as a back-office tool, not a public-facing banking portal. Production deployment behind an internal firewall, VPN, or private network is strongly recommended.

---

## Authentication

### Mechanism: JWT (JSON Web Tokens)

ChequeSense uses stateless JWT-based authentication implemented with `PyJWT`.

- Token type: Bearer
- Signing algorithm: HS256 (HMAC-SHA256)
- Secret key: Loaded from `SECRET_KEY` environment variable (minimum 32 characters)
- Token expiry: Configurable via `ACCESS_TOKEN_EXPIRE_MINUTES` (default: 60 minutes)
- Tokens are never stored server-side (fully stateless)

### Token Lifecycle

```
[User] POST /auth/login (username + password)
    │
    ▼
[Server] Verify username exists in DB
         Verify bcrypt hash matches stored hash
         Issue JWT signed with SECRET_KEY
    │
    ▼
[User] Receives: { "access_token": "eyJ...", "token_type": "bearer" }

[User] Subsequent requests:
    Authorization: Bearer eyJ...
    │
    ▼
[Server] Decode + verify JWT signature
         Check expiry
         Load user from DB
         Check role permissions
```

### Token Claims

| Claim | Value |
|---|---|
| `sub` | Username |
| `role` | User role string |
| `exp` | Expiry timestamp |
| `iat` | Issued-at timestamp |

---

## Password Security

- **Algorithm:** bcrypt via `passlib[bcrypt]`
- **Work factor:** Default bcrypt cost (12 rounds)
- **Storage:** Only the bcrypt hash is stored in the `users.hashed_password` column
- **Plaintext:** Plaintext passwords are never stored, logged, or returned in any API response
- **Validation:** Minimum password length enforced at registration

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# On registration
hashed = pwd_context.hash(plaintext_password)

# On login
is_valid = pwd_context.verify(plaintext_password, stored_hash)
```

---

## Role-Based Access Control (RBAC)

### Roles

| Role | Description |
|---|---|
| `ADMIN` | Full access to all system functions including user management |
| `EMPLOYEE` | Upload and process cheques; view own results |
| `REVIEWER` | Access the review queue; submit manual corrections |
| `ANALYST` | Access analytics and summary reports |

### Permissions Matrix

| Endpoint | ADMIN | EMPLOYEE | REVIEWER | ANALYST |
|---|---|---|---|---|
| `POST /auth/register` | ✓ | ✓ | ✓ | ✓ |
| `POST /auth/login` | ✓ | ✓ | ✓ | ✓ |
| `GET /auth/me` | ✓ | ✓ | ✓ | ✓ |
| `POST /cheques/upload` | ✓ | ✓ | ✗ | ✗ |
| `POST /cheques/{id}/process` | ✓ | ✓ | ✗ | ✗ |
| `GET /cheques/{id}` | ✓ | ✓ | ✓ | ✗ |
| `GET /cheques` | ✓ | ✓ | ✗ | ✗ |
| `PATCH /cheques/{id}/correct` | ✓ | ✗ | ✓ | ✗ |
| `GET /analytics/summary` | ✓ | ✗ | ✗ | ✓ |
| `GET /analytics/trends` | ✓ | ✗ | ✗ | ✓ |
| `GET /health` | ✓ | ✓ | ✓ | ✓ |

### Implementation

Role checks are enforced via FastAPI dependencies:

```python
def require_roles(*roles: UserRole):
    def dependency(current_user: User = Depends(get_current_user)):
        if current_user.role not in [r.value for r in roles]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return dependency

# Usage in route
@router.post("/cheques/upload")
def upload(
    ...,
    current_user: User = Depends(require_roles(UserRole.EMPLOYEE, UserRole.ADMIN))
):
```

---

## Audit Logging

Every significant action is recorded in the `audit_logs` table with the following information:

| Column | Description |
|---|---|
| `timestamp` | UTC timestamp of the action |
| `user_id` | FK to `users` table (nullable for system actions) |
| `username` | Denormalised for immutability even if user is deleted |
| `action` | Action type (see below) |
| `resource_type` | `cheque`, `user`, `validation_result`, `auth` |
| `resource_id` | ID of the affected resource |
| `details_json` | Action-specific structured metadata |
| `ip_address` | Client IP address (IPv4 or IPv6) |

### Audited Action Types

| Action | Trigger |
|---|---|
| `LOGIN` | Successful login |
| `LOGIN_FAILED` | Failed login attempt |
| `USER_REGISTERED` | New user registration |
| `UPLOAD_CHEQUE` | Cheque image uploaded |
| `PROCESS_CHEQUE` | Pipeline run initiated |
| `MANUAL_CORRECTION` | Reviewer corrects a field |
| `STATUS_CHANGE` | Cheque status updated |

### Immutability

Audit log rows are never updated or deleted by the application. The `AuditLog` model has no `UPDATE` or `DELETE` ORM operations. In production, consider applying PostgreSQL row-level security to prevent audit log tampering.

---

## Credential Management

### Rules

1. **No plaintext secrets in code:** All secrets are environment variables.
2. **No secrets in Docker images:** `.env` is excluded by `.dockerignore`.
3. **No secrets in git history:** `.env` is listed in `.gitignore`.
4. **Template provided:** `.env.example` contains all required variable names with placeholder values.

### Required Environment Variables

| Variable | Description | Example |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@db:5432/chequesense` |
| `SECRET_KEY` | JWT signing key | 64-char random string |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token TTL | `60` |

### Generating a Secure Secret Key

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## File Upload Security

| Control | Implementation |
|---|---|
| MIME type validation | Only `image/jpeg`, `image/png`, `image/tiff` accepted |
| File size limit | 10 MB maximum (configurable) |
| Filename sanitisation | UUID v4 + extension; original filename discarded |
| Content validation | PIL `Image.verify()` called on every upload |
| Storage | Files stored outside web root in `UPLOAD_DIR` |
| Deduplication | SHA-256 hash prevents duplicate processing |

---

## Known Limitations and Recommendations for Production

1. **No refresh tokens:** The current implementation uses single-use access tokens. Implement refresh tokens for production.
2. **No account lockout:** Failed login attempts are logged but not rate-limited. Add login rate limiting (e.g., via a reverse proxy or Redis-backed rate limiter).
3. **No HTTPS enforcement:** HTTPS must be configured at the reverse proxy (nginx/Traefik) layer.
4. **Self-registration:** Any user can register with any role. In production, restrict registration to admins or implement an invite-based flow.
5. **No secret rotation:** Implement `SECRET_KEY` rotation with a token invalidation mechanism for production.
6. **Audit log write-once enforcement:** Consider PostgreSQL RLS or a dedicated audit log service.

---

## Security Summary

| Control | Status |
|---|---|
| Password hashing (bcrypt) | ✓ Implemented |
| JWT authentication | ✓ Implemented |
| Role-based access control | ✓ Implemented |
| Audit logging | ✓ Implemented |
| File upload validation | ✓ Implemented |
| No secrets in codebase | ✓ Enforced via .gitignore / .dockerignore |
| HTTPS | ✗ Configured at reverse proxy layer (not in app) |
| Account lockout / rate limiting | ✗ Not implemented |
| Refresh tokens | ✗ Not implemented |
| Signature verification | ✗ Field presence only |
