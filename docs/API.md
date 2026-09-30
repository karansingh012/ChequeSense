# ChequeSense — API Documentation

## Overview

The ChequeSense REST API is built with **FastAPI** and exposes endpoints for authentication, cheque management, analytics, and health checks. Interactive documentation is available at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## Base URL

```
http://localhost:8000
```

---

## Authentication

All protected endpoints require a `Bearer` token in the `Authorization` header:

```
Authorization: Bearer <access_token>
```

Tokens are obtained via `POST /auth/login`. Tokens are JWTs signed with the server's `SECRET_KEY` and expire after a configurable duration.

---

## Endpoints

### Authentication

#### `POST /auth/register`

Register a new user account.

**Request body:**

```json
{
  "username": "john_teller",
  "email": "john@bank.com",
  "password": "SecureP@ss123",
  "role": "EMPLOYEE"
}
```

**Response `201`:**

```json
{
  "id": 1,
  "username": "john_teller",
  "email": "john@bank.com",
  "role": "EMPLOYEE",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00Z"
}
```

---

#### `POST /auth/login`

Authenticate and receive a JWT access token.

**Request body (form data):**

```
username=john_teller
password=SecureP@ss123
```

**Response `200`:**

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGci...",
  "token_type": "bearer"
}
```

---

#### `GET /auth/me`

Return the profile of the currently authenticated user.

**Response `200`:**

```json
{
  "id": 1,
  "username": "john_teller",
  "email": "john@bank.com",
  "role": "EMPLOYEE",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00Z"
}
```

---

### Cheque Endpoints

#### `POST /api/v1/cheques/upload`

Upload a new cheque image and register it in the database.

**Required role:** `EMPLOYEE`, `ADMIN`

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | Cheque image (JPEG, PNG, TIFF) |

**Constraints:**
- Allowed MIME types: `image/jpeg`, `image/png`, `image/tiff`
- Maximum file size: 10 MB
- Duplicate detection: SHA-256 hash; returns existing record if already uploaded

**Response `201`:**

```json
{
  "cheque_id": 42,
  "cheque_identifier": "CHQ-2024011500001",
  "status": "PROCESSED",
  "image_path": "data/uploads/3f9a2c1d-....jpg",
  "created_at": "2024-01-15T10:35:00Z"
}
```

**Errors:**

| Status | Description |
|---|---|
| `400` | Unsupported file type or corrupted image |
| `413` | File exceeds 10 MB limit |
| `409` | Duplicate cheque (same SHA-256 hash) |
| `401` | Missing or invalid token |
| `403` | Insufficient role |

---

#### `POST /api/v1/cheques/{cheque_id}/process`

Run the ChequeSense ML inference pipeline on an uploaded cheque.

**Required role:** `EMPLOYEE`, `ADMIN`

**Path parameters:**

| Parameter | Type | Description |
|---|---|---|
| `cheque_id` | integer | Database ID of the cheque |

**Response `200`:**

```json
{
  "cheque_id": 42,
  "cheque_identifier": "CHQ-2024011500001",
  "status": "REVIEW_REQUIRED",
  "overall_confidence": 0.31,
  "fields": {
    "date": {
      "value": "15/01/2024",
      "raw_value": "15/01/2024",
      "confidence": 0.72,
      "detection_confidence": 0.94,
      "extraction_confidence": 0.77,
      "confidence_tier": "MEDIUM",
      "validation_status": "VALID"
    },
    "amount": {
      "value": "",
      "raw_value": "",
      "confidence": 0.0,
      "detection_confidence": 0.88,
      "extraction_confidence": 0.0,
      "confidence_tier": "LOW",
      "validation_status": "INVALID",
      "validation_reason": "[AMOUNT] Amount field is empty"
    }
  },
  "processing_time_ms": 712.4
}
```

---

#### `GET /api/v1/cheques/{cheque_id}`

Retrieve full cheque details including all extracted fields and validation results.

**Required role:** `EMPLOYEE`, `REVIEWER`, `ADMIN`

**Response `200`:**

```json
{
  "cheque_id": 42,
  "cheque_identifier": "CHQ-2024011500001",
  "status": "REVIEW_REQUIRED",
  "overall_confidence": 0.31,
  "image_path": "data/uploads/3f9a2c1d-....jpg",
  "created_at": "2024-01-15T10:35:00Z",
  "fields": { ... },
  "validation_results": [ ... ],
  "processing_runs": [ ... ]
}
```

---

#### `GET /api/v1/cheques`

List cheques with pagination and optional filters.

**Required role:** `EMPLOYEE`, `ADMIN`

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `page` | int | 1 | Page number |
| `page_size` | int | 20 | Items per page (max 100) |
| `status` | string | — | Filter by status |
| `review_required` | bool | — | Filter by review flag |

**Response `200`:**

```json
{
  "total": 150,
  "page": 1,
  "page_size": 20,
  "cheques": [ ... ]
}
```

---

#### `PATCH /api/v1/cheques/{cheque_id}/correct`

Submit a manual field correction for a cheque in `REVIEW_REQUIRED` state.

**Required role:** `REVIEWER`, `ADMIN`

**Request body:**

```json
{
  "field_name": "amount",
  "corrected_value": "25000.00",
  "correction_note": "Handwriting was unclear; value confirmed from physical cheque"
}
```

**Response `200`:**

```json
{
  "cheque_id": 42,
  "field_name": "amount",
  "corrected_value": "25000.00",
  "review_resolution": "CORRECTED",
  "reviewed_by": "supervisor_user",
  "reviewed_at": "2024-01-15T11:00:00Z"
}
```

---

### Analytics Endpoints

#### `GET /api/v1/analytics/summary`

Return aggregate system metrics.

**Required role:** `ANALYST`, `ADMIN`

**Response `200`:**

```json
{
  "total_cheques": 1500,
  "verified_count": 320,
  "review_required_count": 890,
  "invalid_count": 290,
  "verification_rate": 0.213,
  "review_rate": 0.593,
  "average_confidence": 0.29,
  "average_processing_time_ms": 692.3
}
```

---

#### `GET /api/v1/analytics/trends`

Return processing volume grouped by time period.

**Required role:** `ANALYST`, `ADMIN`

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | string | `day` | Grouping: `hour`, `day`, `week`, `month` |
| `days` | int | 30 | Number of days to look back |

**Response `200`:**

```json
{
  "period": "day",
  "data": [
    { "date": "2024-01-14", "count": 45, "verified": 12, "review_required": 28, "invalid": 5 },
    { "date": "2024-01-15", "count": 38, "verified": 9, "review_required": 25, "invalid": 4 }
  ]
}
```

---

### Health

#### `GET /api/v1/health`

Service liveness check.

**Response `200`:**

```json
{
  "status": "healthy",
  "service": "ChequeSense API",
  "version": "1.0.0",
  "database": "connected",
  "pipeline": "loaded"
}
```

---

## Error Response Format

All error responses follow a consistent schema:

```json
{
  "detail": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE"
}
```

---

## Rate Limiting

Rate limiting is not implemented in the current version. For production deployment behind a reverse proxy (nginx, Traefik), configure rate limiting at the proxy layer.

---

## Pagination

All list endpoints use cursor-free offset pagination:

```
GET /api/v1/cheques?page=2&page_size=20
```

The response includes `total`, `page`, and `page_size` for client-side navigation.

---

## File Upload Constraints

| Constraint | Value |
|---|---|
| Allowed MIME types | `image/jpeg`, `image/png`, `image/tiff` |
| Maximum file size | 10 MB |
| Filename sanitisation | UUID v4 + original extension |
| Deduplication | SHA-256 content hash |

---

## Environment Variables

| Variable | Description |
|---|---|
| `SECRET_KEY` | JWT signing key (minimum 32 characters, keep secret) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime in minutes (default: 60) |
| `DATABASE_URL` | PostgreSQL connection string |
| `UPLOAD_DIR` | Filesystem directory for uploaded images |
| `MAX_FILE_SIZE_MB` | Upload size limit in MB (default: 10) |

See [`.env.example`](../.env.example) for the complete list.
