# ChequeSense — Database Architecture

## Overview

ChequeSense uses **PostgreSQL 14+** as its primary persistence layer, accessed via **SQLAlchemy 2.0** ORM with the `mapped_column` / `Mapped` typed column syntax. Schema migrations are managed with **Alembic**.

---

## Design Principles

1. **No image binaries in the database.** Only file paths or object-storage URI references are stored.
2. **Preserve raw data.** Both `raw_value` (exact model output) and `normalized_value` (cleaned value) are stored for every extracted field.
3. **Preserve model provenance.** Every `ProcessingRun` records which model versions were used.
4. **Normalised schema.** Each entity has a single table. No denormalisation except the `username` column in `audit_logs` (for immutability).
5. **Environment-variable credentials.** No passwords in code or config files.

---

## Schema Diagram

```
users
  id, username, email, hashed_password, role, is_active, created_at, updated_at
  │
  ├─── cheques (created_by_user_id → users.id)
  │      id, cheque_identifier, image_path, image_hash,
  │      image_width, image_height, source_dataset,
  │      status, overall_confidence, review_required,
  │      created_by_user_id, created_at, updated_at
  │      │
  │      ├─── processing_runs (cheque_id → cheques.id)
  │      │      id, cheque_id, run_timestamp, status,
  │      │      detector_model_version, recognizer_model_version, ocr_engine_version,
  │      │      total_latency_ms, stage_latencies_json
  │      │      │
  │      │      ├─── predictions (processing_run_id, cheque_id)
  │      │      │      id, stage, model_name, model_version, field_name,
  │      │      │      raw_output, raw_confidence, bounding_box_json, created_at
  │      │      │
  │      │      ├─── extracted_fields (processing_run_id, cheque_id)
  │      │      │      id, field_name, raw_value, normalized_value,
  │      │      │      confidence, detection_confidence, extraction_confidence,
  │      │      │      extraction_method, confidence_tier, bounding_box_json, created_at
  │      │      │
  │      │      └─── validation_results (processing_run_id, cheque_id)
  │      │             id, field_name, check_type, validation_status, is_valid,
  │      │             validation_reason, review_priority,
  │      │             reviewed_by_user_id, review_resolution, reviewed_at, created_at
  │      │
  └─── audit_logs (user_id → users.id)
         id, timestamp, user_id, username, action,
         resource_type, resource_id, details_json, ip_address
```

---

## Table Definitions

### `users`

Banking system users, compliance officers, tellers, and service accounts.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PK, autoincrement | Internal user ID |
| `username` | VARCHAR(64) | UNIQUE, NOT NULL, INDEX | Login username |
| `email` | VARCHAR(255) | UNIQUE, NOT NULL, INDEX | Email address |
| `hashed_password` | VARCHAR(255) | NOT NULL | bcrypt hash |
| `role` | VARCHAR(32) | NOT NULL | ADMIN / EMPLOYEE / REVIEWER / ANALYST |
| `is_active` | BOOLEAN | NOT NULL, default TRUE | Account enabled flag |
| `created_at` | TIMESTAMP WITH TIME ZONE | NOT NULL | Creation time |
| `updated_at` | TIMESTAMP WITH TIME ZONE | NOT NULL | Last update time |

---

### `cheques`

Core cheque document entity.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PK | Internal cheque ID |
| `cheque_identifier` | VARCHAR(128) | UNIQUE, NOT NULL, INDEX | Business-level ID (e.g., CHQ-20240115-00001) |
| `image_path` | VARCHAR(512) | NOT NULL | Filesystem path or object-storage URI |
| `image_width` | INTEGER | NOT NULL | Image width in pixels |
| `image_height` | INTEGER | NOT NULL | Image height in pixels |
| `image_hash` | VARCHAR(64) | NOT NULL, INDEX | SHA-256 hex of image bytes (deduplication) |
| `source_dataset` | VARCHAR(64) | NOT NULL | Source dataset label or "production" |
| `status` | VARCHAR(32) | NOT NULL, INDEX | PROCESSED / VERIFIED / REVIEW_REQUIRED / INVALID |
| `overall_confidence` | FLOAT | NOT NULL | Mean composite confidence across fields |
| `review_required` | BOOLEAN | NOT NULL, INDEX | Quick filter flag |
| `created_by_user_id` | INTEGER | FK → users.id, nullable | Uploading user |
| `created_at` | TIMESTAMP WITH TIME ZONE | NOT NULL, INDEX | Upload timestamp |
| `updated_at` | TIMESTAMP WITH TIME ZONE | NOT NULL | Last update |

**Compound indexes:**
- `idx_cheques_status_review` on (`status`, `review_required`)
- `idx_cheques_created_at_status` on (`created_at`, `status`)

---

### `processing_runs`

Audit log of individual pipeline execution passes.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | |
| `cheque_id` | INTEGER FK | References `cheques.id` (CASCADE DELETE) |
| `run_timestamp` | TIMESTAMP WITH TIME ZONE | When the run started |
| `status` | VARCHAR(32) | SUCCESS / FAILED |
| `detector_model_version` | VARCHAR(128) | E.g., "faster_rcnn_resnet50_fpn_v1" |
| `recognizer_model_version` | VARCHAR(128) | E.g., "mnist_cnn_v1" |
| `ocr_engine_version` | VARCHAR(128) | E.g., "tesseract-5.3.3" |
| `total_latency_ms` | FLOAT | Total wall-clock time in milliseconds |
| `stage_latencies_json` | JSONB | Per-stage breakdown: preprocessing, detection, extraction |

---

### `predictions`

Raw model outputs before post-processing.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | |
| `processing_run_id` | INTEGER FK | References `processing_runs.id` |
| `cheque_id` | INTEGER FK | References `cheques.id` |
| `stage` | VARCHAR(64) | FIELD_DETECTION / OCR_EXTRACTION / HANDWRITTEN_RECOGNITION |
| `model_name` | VARCHAR(128) | Name of the model that produced this prediction |
| `model_version` | VARCHAR(64) | Version string |
| `field_name` | VARCHAR(64) | date / amount / ifsc / acno / sign / name |
| `raw_output` | TEXT | Verbatim model output (JSON string for detection, text for OCR) |
| `raw_confidence` | FLOAT | Model-reported confidence |
| `bounding_box_json` | JSONB | `{x1, y1, x2, y2}` in pixels, nullable |

**Index:** `idx_predictions_run_field` on (`processing_run_id`, `field_name`)

---

### `extracted_fields`

Sanitised, normalised field values ready for downstream use.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | |
| `cheque_id` | INTEGER FK | |
| `processing_run_id` | INTEGER FK | |
| `field_name` | VARCHAR(64) | |
| `raw_value` | TEXT | Exact OCR/recogniser output, no cleaning |
| `normalized_value` | TEXT | Post-processed: leading zeros preserved, date formatted |
| `confidence` | FLOAT | Composite confidence (det × extraction) |
| `detection_confidence` | FLOAT | Faster R-CNN objectness score |
| `extraction_confidence` | FLOAT | Tesseract word confidence or CNN softmax probability |
| `extraction_method` | VARCHAR(32) | `ocr` / `recognizer` / `hybrid` |
| `confidence_tier` | VARCHAR(16) | HIGH / MEDIUM / LOW |
| `bounding_box_json` | JSONB | Field bounding box |

**Index:** `idx_extracted_fields_cheque_name` on (`cheque_id`, `field_name`)

---

### `validation_results`

Validation outcomes, business rule checks, and reviewer resolution records.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | |
| `cheque_id` | INTEGER FK | |
| `processing_run_id` | INTEGER FK | |
| `field_name` | VARCHAR(64) | Nullable (for cross-field or document-level checks) |
| `check_type` | VARCHAR(64) | FORMAT / DATE_VALIDITY / CONSISTENCY / REQUIRED_FIELD / CONFIDENCE_GATE |
| `validation_status` | VARCHAR(32) | VALID / INVALID / WARNING |
| `is_valid` | BOOLEAN | Quick boolean |
| `validation_reason` | TEXT | Human-readable failure reason |
| `review_priority` | VARCHAR(16) | HIGH / MEDIUM / LOW |
| `reviewed_by_user_id` | INTEGER FK | Reviewer (nullable) |
| `review_resolution` | VARCHAR(32) | PENDING / ACCEPTED / REJECTED / CORRECTED |
| `reviewed_at` | TIMESTAMP | When reviewer acted |

**Index:** `idx_validation_cheque_status` on (`cheque_id`, `validation_status`)

---

### `audit_logs`

Immutable audit trail for compliance and security events.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | |
| `timestamp` | TIMESTAMP WITH TIME ZONE | UTC time of action |
| `user_id` | INTEGER FK | References `users.id` (SET NULL on user delete) |
| `username` | VARCHAR(64) | Denormalised: preserved even if user is deleted |
| `action` | VARCHAR(64) | LOGIN / UPLOAD_CHEQUE / PROCESS_CHEQUE / MANUAL_CORRECTION / STATUS_CHANGE |
| `resource_type` | VARCHAR(64) | cheque / user / validation_result / auth |
| `resource_id` | VARCHAR(128) | ID of the affected resource |
| `details_json` | JSONB | Action-specific metadata |
| `ip_address` | VARCHAR(45) | Client IP (supports IPv6) |

**Index:** `idx_audit_logs_action_timestamp` on (`action`, `timestamp`)

---

## Migrations

ChequeSense uses **Alembic** for schema versioning.

### Commands

```bash
# Apply all pending migrations
alembic upgrade head

# Generate a new migration from model changes
alembic revision --autogenerate -m "add_field_description"

# Downgrade one step
alembic downgrade -1

# Show current migration state
alembic current
```

Migration scripts are stored in `src/database/migrations/versions/`.

---

## Indexes

The following indexes are created for frequent query patterns:

| Index | Table | Columns | Purpose |
|---|---|---|---|
| `idx_cheques_status_review` | cheques | status, review_required | Review queue queries |
| `idx_cheques_created_at_status` | cheques | created_at, status | Time-series analytics |
| `idx_predictions_run_field` | predictions | processing_run_id, field_name | Field-level prediction lookup |
| `idx_extracted_fields_cheque_name` | extracted_fields | cheque_id, field_name | Field value retrieval |
| `idx_validation_cheque_status` | validation_results | cheque_id, validation_status | Validation filtering |
| `idx_audit_logs_action_timestamp` | audit_logs | action, timestamp | Audit queries |

---

## Connection Configuration

```python
# src/database/connection.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ["DATABASE_URL"]  # No hard-coded credentials

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```

See [`.env.example`](../.env.example) for the `DATABASE_URL` format.

---

## CRUD Operations

All database operations are implemented in `src/database/crud.py`:

| Function | Description |
|---|---|
| `create_cheque()` | Insert new cheque record |
| `get_cheque_by_id()` | Fetch cheque with all relationships |
| `get_cheque_by_hash()` | Deduplication lookup |
| `list_cheques()` | Paginated list with status/review filters |
| `save_pipeline_execution()` | Atomic save of run, predictions, fields, and validations |
| `correct_cheque_fields()` | Update field value from reviewer correction |
