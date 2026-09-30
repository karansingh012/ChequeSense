# ChequeSense Database Persistence Layer (Phase 7)

## 1. Architecture & Design Principles

The ChequeSense persistence layer provides a normalized, audit-compliant PostgreSQL database backed by **SQLAlchemy 2.0** ORM and **Alembic** migrations. It supports high-throughput cheque processing pipelines, human teller review workflows, and long-term regulatory compliance.

### Core Architectural Principles:
1. **Normalized Relational Schema (3NF)**: Eliminates redundancy across users, cheques, processing passes, extracted fields, raw model inferences, and review resolutions.
2. **Externalized Binary Storage**: Large cheque image binaries are **never stored directly in PostgreSQL**. Only storage URIs/paths (`cheques.image_path`) and cryptographic SHA-256 digests (`cheques.image_hash`) are persisted. This prevents database bloat, maintains lean buffer pools, and enables standard cloud object storage (S3 / GCS / NAS) integration.
3. **Leading-Zero & Identity Preservation**: Banking identifiers (`cheque_number`, `account_number`, `branch_code`, `date`) are stored as explicit text fields (`VARCHAR` / `TEXT`) rather than integers, guaranteeing zero data truncation or loss of leading zeros (e.g. `'004128'`).
4. **Complete Audit Trail & Model Provenance**:
   - Model versions (`detector_model_version`, `recognizer_model_version`, `ocr_engine_version`) are permanently recorded per execution run.
   - Both **raw model outputs** (pre-normalization) and **sanitized values** (post-normalization) are retained side-by-side.
   - Multi-stage confidence scores (detection, extraction, composite) are preserved.
   - Validation outcomes, failure gates, and exact reasons for review are logged with teller resolution histories.
5. **Zero Hardcoded Secrets**: All connection credentials and host parameters are driven strictly by environment variables with fallback parameter reconstruction.

---

## 2. Entity-Relationship Diagram (ERD)

```mermaid
erDiagram
    users ||--o{ cheques : "creates/submits"
    users ||--o{ validation_results : "reviews/approves"
    cheques ||--|{ processing_runs : "executes"
    cheques ||--o{ predictions : "has raw inferences"
    cheques ||--o{ extracted_fields : "has normalized fields"
    cheques ||--o{ validation_results : "undergoes validation"
    processing_runs ||--o{ predictions : "generates"
    processing_runs ||--o{ extracted_fields : "produces"
    processing_runs ||--o{ validation_results : "evaluates"

    users {
        int id PK
        string username UK
        string email UK
        string hashed_password
        string role
        boolean is_active
        datetime created_at
        datetime updated_at
    }

    cheques {
        int id PK
        string cheque_identifier UK
        string image_path
        int image_width
        int image_height
        string image_hash IX
        string source_dataset
        string status IX
        float overall_confidence
        boolean review_required IX
        int created_by_user_id FK
        datetime created_at IX
        datetime updated_at
    }

    processing_runs {
        int id PK
        int cheque_id FK
        datetime run_timestamp IX
        string status
        string detector_model_version
        string recognizer_model_version
        string ocr_engine_version
        float total_latency_ms
        jsonb stage_latencies_json
    }

    predictions {
        int id PK
        int processing_run_id FK
        int cheque_id FK
        string stage
        string model_name
        string model_version
        string field_name IX
        text raw_output
        float raw_confidence
        jsonb bounding_box_json
        datetime created_at
    }

    extracted_fields {
        int id PK
        int cheque_id FK
        int processing_run_id FK
        string field_name IX
        text raw_value
        text normalized_value
        float confidence
        float detection_confidence
        float extraction_confidence
        string extraction_method
        string confidence_tier IX
        jsonb bounding_box_json
        datetime created_at
    }

    validation_results {
        int id PK
        int cheque_id FK
        int processing_run_id FK
        string field_name IX
        string check_type IX
        string validation_status IX
        boolean is_valid
        text validation_reason
        string review_priority
        int reviewed_by_user_id FK
        string review_resolution
        datetime reviewed_at
        datetime created_at
    }
```

---

## 3. Database Schema Specification

### 3.1 `users`
Represents bank tellers, compliance officers, and system administrators.
- `id` (INTEGER, PK, Autoincrement)
- `username` (VARCHAR(64), UNIQUE, NOT NULL, Index)
- `email` (VARCHAR(255), UNIQUE, NOT NULL, Index)
- `hashed_password` (VARCHAR(255), NOT NULL): SHA-256 / bcrypt hash; raw password is never stored.
- `role` (VARCHAR(32), NOT NULL, Default: `'TELLER'`): e.g. `'TELLER'`, `'AUDITOR'`, `'ADMIN'`.
- `is_active` (BOOLEAN, NOT NULL, Default: `TRUE`)
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL, Default: `now()`)
- `updated_at` (TIMESTAMP WITH TIME ZONE, NOT NULL, Default: `now()`)

### 3.2 `cheques`
Master document record for each incoming cheque.
- `id` (INTEGER, PK, Autoincrement)
- `cheque_identifier` (VARCHAR(128), UNIQUE, NOT NULL, Index): Unique reference number or filename.
- `image_path` (VARCHAR(512), NOT NULL): URI or path to image on disk or cloud storage bucket.
- `image_width` (INTEGER, NOT NULL): Pixel width.
- `image_height` (INTEGER, NOT NULL): Pixel height.
- `image_hash` (VARCHAR(64), NOT NULL, Index): SHA-256 digest of image bytes for deduplication.
- `source_dataset` (VARCHAR(64), NOT NULL, Default: `'production'`): Dataset origin tag.
- `status` (VARCHAR(32), NOT NULL, Default: `'PROCESSED'`, Index): `'PROCESSED'`, `'VERIFIED'`, `'REVIEW_REQUIRED'`, `'INVALID'`.
- `overall_confidence` (FLOAT, NOT NULL, Default: `0.0`): Composite confidence across detected fields.
- `review_required` (BOOLEAN, NOT NULL, Default: `FALSE`, Index): Urgent queue filter flag.
- `created_by_user_id` (INTEGER, FK -> `users.id`, NULLABLE): Teller or ingest operator.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL, Default: `now()`, Index)
- `updated_at` (TIMESTAMP WITH TIME ZONE, NOT NULL, Default: `now()`)
- **Compound Indexes**:
  - `idx_cheques_status_review` on `(status, review_required)`
  - `idx_cheques_created_at_status` on `(created_at, status)`

### 3.3 `processing_runs`
Audit telemetry capturing individual ML pipeline execution attempts.
- `id` (INTEGER, PK, Autoincrement)
- `cheque_id` (INTEGER, FK -> `cheques.id` ON DELETE CASCADE, NOT NULL, Index)
- `run_timestamp` (TIMESTAMP WITH TIME ZONE, NOT NULL, Default: `now()`, Index)
- `status` (VARCHAR(32), NOT NULL, Default: `'SUCCESS'`)
- `detector_model_version` (VARCHAR(128), NOT NULL): e.g. `'fasterrcnn_mobilenet_v3_large_fpn:v1.0'`
- `recognizer_model_version` (VARCHAR(128), NOT NULL): e.g. `'cheque_digit_cnn:v1.0'`
- `ocr_engine_version` (VARCHAR(128), NOT NULL): e.g. `'tesseract:5.5.1'`
- `total_latency_ms` (FLOAT, NOT NULL, Default: `0.0`): Execution time.
- `stage_latencies_json` (JSONB / JSON, NULLABLE): Detailed breakdown per pipeline stage.

### 3.4 `predictions`
Raw model inferences recorded before sanitization, business filtering, or post-processing.
- `id` (INTEGER, PK, Autoincrement)
- `processing_run_id` (INTEGER, FK -> `processing_runs.id` ON DELETE CASCADE, NOT NULL, Index)
- `cheque_id` (INTEGER, FK -> `cheques.id` ON DELETE CASCADE, NOT NULL, Index)
- `stage` (VARCHAR(64), NOT NULL): `'FIELD_DETECTION'`, `'OCR_EXTRACTION'`, `'HANDWRITTEN_RECOGNITION'`.
- `model_name` (VARCHAR(128), NOT NULL)
- `model_version` (VARCHAR(64), NOT NULL)
- `field_name` (VARCHAR(64), NOT NULL, Index): e.g. `'cheque_number'`, `'amount'`, `'date'`.
- `raw_output` (TEXT, NOT NULL): Unaltered string or bounding coordinate string.
- `raw_confidence` (FLOAT, NOT NULL, Default: `0.0`)
- `bounding_box_json` (JSONB / JSON, NULLABLE): Spatial coordinates `{xmin, ymin, xmax, ymax}`.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL, Default: `now()`)
- **Compound Index**:
  - `idx_predictions_run_field` on `(processing_run_id, field_name)`

### 3.5 `extracted_fields`
Sanitized, validated field values ready for banking ledger posting.
- `id` (INTEGER, PK, Autoincrement)
- `cheque_id` (INTEGER, FK -> `cheques.id` ON DELETE CASCADE, NOT NULL, Index)
- `processing_run_id` (INTEGER, FK -> `processing_runs.id` ON DELETE CASCADE, NOT NULL, Index)
- `field_name` (VARCHAR(64), NOT NULL, Index): `'cheque_number'`, `'amount'`, `'date'`, etc.
- `raw_value` (TEXT, NOT NULL): Raw input value for auditing.
- `normalized_value` (TEXT, NOT NULL): Formatted value preserving leading zeros (`'004128'`).
- `confidence` (FLOAT, NOT NULL, Default: `0.0`): Composite confidence score.
- `detection_confidence` (FLOAT, NOT NULL, Default: `0.0`)
- `extraction_confidence` (FLOAT, NOT NULL, Default: `0.0`)
- `extraction_method` (VARCHAR(32), NOT NULL, Default: `'ocr'`): `'ocr'`, `'recognizer'`, `'hybrid'`.
- `confidence_tier` (VARCHAR(16), NOT NULL, Default: `'MEDIUM'`, Index): `'HIGH'`, `'MEDIUM'`, `'LOW'`.
- `bounding_box_json` (JSONB / JSON, NULLABLE): Spatial bounds.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL, Default: `now()`)
- **Compound Index**:
  - `idx_extracted_fields_cheque_name` on `(cheque_id, field_name)`

### 3.6 `validation_results`
Business compliance validation outcomes and human teller audit resolutions.
- `id` (INTEGER, PK, Autoincrement)
- `cheque_id` (INTEGER, FK -> `cheques.id` ON DELETE CASCADE, NOT NULL, Index)
- `processing_run_id` (INTEGER, FK -> `processing_runs.id` ON DELETE CASCADE, NOT NULL, Index)
- `field_name` (VARCHAR(64), NULLABLE, Index): Specific field evaluated or NULL for document-level check.
- `check_type` (VARCHAR(64), NOT NULL, Index): `'FORMAT'`, `'DATE_VALIDITY'`, `'CONSISTENCY'`, `'REQUIRED_FIELD'`, `'CONFIDENCE_GATE'`.
- `validation_status` (VARCHAR(32), NOT NULL, Default: `'VALID'`, Index): `'VALID'`, `'INVALID'`, `'WARNING'`.
- `is_valid` (BOOLEAN, NOT NULL, Default: `TRUE`)
- `validation_reason` (TEXT, NOT NULL): Preserved diagnostic description for review trigger.
- `review_priority` (VARCHAR(16), NOT NULL, Default: `'MEDIUM'`): `'HIGH'`, `'MEDIUM'`, `'LOW'`.
- `reviewed_by_user_id` (INTEGER, FK -> `users.id`, NULLABLE): Reviewing teller.
- `review_resolution` (VARCHAR(32), NULLABLE, Default: `'PENDING'`): `'PENDING'`, `'ACCEPTED'`, `'REJECTED'`, `'CORRECTED'`.
- `reviewed_at` (TIMESTAMP WITH TIME ZONE, NULLABLE)
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL, Default: `now()`)
- **Compound Index**:
  - `idx_validation_cheque_status` on `(cheque_id, validation_status)`

---

## 4. Environment Configuration & Security

Credentials must **never** be hardcoded. The application reads configuration from environment variables (or `.env` file via `python-dotenv`).

### Key Parameters:
```ini
# Database Connection Parameters
POSTGRES_USER=karansingh
POSTGRES_PASSWORD=
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=chequesense

# Or Direct Connection URI
DATABASE_URL=postgresql+psycopg2:///chequesense
```

### Connection URL Resolution:
In `src/database/connection.py`:
1. Checks for direct `DATABASE_URL`.
2. If absent, dynamically constructs `postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{db_name}`.
3. Automatically falls back gracefully for unit tests using in-memory SQLite when configured.

---

## 5. Database Initialization & Alembic Migrations

### Initializing Tables via SQLAlchemy
To create all tables programmatically without migrations (useful in clean environments or tests):
```python
from src.database.connection import init_db
init_db()
```

### Managing Schema Migrations with Alembic
Alembic migration configuration is located at `alembic.ini` and `src/database/migrations/`.

1. **Check Migration Status**:
   ```bash
   alembic current
   ```

2. **Generate New Migration Revision**:
   ```bash
   alembic revision --autogenerate -m "add_new_audit_column"
   ```

3. **Apply Migrations (Upgrade to Head)**:
   ```bash
   alembic upgrade head
   ```

4. **Rollback Last Migration**:
   ```bash
   alembic downgrade -1
   ```

---

## 6. CRUD & Atomic Pipeline Ingestion

The repository layer (`src/database/crud.py`) provides typed functions for all entities, as well as an atomic transaction helper `save_pipeline_execution()` bridging the ML pipeline directly into the database.

### Example: Ingesting Pipeline Output
```python
from src.database.connection import db_session_scope
from src.database.crud import save_pipeline_execution
from src.pipeline.pipeline import ChequeProcessingPipeline
from src.validation.review_queue import ReviewQueueManager

pipeline = ChequeProcessingPipeline()
queue_mgr = ReviewQueueManager()

# Process image
pipeline_result = pipeline.process_image("path/to/cheque.png")
review_item = queue_mgr.process_and_enqueue(pipeline_result)

# Atomic database save
with db_session_scope() as session:
    cheque = save_pipeline_execution(
        db=session,
        cheque_identifier="CHQ_CANARA_0091",
        image_path="data/raw/cheque_0091.png",
        image_width=1200,
        image_height=600,
        image_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        pipeline_result=pipeline_result,
        review_item=review_item,
        detector_version="fasterrcnn:v1.0",
        recognizer_version="cnn:v1.0",
        ocr_version="tesseract:v5.5.1",
    )
    print(f"Persisted Cheque ID: {cheque.id}, Status: {cheque.status}")
```

### Example: Teller Review Resolution
```python
from src.database.connection import db_session_scope
from src.database.crud import update_review_resolution

with db_session_scope() as session:
    resolved_val = update_review_resolution(
        db=session,
        validation_id=14,
        reviewed_by_user_id=1,
        resolution="ACCEPTED",
    )
```

---

## 7. Performance & Indexing Strategy

To support sub-millisecond lookups under high transaction volume:
- **`cheques.cheque_identifier` (Unique B-tree)**: Instant primary business key lookups.
- **`cheques.image_hash` (B-tree)**: Rapid deduplication prevention before running expensive GPU/CPU inference passes.
- **`idx_cheques_status_review` (Compound B-tree)**: Optimizes teller review dashboards (`WHERE status = 'REVIEW_REQUIRED' AND review_required = TRUE`).
- **`idx_extracted_fields_cheque_name` (Compound B-tree)**: Rapid retrieval of individual cheque fields for verification screens.
- **`idx_predictions_run_field` (Compound B-tree)**: Accelerates ML audit and model comparison queries.
- **`idx_validation_cheque_status` (Compound B-tree)**: Fast compliance reporting and audit sampling.
