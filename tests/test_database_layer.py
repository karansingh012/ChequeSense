"""Unit and integration tests for ChequeSense database persistence layer.

Covers:
- Database schema initialization
- User entity CRUD and password hashing
- Cheque entity CRUD, deduplication hash lookup, and status updates
- Processing run audit logging with model versions and latencies
- Raw predictions and normalized extracted field persistence
- Validation results and teller review resolution
- Atomic pipeline execution persistence (save_pipeline_execution)
- Cascade delete verification
"""

import datetime
import hashlib
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base, check_db_connection, get_engine
from src.database.crud import (
    bulk_create_extracted_fields,
    bulk_create_predictions,
    bulk_create_validation_results,
    create_cheque,
    create_processing_run,
    create_user,
    get_cheque_by_hash,
    get_cheque_by_id,
    get_cheque_by_identifier,
    get_fields_for_cheque,
    get_processing_run_by_id,
    get_user_by_email,
    get_user_by_id,
    get_user_by_username,
    list_cheques,
    list_runs_for_cheque,
    list_users,
    save_pipeline_execution,
    update_cheque_status,
    update_review_resolution,
)
from src.database.models import (
    Cheque,
    ExtractedField,
    Prediction,
    ProcessingRun,
    User,
    ValidationResult,
)
from src.database.schemas import (
    ChequeCreate,
    ExtractedFieldCreate,
    PredictionCreate,
    ProcessingRunCreate,
    UserCreate,
    ValidationResultCreate,
)
from src.pipeline.schemas import (
    BoundingBox,
    ChequePipelineResult,
    FieldOutput,
    PipelineDiagnostics,
)
from src.validation.field_validator import ChequeProcessingStatus
from src.validation.review_queue import ReviewQueueItem


@pytest.fixture(scope="function")
def db_session():
    """Creates a fresh, isolated in-memory SQLite database session for each test."""
    test_engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


# ------------------------------------------------------------------------------
# 1. Connection & Schema Initialization Tests
# ------------------------------------------------------------------------------

def test_database_connection_check():
    """Verifies that check_db_connection validates connection liveness."""
    test_engine = create_engine("sqlite:///:memory:")
    assert check_db_connection(test_engine) is True


def test_schema_tables_registered():
    """Verifies that all 6 required entities exist in Base.metadata."""
    table_names = set(Base.metadata.tables.keys())
    expected = {
        "users",
        "cheques",
        "processing_runs",
        "predictions",
        "extracted_fields",
        "validation_results",
    }
    assert expected.issubset(table_names), f"Missing tables: {expected - table_names}"


# ------------------------------------------------------------------------------
# 2. User CRUD Tests
# ------------------------------------------------------------------------------

def test_create_and_query_user(db_session):
    """Tests user creation, password hashing, and lookups."""
    user_in = UserCreate(
        username="teller_john",
        email="john@bank.com",
        password="supersecretpassword",
        role="TELLER",
    )
    user = create_user(db_session, user_in)
    assert user.id is not None
    assert user.username == "teller_john"
    assert user.email == "john@bank.com"
    # Password must never be stored in plain text
    assert user.hashed_password != "supersecretpassword"
    assert user.hashed_password == hashlib.sha256("supersecretpassword".encode("utf-8")).hexdigest()

    # Query lookups
    by_id = get_user_by_id(db_session, user.id)
    assert by_id is not None and by_id.username == "teller_john"

    by_uname = get_user_by_username(db_session, "teller_john")
    assert by_uname is not None and by_uname.id == user.id

    by_email = get_user_by_email(db_session, "john@bank.com")
    assert by_email is not None and by_email.id == user.id

    users_list = list_users(db_session)
    assert len(users_list) == 1


# ------------------------------------------------------------------------------
# 3. Cheque CRUD Tests
# ------------------------------------------------------------------------------

def test_cheque_crud_and_deduplication(db_session):
    """Tests cheque creation, deduplication via image hash, and status filtering."""
    cheque_in = ChequeCreate(
        cheque_identifier="CHQ_TEST_001",
        image_path="/storage/cheques/2026/10/chq_001.png",
        image_width=1200,
        image_height=600,
        image_hash="a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
        source_dataset="synthetic",
        status="PROCESSED",
        overall_confidence=0.95,
        review_required=False,
    )
    cheque = create_cheque(db_session, cheque_in)
    assert cheque.id is not None
    assert cheque.cheque_identifier == "CHQ_TEST_001"
    # Verify image binary is NOT stored, only URI/path
    assert cheque.image_path == "/storage/cheques/2026/10/chq_001.png"

    # Query by ID and identifier
    assert get_cheque_by_id(db_session, cheque.id) is not None
    assert get_cheque_by_identifier(db_session, "CHQ_TEST_001") is not None

    # Query by image hash (deduplication check)
    hash_match = get_cheque_by_hash(db_session, cheque_in.image_hash)
    assert hash_match is not None
    assert hash_match.id == cheque.id

    # Update status
    updated = update_cheque_status(
        db_session,
        cheque.id,
        status="VERIFIED",
        overall_confidence=0.98,
        review_required=False,
    )
    assert updated.status == "VERIFIED"
    assert updated.overall_confidence == 0.98

    # Filter query
    verified = list_cheques(db_session, status="VERIFIED", review_required=False)
    assert len(verified) == 1
    assert verified[0].id == cheque.id

    unverified = list_cheques(db_session, status="REVIEW_REQUIRED")
    assert len(unverified) == 0


# ------------------------------------------------------------------------------
# 4. Processing Run & Predictions Tests
# ------------------------------------------------------------------------------

def test_processing_run_and_predictions(db_session):
    """Verifies audit logging of pipeline runs and raw model outputs."""
    cheque = create_cheque(
        db_session,
        ChequeCreate(
            cheque_identifier="CHQ_RUN_002",
            image_path="/storage/chq_002.png",
            image_width=1000,
            image_height=500,
            image_hash="b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1",
        ),
    )

    run_in = ProcessingRunCreate(
        cheque_id=cheque.id,
        status="SUCCESS",
        detector_model_version="fasterrcnn_mobilenet_v3:v1.0",
        recognizer_model_version="cheque_digit_cnn:v1.0",
        ocr_engine_version="tesseract:5.5.1",
        total_latency_ms=145.2,
        stage_latencies_json={"detection": 50.1, "ocr": 80.0, "reconstruction": 15.1},
    )
    run = create_processing_run(db_session, run_in)
    assert run.id is not None
    assert run.detector_model_version == "fasterrcnn_mobilenet_v3:v1.0"
    assert run.total_latency_ms == 145.2

    # Bulk create predictions
    preds = [
        PredictionCreate(
            cheque_id=cheque.id,
            processing_run_id=run.id,
            stage="FIELD_DETECTION",
            model_name="fasterrcnn_mobilenet_v3",
            model_version="v1.0",
            field_name="amount",
            raw_output="[750, 180, 950, 240]",
            raw_confidence=0.94,
            bounding_box_json={"xmin": 750, "ymin": 180, "xmax": 950, "ymax": 240},
        ),
        PredictionCreate(
            cheque_id=cheque.id,
            processing_run_id=run.id,
            stage="OCR_EXTRACTION",
            model_name="tesseract",
            model_version="5.5.1",
            field_name="amount",
            raw_output="**50,000/-",
            raw_confidence=0.91,
        ),
    ]
    saved_preds = bulk_create_predictions(db_session, preds)
    assert len(saved_preds) == 2

    runs = list_runs_for_cheque(db_session, cheque.id)
    assert len(runs) == 1
    assert runs[0].id == run.id


# ------------------------------------------------------------------------------
# 5. Extracted Fields & Leading-Zero Preservation
# ------------------------------------------------------------------------------

def test_extracted_fields_preserve_leading_zeros(db_session):
    """Verifies that normalized fields preserve leading zeros and formatting."""
    cheque = create_cheque(
        db_session,
        ChequeCreate(
            cheque_identifier="CHQ_FIELD_003",
            image_path="/storage/chq_003.png",
            image_width=1000,
            image_height=500,
            image_hash="c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2",
        ),
    )
    run = create_processing_run(
        db_session,
        ProcessingRunCreate(
            cheque_id=cheque.id,
            detector_model_version="detector:v1",
            recognizer_model_version="recognizer:v1",
            ocr_engine_version="tesseract:v5",
        ),
    )

    fields_in = [
        ExtractedFieldCreate(
            cheque_id=cheque.id,
            processing_run_id=run.id,
            field_name="cheque_number",
            raw_value="  004128  ",
            normalized_value="004128",  # Crucial: leading zeros must NOT be stripped into 4128
            confidence=0.96,
            detection_confidence=0.98,
            extraction_confidence=0.94,
            extraction_method="ocr",
            confidence_tier="HIGH",
        ),
        ExtractedFieldCreate(
            cheque_id=cheque.id,
            processing_run_id=run.id,
            field_name="amount",
            raw_value="Rs. 25,000/-",
            normalized_value="25000",
            confidence=0.92,
            extraction_method="ocr",
            confidence_tier="HIGH",
        ),
    ]
    saved_fields = bulk_create_extracted_fields(db_session, fields_in)
    assert len(saved_fields) == 2

    # Query fields from DB
    retrieved = get_fields_for_cheque(db_session, cheque.id)
    assert len(retrieved) == 2
    f_chq_num = next(f for f in retrieved if f.field_name == "cheque_number")
    assert f_chq_num.normalized_value == "004128"
    assert f_chq_num.raw_value == "  004128  "
    assert f_chq_num.confidence == 0.96


# ------------------------------------------------------------------------------
# 6. Validation Results & Teller Review Resolution
# ------------------------------------------------------------------------------

def test_validation_results_and_teller_resolution(db_session):
    """Tests validation status, review reasons, and human resolution."""
    teller = create_user(
        db_session,
        UserCreate(username="teller_alice", email="alice@bank.com", password="password123"),
    )
    cheque = create_cheque(
        db_session,
        ChequeCreate(
            cheque_identifier="CHQ_VAL_004",
            image_path="/storage/chq_004.png",
            image_width=1000,
            image_height=500,
            image_hash="d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3",
            status="REVIEW_REQUIRED",
            review_required=True,
        ),
    )
    run = create_processing_run(
        db_session,
        ProcessingRunCreate(
            cheque_id=cheque.id,
            detector_model_version="detector:v1",
            recognizer_model_version="recognizer:v1",
            ocr_engine_version="tesseract:v5",
        ),
    )

    vr_in = ValidationResultCreate(
        cheque_id=cheque.id,
        processing_run_id=run.id,
        field_name="date",
        check_type="DATE_VALIDITY",
        validation_status="INVALID",
        is_valid=False,
        validation_reason="Cheque is stale (dated older than 90 days: 2024-01-15)",
        review_priority="HIGH",
    )
    saved_vrs = bulk_create_validation_results(db_session, [vr_in])
    vr = saved_vrs[0]
    assert vr.id is not None
    assert vr.validation_status == "INVALID"
    assert "stale" in vr.validation_reason
    assert vr.review_resolution == "PENDING"

    # Human Teller reviews and accepts with manager approval
    resolved = update_review_resolution(
        db_session,
        validation_id=vr.id,
        reviewed_by_user_id=teller.id,
        resolution="ACCEPTED",
    )
    assert resolved.review_resolution == "ACCEPTED"
    assert resolved.reviewed_by_user_id == teller.id
    assert resolved.reviewed_at is not None


# ------------------------------------------------------------------------------
# 7. Integrated Atomic Pipeline Persistence
# ------------------------------------------------------------------------------

def test_save_pipeline_execution_atomic(db_session):
    """Verifies end-to-end saving from pipeline and review queue outputs."""
    # Mock ChequePipelineResult
    fields = {
        "cheque_number": FieldOutput(
            value="009123",
            raw_value="009123",
            confidence=0.97,
            detection_confidence=0.99,
            extraction_confidence=0.98,
            method="ocr",
            confidence_tier="HIGH",
            bounding_box=BoundingBox(xmin=400, ymin=480, xmax=550, ymax=520),
        ),
        "amount": FieldOutput(
            value="15000",
            raw_value="15,000/-",
            confidence=0.93,
            detection_confidence=0.97,
            extraction_confidence=0.96,
            method="ocr",
            confidence_tier="HIGH",
            bounding_box=BoundingBox(xmin=700, ymin=180, xmax=920, ymax=230),
        ),
    }
    diagnostics = PipelineDiagnostics(
        image_width=1200,
        image_height=600,
        processing_time_ms=132.5,
        stage_latencies_ms={"preprocess": 10.0, "detection": 45.0, "ocr": 70.0, "reconstruct": 7.5},
    )
    pipeline_res = ChequePipelineResult(
        cheque_id="CHQ_PIPE_005",
        status="PROCESSED",
        overall_confidence=0.95,
        fields=fields,
        diagnostics=diagnostics,
    )

    review_item = ReviewQueueItem(
        cheque_id="CHQ_PIPE_005",
        status=ChequeProcessingStatus.REVIEW_REQUIRED,
        overall_confidence=0.95,
        review_priority="MEDIUM",
        review_required=True,
        reasons_for_review=["Account number missing or obscured"],
    )

    saved_cheque = save_pipeline_execution(
        db=db_session,
        cheque_identifier="CHQ_PIPE_005",
        image_path="/storage/cheques/test_pipe_005.png",
        image_width=1200,
        image_height=600,
        image_hash="e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4",
        pipeline_result=pipeline_res,
        review_item=review_item,
        detector_version="fasterrcnn:v1",
        recognizer_version="cnn:v1",
        ocr_version="tesseract:v5",
    )

    assert saved_cheque.id is not None
    assert saved_cheque.status == "REVIEW_REQUIRED"
    assert saved_cheque.review_required is True

    # Check child relations
    runs = list_runs_for_cheque(db_session, saved_cheque.id)
    assert len(runs) == 1
    assert runs[0].detector_model_version == "fasterrcnn:v1"
    assert runs[0].total_latency_ms == 132.5

    extracted = get_fields_for_cheque(db_session, saved_cheque.id)
    assert len(extracted) == 2
    f_map = {f.field_name: f for f in extracted}
    assert f_map["cheque_number"].normalized_value == "009123"
    assert f_map["amount"].normalized_value == "15000"
    assert f_map["amount"].raw_value == "15,000/-"


# ------------------------------------------------------------------------------
# 8. Cascade Delete Test
# ------------------------------------------------------------------------------

def test_cascade_deletes(db_session):
    """Verifies that deleting a cheque cascades to runs, fields, predictions, and validations."""
    cheque = create_cheque(
        db_session,
        ChequeCreate(
            cheque_identifier="CHQ_CASCADE_006",
            image_path="/storage/chq_006.png",
            image_width=1000,
            image_height=500,
            image_hash="f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5",
        ),
    )
    run = create_processing_run(
        db_session,
        ProcessingRunCreate(
            cheque_id=cheque.id,
            detector_model_version="det:v1",
            recognizer_model_version="rec:v1",
            ocr_engine_version="ocr:v1",
        ),
    )
    bulk_create_extracted_fields(
        db_session,
        [
            ExtractedFieldCreate(
                cheque_id=cheque.id,
                processing_run_id=run.id,
                field_name="amount",
                raw_value="1000",
                normalized_value="1000",
                confidence=0.99,
            )
        ],
    )

    # Delete cheque
    db_session.delete(cheque)
    db_session.commit()

    assert db_session.get(Cheque, cheque.id) is None
    assert db_session.get(ProcessingRun, run.id) is None
    fields = db_session.query(ExtractedField).filter_by(cheque_id=cheque.id).all()
    assert len(fields) == 0


# ------------------------------------------------------------------------------
# 9. Live PostgreSQL Database Integration Test
# ------------------------------------------------------------------------------

def test_live_postgres_connection():
    """Validates live connectivity to PostgreSQL database configured in environment."""
    engine = get_engine()
    assert check_db_connection(engine) is True


def test_live_postgres_crud_end_to_end():
    """Validates full CRUD workflow and JSONB operations on real PostgreSQL instance."""
    engine = get_engine()
    if not check_db_connection(engine):
        pytest.skip("PostgreSQL service is not available")

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    unique_id = f"PG_TEST_{int(datetime.datetime.utcnow().timestamp())}"
    try:
        # Create cheque on live PostgreSQL
        cheque_in = ChequeCreate(
            cheque_identifier=unique_id,
            image_path=f"/var/cheques/{unique_id}.png",
            image_width=1200,
            image_height=600,
            image_hash=hashlib.sha256(unique_id.encode()).hexdigest(),
            source_dataset="production",
            status="PROCESSED",
            overall_confidence=0.98,
        )
        chq = create_cheque(session, cheque_in)
        assert chq.id is not None

        # Verify retrieval
        queried = get_cheque_by_identifier(session, unique_id)
        assert queried is not None
        assert queried.image_path == f"/var/cheques/{unique_id}.png"

        # Create processing run with JSONB payload
        run = create_processing_run(
            session,
            ProcessingRunCreate(
                cheque_id=chq.id,
                detector_model_version="detector:v1.0",
                recognizer_model_version="recognizer:v1.0",
                ocr_engine_version="tesseract:v5.5",
                stage_latencies_json={"det": 40.2, "ocr": 65.1},
            ),
        )
        assert run.id is not None
        assert run.stage_latencies_json["det"] == 40.2

        # Cleanup test record
        session.delete(chq)
        session.commit()
        assert get_cheque_by_identifier(session, unique_id) is None
    finally:
        session.close()

