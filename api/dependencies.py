"""FastAPI dependency injection providers for ChequeSense."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from src.database.connection import get_db as db_generator
from src.pipeline.pipeline import ChequeInferencePipeline
from src.validation.review_queue import ReviewQueueManager

logger = logging.getLogger("chequesense.api.dependencies")

# File upload configuration constants
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
    "application/octet-stream",  # Fallback client headers
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def get_upload_dir() -> Path:
    """Returns the validated directory for persistent cheque image storage."""
    configured = os.getenv("UPLOAD_DIR", "data/uploads")
    upload_path = Path(configured)
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


def get_db() -> Generator[Session, None, None]:
    """Yields a database session with automatic lifecycle management."""
    yield from db_generator()


@lru_cache(maxsize=1)
def get_pipeline() -> ChequeInferencePipeline:
    """Returns a singleton ChequeInferencePipeline instance to avoid model re-initialization."""
    logger.info("Initializing ChequeInferencePipeline singleton for API...")
    return ChequeInferencePipeline()


@lru_cache(maxsize=1)
def get_review_queue_manager() -> ReviewQueueManager:
    """Returns a singleton ReviewQueueManager instance for rule validation."""
    return ReviewQueueManager()
