"""FastAPI dependency injection providers for ChequeSense."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Generator, List, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.database.connection import get_db as db_generator
from src.database.crud import get_user_by_username
from src.database.models import User
from src.pipeline.pipeline import ChequeInferencePipeline
from src.security.auth import UserRole, decode_access_token
from src.validation.review_queue import ReviewQueueManager

logger = logging.getLogger("chequesense.api.dependencies")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=True)

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


def get_client_ip(request: Request) -> Optional[str]:
    """Extracts client IP address respecting X-Forwarded-For if available."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Validates JWT access token and resolves active database User."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
    except Exception as e:
        logger.warning("Token decoding failed: %s", e)
        raise credentials_exception

    user = get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account",
        )
    return user


def require_roles(*allowed_roles: UserRole) -> Callable[[User], User]:
    """Dependency factory enforcing Role-Based Access Control (RBAC).
    
    ADMIN role is granted global operational access across all endpoints.
    """
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        try:
            curr_role = UserRole(current_user.role.upper())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid user role: {current_user.role}",
            )

        if curr_role == UserRole.ADMIN or curr_role in allowed_roles:
            return current_user

        role_names = [r.value for r in allowed_roles]
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access forbidden: requires one of {role_names}, but user '{current_user.username}' has role '{curr_role.value}'",
        )

    return role_checker

