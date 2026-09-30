"""Authentication, password security, and Role-Based Access Control (RBAC) for ChequeSense."""

from __future__ import annotations

import datetime
from datetime import timezone
import hashlib
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import bcrypt
import jwt

logger = logging.getLogger("chequesense.security.auth")

# JWT Configuration (Environment variables with secure fallback defaults)
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7",  # 256-bit default key
)
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours default


# ------------------------------------------------------------------------------
# 1. Roles & Permissions Definitions
# ------------------------------------------------------------------------------

class UserRole(str, Enum):
    """Supported banking operational roles."""

    ADMIN = "ADMIN"
    EMPLOYEE = "EMPLOYEE"
    ANALYST = "ANALYST"
    REVIEWER = "REVIEWER"


class Permission(str, Enum):
    """Fine-grained operational permissions."""

    MANAGE_USERS = "manage_users"
    UPLOAD_CHEQUES = "upload_cheques"
    PROCESS_CHEQUES = "process_cheques"
    VIEW_RESULTS = "view_results"
    REVIEW_QUEUE = "review_queue"
    CORRECT_VALUES = "correct_values"
    ACCESS_ANALYTICS = "access_analytics"


ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: {
        Permission.MANAGE_USERS,
        Permission.UPLOAD_CHEQUES,
        Permission.PROCESS_CHEQUES,
        Permission.VIEW_RESULTS,
        Permission.REVIEW_QUEUE,
        Permission.CORRECT_VALUES,
        Permission.ACCESS_ANALYTICS,
    },
    UserRole.EMPLOYEE: {
        Permission.UPLOAD_CHEQUES,
        Permission.PROCESS_CHEQUES,
        Permission.VIEW_RESULTS,
    },
    UserRole.REVIEWER: {
        Permission.REVIEW_QUEUE,
        Permission.CORRECT_VALUES,
        Permission.VIEW_RESULTS,
    },
    UserRole.ANALYST: {
        Permission.ACCESS_ANALYTICS,
        Permission.VIEW_RESULTS,
    },
}


def has_permission(role: str, permission: Permission) -> bool:
    """Verifies whether a given role holds the requested permission."""
    try:
        urole = UserRole(role.upper())
        return permission in ROLE_PERMISSIONS.get(urole, set())
    except ValueError:
        return False


# ------------------------------------------------------------------------------
# 2. Password Hashing & Verification (Bcrypt)
# ------------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """Hashes a password securely using bcrypt with cost factor 12.
    
    Plaintext passwords are NEVER stored in database.
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against the stored bcrypt hash.
    
    Includes backward-compatibility fallback for legacy SHA-256 test hashes.
    """
    if not plain_password or not hashed_password:
        return False

    # 1. Standard Bcrypt verification
    try:
        if hashed_password.startswith("$2b$") or hashed_password.startswith("$2a$"):
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception as e:
        logger.warning("Bcrypt verification failed with exception: %s", e)

    # 2. Backward compatibility fallback for legacy 64-char SHA-256 test hashes
    if len(hashed_password) == 64:
        legacy_hash = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
        return legacy_hash == hashed_password

    return False


# ------------------------------------------------------------------------------
# 3. JWT Token Generation & Validation
# ------------------------------------------------------------------------------

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[datetime.timedelta] = None,
) -> str:
    """Encodes a signed JWT access token containing user identity and role claims."""
    to_encode = data.copy()
    now = datetime.datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + datetime.timedelta(minutes=JWT_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": now,
    })

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decodes and validates a signed JWT token signature and expiration.
    
    Raises:
        jwt.ExpiredSignatureError: If token expired.
        jwt.InvalidTokenError: If token signature or structure is invalid.
    """
    payload = jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
        options={"require": ["exp", "sub", "role"]},
    )
    return payload
