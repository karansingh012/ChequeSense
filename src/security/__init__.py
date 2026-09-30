"""Security module for ChequeSense providing RBAC, bcrypt password security, JWT auth, and audit logging."""

from src.security.auth import (
    Permission,
    ROLE_PERMISSIONS,
    UserRole,
    create_access_token,
    decode_access_token,
    has_permission,
    hash_password,
    verify_password,
)
from src.security.audit import (
    AuditAction,
    record_audit_event,
)

__all__ = [
    "UserRole",
    "Permission",
    "ROLE_PERMISSIONS",
    "has_permission",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "AuditAction",
    "record_audit_event",
]
