"""Audit logging service for tracking critical security and banking operations."""

from __future__ import annotations

import datetime
import json
import logging
from enum import Enum
from typing import Any, Dict, Optional, Union

from sqlalchemy.orm import Session

from src.database.models import AuditLog

logger = logging.getLogger("chequesense.audit")


class AuditAction(str, Enum):
    """Categorized actions subject to strict audit trails."""

    LOGIN = "LOGIN"
    UPLOAD_CHEQUE = "UPLOAD_CHEQUE"
    PROCESS_CHEQUE = "PROCESS_CHEQUE"
    MANUAL_CORRECTION = "MANUAL_CORRECTION"
    STATUS_CHANGE = "STATUS_CHANGE"


def record_audit_event(
    session: Session,
    action: Union[AuditAction, str],
    username: str,
    user_id: Optional[int] = None,
    resource_type: str = "cheque",
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> AuditLog:
    """Persists an immutable audit log entry and outputs a structured log event."""
    action_str = action.value if isinstance(action, AuditAction) else str(action)

    audit_entry = AuditLog(
        timestamp=datetime.datetime.utcnow(),
        user_id=user_id,
        username=username,
        action=action_str,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        details_json=details or {},
        ip_address=ip_address,
    )
    session.add(audit_entry)
    session.commit()
    session.refresh(audit_entry)

    # Structured audit log output for SIEM ingestion
    logger.info(
        "AUDIT_EVENT: action=%s user=%s resource=%s/%s ip=%s details=%s",
        action_str,
        username,
        resource_type,
        resource_id,
        ip_address or "unknown",
        json.dumps(details or {}),
    )

    return audit_entry
