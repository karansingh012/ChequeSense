"""Health and readiness diagnostic endpoints for ChequeSense API."""

from __future__ import annotations

import datetime
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_pipeline
from api.schemas import HealthResponse
from src.database.connection import check_db_connection

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check API and infrastructure health",
    description="Returns connectivity status for PostgreSQL database, pipeline state, and API version.",
)
def get_health(db: Session = Depends(get_db)) -> HealthResponse:
    db_connected = check_db_connection(db.get_bind())
    db_status = "connected" if db_connected else "disconnected"

    try:
        pipeline = get_pipeline()
        pipeline_status = "available" if pipeline is not None else "degraded"
    except Exception:
        pipeline_status = "degraded"

    overall_status = "healthy" if (db_connected and pipeline_status == "available") else "degraded"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.datetime.utcnow(),
        database=db_status,
        pipeline=pipeline_status,
    )
