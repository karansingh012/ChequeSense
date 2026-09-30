"""FastAPI application entrypoint for ChequeSense."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_upload_dir
from api.routes import analytics_router, cheque_router, health_router
from api.schemas import ErrorResponse
from src.database.connection import check_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chequesense.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle setup and teardown."""
    logger.info("Starting ChequeSense API backend...")
    upload_dir = get_upload_dir()
    logger.info("Cheque image upload storage initialized at: %s", upload_dir.resolve())

    # Check database liveness
    db_ok = check_db_connection()
    if db_ok:
        logger.info("PostgreSQL database connection verified successfully.")
    else:
        logger.warning("PostgreSQL database is currently unreachable. Requests requiring DB may fail.")

    yield

    logger.info("Shutting down ChequeSense API backend.")


app = FastAPI(
    title="ChequeSense API",
    version="1.0.0",
    description=(
        "Enterprise AI-powered banking backend for automated cheque digitization, "
        "field extraction, human review routing, and analytics."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for development & web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "detail": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "detail": "An internal server error occurred while processing the request.",
        },
    )


# Register API Routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(cheque_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root_redirect():
    """Convenience redirect to API documentation."""
    return {
        "service": "ChequeSense API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
    }
