"""API Route modules for ChequeSense."""

from api.routes.analytics import router as analytics_router
from api.routes.cheque import router as cheque_router
from api.routes.health import router as health_router

__all__ = ["analytics_router", "cheque_router", "health_router"]
