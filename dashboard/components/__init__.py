"""Dashboard components for ChequeSense."""

from dashboard.components.api_client import BackendClient, backend_client
from dashboard.components.styles import (
    BANKING_CSS,
    get_confidence_badge_html,
    get_status_badge_html,
)
from dashboard.components.visualizations import (
    draw_bounding_boxes_on_cheque,
    format_currency_inr,
)

__all__ = [
    "BackendClient",
    "backend_client",
    "BANKING_CSS",
    "get_status_badge_html",
    "get_confidence_badge_html",
    "draw_bounding_boxes_on_cheque",
    "format_currency_inr",
]
