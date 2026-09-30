"""Professional banking UI styling and design system tokens for ChequeSense Streamlit dashboard."""

from __future__ import annotations

BANKING_CSS = """
<style>
/* Main Container & Font Configuration */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Metric Card Containers */
.metric-container {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    margin-bottom: 12px;
}
.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
}
.metric-sub {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* Status Pill Badges */
.badge {
    display: inline-block;
    padding: 4px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 12px;
    text-align: center;
    letter-spacing: 0.02em;
}
.badge-verified {
    background-color: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}
.badge-review {
    background-color: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
}
.badge-invalid {
    background-color: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}
.badge-processed {
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}

/* Confidence Badges */
.badge-high {
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #86efac;
}
.badge-medium {
    background-color: #fef9c3;
    color: #a16207;
    border: 1px solid #fde047;
}
.badge-low {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fca5a5;
    font-weight: 700;
}

/* Alert Boxes */
.alert-card {
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 12px;
    font-size: 0.9rem;
}
.alert-warning {
    background-color: #fffbeb;
    border-left: 4px solid #f59e0b;
    color: #92400e;
}
.alert-info {
    background-color: #f0fdfa;
    border-left: 4px solid #0d9488;
    color: #134e4a;
}
.alert-danger {
    background-color: #fef2f2;
    border-left: 4px solid #ef4444;
    color: #991b1b;
}

/* Table Enhancements */
.stTable {
    border-radius: 8px;
    overflow: hidden;
}
</style>
"""


def get_status_badge_html(status_val: str) -> str:
    """Returns HTML for an operational status badge."""
    norm = str(status_val).upper()
    if norm == "VERIFIED":
        return '<span class="badge badge-verified">VERIFIED</span>'
    elif norm == "REVIEW_REQUIRED":
        return '<span class="badge badge-review">REVIEW REQUIRED</span>'
    elif norm == "INVALID":
        return '<span class="badge badge-invalid">INVALID</span>'
    else:
        return f'<span class="badge badge-processed">{norm}</span>'


def get_confidence_badge_html(confidence: float, tier: str = "MEDIUM") -> str:
    """Returns HTML for a color-coded confidence score badge."""
    pct = round(confidence * 100.0, 1)
    norm_tier = (tier or "").upper()
    if confidence >= 0.85 or norm_tier == "HIGH":
        css_cls = "badge-high"
    elif confidence >= 0.70 or norm_tier == "MEDIUM":
        css_cls = "badge-medium"
    else:
        css_cls = "badge-low"

    return f'<span class="badge {css_cls}">{pct}% ({norm_tier or "SCORE"})</span>'
