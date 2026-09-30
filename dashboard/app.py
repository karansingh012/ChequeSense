"""Main Streamlit application entrypoint for ChequeSense Banking Suite."""

from __future__ import annotations

import streamlit as st

from dashboard.components import BANKING_CSS, backend_client

# Configure page layout and metadata
st.set_page_config(
    page_title="ChequeSense | AI Banking Suite",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply corporate banking design system
st.markdown(BANKING_CSS, unsafe_allow_html=True)

# Sidebar Header & Brand
with st.sidebar:
    st.markdown("## 🏦 ChequeSense")
    st.caption("AI-Powered Cheque Digitization & Verification")

    # System Status Indicator
    is_http = backend_client.is_http_backend_available()
    if is_http:
        st.markdown(
            '<div style="font-size:0.75rem; color:#059669; font-weight:600;">● Backend API: CONNECTED (HTTP)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.75rem; color:#2563eb; font-weight:600;">● Mode: DIRECT SERVICE LAYER</div>',
            unsafe_allow_html=True,
        )

    st.divider()

# Configure Multi-Page Navigation
pages = {
    "Cheque Operations": [
        st.Page("pages/upload.py", title="Upload & Process", icon="📤"),
        st.Page("pages/results.py", title="Extraction Results", icon="🔍"),
        st.Page("pages/review.py", title="Teller Review Queue", icon="📋"),
    ],
    "Banking Intelligence": [
        st.Page("pages/history.py", title="Cheque History", icon="🗄️"),
        st.Page("pages/analytics.py", title="Executive Analytics", icon="📊"),
    ],
}

nav = st.navigation(pages)

# Sidebar Footer
with st.sidebar:
    st.divider()
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#64748b;">
            <strong>ChequeSense v1.0.0</strong><br>
            Compliance & Audit Ready<br>
            PostgreSQL & PyTorch
        </div>
        """,
        unsafe_allow_html=True,
    )

# Run navigated page
nav.run()
