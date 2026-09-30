"""Cheque Ingestion & Processing History Page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import (
    BANKING_CSS,
    backend_client,
    format_currency_inr,
    get_confidence_badge_html,
    get_status_badge_html,
)
from src.analytics.reports import generate_cheque_reconciliation_csv
from src.database.connection import db_session_scope

st.markdown(BANKING_CSS, unsafe_allow_html=True)

st.title("🗄️ Cheque Processing History")
st.markdown("Searchable ledger of all processed cheque documents, extracted field states, and pipeline audit logs.")

# Filter Bar
f_col1, f_col2, f_col3 = st.columns([2, 2, 3])
with f_col1:
    status_choice = st.selectbox(
        "Filter Status:",
        ["ALL", "VERIFIED", "REVIEW_REQUIRED", "PROCESSED", "INVALID"],
    )
with f_col2:
    review_choice = st.selectbox(
        "Review Required:",
        ["ALL", "YES", "NO"],
    )
with f_col3:
    search_query = st.text_input("Search Identifier / Cheque No:", placeholder="e.g. CHQ_ or 004128")

status_filter = None if status_choice == "ALL" else status_choice
review_filter = None if review_choice == "ALL" else (True if review_choice == "YES" else False)

# Fetch data
cheques_resp = backend_client.list_cheques(
    status=status_filter,
    review_required=review_filter,
    limit=100,
)
cheques = cheques_resp.get("items", [])

# Apply client-side search query if specified
if search_query.strip():
    q = search_query.strip().lower()
    filtered = []
    for c in cheques:
        ident_match = q in c["cheque_identifier"].lower()
        field_match = any(q in f["normalized_value"].lower() for f in c.get("extracted_fields", []))
        if ident_match or field_match:
            filtered.append(c)
    cheques = filtered

st.caption(f"Showing {len(cheques)} record(s)")

# Export Ledger Button
c_exp, _ = st.columns([2, 5])
with c_exp:
    with db_session_scope() as session:
        csv_data = generate_cheque_reconciliation_csv(session, limit=500)
    st.download_button(
        label="📥 Export Reconciliation Ledger (CSV)",
        data=csv_data,
        file_name="chequesense_ledger.csv",
        mime="text/csv",
    )

if not cheques:
    st.info("No matching cheques found.")
    st.stop()

# Build Table Data
table_rows = []
for c in cheques:
    f_map = {f["field_name"]: f["normalized_value"] for f in c.get("extracted_fields", [])}
    runs = c.get("processing_runs", [])
    latency = f"{runs[0]['total_latency_ms']:.1f} ms" if runs else "—"

    amt_str = format_currency_inr(f_map.get("amount", "—")) if "amount" in f_map else "—"

    table_rows.append({
        "ID": c["id"],
        "Identifier": c["cheque_identifier"],
        "Date Ingested": c["created_at"][:19] if c["created_at"] else "—",
        "Status": c["status"],
        "Confidence": f"{c['overall_confidence']:.1%}",
        "Review Required": "YES" if c["review_required"] else "NO",
        "Cheque No": f_map.get("cheque_number", "—"),
        "Amount": amt_str,
        "Account No": f_map.get("account_number", "—"),
        "Latency": latency,
    })

df = pd.DataFrame(table_rows)

# Render interactive dataframe
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)

# Quick Inspection Selector
st.divider()
st.subheader("Select Cheque to Inspect Details")
c_pick, c_btn = st.columns([4, 1])
with c_pick:
    pick_id = st.selectbox(
        "Choose Cheque:",
        options=[c["id"] for c in cheques],
        format_func=lambda cid: next(f"ID {c['id']} - {c['cheque_identifier']} ({c['status']})" for c in cheques if c["id"] == cid),
    )
with c_btn:
    st.write("")
    st.write("")
    if st.button("Inspect Details 🔍", type="primary"):
        st.session_state["current_cheque_id"] = pick_id
        st.info("Loaded into Extraction Results. Switch to the **Extraction Results** page to view.")
