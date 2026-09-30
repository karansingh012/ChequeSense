"""Cheque Extraction Results & Bounding Box Inspection Page."""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import streamlit as st

from dashboard.components import (
    BANKING_CSS,
    backend_client,
    draw_bounding_boxes_on_cheque,
    format_currency_inr,
    get_confidence_badge_html,
    get_status_badge_html,
)

st.markdown(BANKING_CSS, unsafe_allow_html=True)

st.title("🔍 Cheque Extraction & Verification Results")

# Retrieve current cheque ID or select from recent cheques
cheques_data = backend_client.list_cheques(limit=20)
cheque_list = cheques_data.get("items", [])

cheque_options = {c["id"]: f"{c['cheque_identifier']} ({c['status']}) - ID {c['id']}" for c in cheque_list}

selected_id = st.session_state.get("current_cheque_id")
if not selected_id and cheque_list:
    selected_id = cheque_list[0]["id"]

if cheque_options:
    selected_id = st.selectbox(
        "Select Cheque to Inspect:",
        options=list(cheque_options.keys()),
        index=list(cheque_options.keys()).index(selected_id) if selected_id in cheque_options else 0,
        format_func=lambda cid: cheque_options[cid],
    )
    st.session_state["current_cheque_id"] = selected_id
else:
    st.info("No cheques found in system. Please upload a cheque first on the **Upload Cheque** page.")
    st.stop()

# Fetch cheque details
cheque = backend_client.get_cheque(selected_id)
if not cheque:
    st.error(f"Cheque ID {selected_id} not found.")
    st.stop()

# Top Metadata Summary Bar
c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
with c1:
    st.markdown(f"### {cheque['cheque_identifier']}")
    st.caption(f"Created: {cheque['created_at'][:19]} | Hash: `{cheque['image_hash'][:16]}...`")
with c2:
    st.markdown("**Status**")
    st.markdown(get_status_badge_html(cheque["status"]), unsafe_allow_html=True)
with c3:
    st.markdown("**Overall Confidence**")
    tier = "HIGH" if cheque["overall_confidence"] >= 0.85 else ("MEDIUM" if cheque["overall_confidence"] >= 0.70 else "LOW")
    st.markdown(get_confidence_badge_html(cheque["overall_confidence"], tier), unsafe_allow_html=True)
with c4:
    st.markdown("**Review Required**")
    if cheque["review_required"]:
        st.markdown('<span class="badge badge-review">YES - ACTION REQUIRED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-verified">NO (AUTO-CLEARED)</span>', unsafe_allow_html=True)

st.divider()

# Check if processing has taken place
extracted_fields = cheque.get("extracted_fields", [])
if not extracted_fields:
    st.warning("⚠️ This cheque has not been processed through the extraction pipeline yet.")
    if st.button("▶️ Run ML Processing Pipeline Now", type="primary"):
        with st.spinner("Processing cheque image..."):
            backend_client.process_cheque(selected_id)
            st.rerun()
    st.stop()

# Main Visual & Field Layout
col_img, col_fields = st.columns([1, 1])

with col_img:
    st.subheader("Cheque Visual Inspection")
    img_path = Path(cheque["image_path"])

    if img_path.exists():
        original_img = Image.open(img_path)
        view_mode = st.radio(
            "Display Mode:",
            ["Detected Bounding Boxes", "Original Clean Cheque"],
            horizontal=True,
        )

        if view_mode == "Detected Bounding Boxes":
            annotated_img = draw_bounding_boxes_on_cheque(
                original_img,
                extracted_fields,
                highlight_low_confidence=True,
            )
            st.image(annotated_img, use_container_width=True, caption="Spatial Field Detection & Confidence Overlays")
        else:
            st.image(original_img, use_container_width=True, caption=f"Original scan ({cheque['image_width']}x{cheque['image_height']})")
    else:
        st.error(f"Image file not found on disk: {img_path}")

with col_fields:
    st.subheader("Extracted Banking Fields")

    # Check for low-confidence warnings
    low_conf_fields = [f for f in extracted_fields if f["confidence"] < 0.70 or f["confidence_tier"] == "LOW"]
    if low_conf_fields:
        st.markdown(
            f"""
            <div class="alert-card alert-warning">
                <strong>⚠️ Low Confidence Warning:</strong> {len(low_conf_fields)} field(s) have confidence below 70%.
                Please verify their accuracy before final ledger approval.
            </div>
            """,
            unsafe_allow_html=True,
        )

    for field in extracted_fields:
        fname = field["field_name"]
        val = field["normalized_value"]
        raw = field["raw_value"]
        conf = field["confidence"]
        tier = field["confidence_tier"]
        method = field["extraction_method"].upper()
        is_low = (conf < 0.70 or tier == "LOW")

        border_col = "#ef4444" if is_low else "#e2e8f0"
        bg_col = "#fef2f2" if is_low else "#ffffff"

        # Format display value for currency if amount
        display_val = format_currency_inr(val) if fname == "amount" else val

        badge_html = get_confidence_badge_html(conf, tier)
        st.markdown(
            f"""
            <div style="background:{bg_col}; border:1px solid {border_col}; border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; font-size:0.9rem; color:#1e293b; text-transform:uppercase;">{fname}</span>
                    <div>{badge_html}</div>
                </div>
                <div style="font-size:1.25rem; font-weight:700; color:#0f172a; margin:4px 0;">
                    {display_val}
                </div>
                <div style="font-size:0.75rem; color:#64748b; display:flex; justify-content:space-between;">
                    <span>Raw Model Output: <code>{raw}</code></span>
                    <span>Method: {method}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Validation Rules & Review Queue Section
st.divider()
st.subheader("Validation Rules & Compliance Gates")

validations = cheque.get("validation_results", [])
if validations:
    for v in validations:
        v_status = v["validation_status"]
        status_badge = get_status_badge_html(v_status)
        field_tag = f"[{v['field_name'].upper()}] " if v.get("field_name") else ""
        res_tag = f" • Teller Resolution: <strong>{v['review_resolution']}</strong>" if v.get("review_resolution") else ""

        box_cls = "alert-danger" if v_status == "INVALID" else ("alert-warning" if v_status == "WARNING" else "alert-info")

        st.markdown(
            f"""
            <div class="alert-card {box_cls}" style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong>{field_tag}{v['check_type']}</strong>: {v['validation_reason']}{res_tag}
                </div>
                <div>{status_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("No compliance validation records logged for this cheque.")

# Reviewer Correction Expander
if cheque["review_required"] or cheque["status"] == "REVIEW_REQUIRED":
    st.divider()
    with st.expander("📝 Human Reviewer Correction Panel", expanded=True):
        st.markdown("Modify any uncertain fields and record the official teller audit decision:")
        corrections: dict[str, str] = {}

        cols = st.columns(len(extracted_fields)) if len(extracted_fields) <= 4 else st.columns(3)
        for idx, f in enumerate(extracted_fields):
            col_target = cols[idx % len(cols)]
            with col_target:
                corrections[f["field_name"]] = st.text_input(
                    f"Correct {f['field_name']}:",
                    value=f["normalized_value"],
                    key=f"corr_{f['field_name']}",
                )

        rev_decision = st.selectbox(
            "Audit Decision:",
            ["ACCEPTED", "CORRECTED", "REJECTED"],
            help="Accepted: verify as is. Corrected: apply updated values. Rejected: invalid/fraudulent cheque.",
        )

        if st.button("Submit Teller Review Resolution", type="primary"):
            for v in validations:
                backend_client.submit_review_resolution(
                    validation_id=v["id"],
                    resolution=rev_decision,
                    reviewer_id=1,
                    corrected_fields=corrections if rev_decision == "CORRECTED" else None,
                )
            st.success("Review decision saved successfully. Cheque updated.")
            st.rerun()
