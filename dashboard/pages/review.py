"""Teller Review Queue and Human-in-the-Loop Verification Page."""

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

st.title("📋 Teller Review Queue")
st.markdown("Review cheques requiring manual verification due to low confidence, missing fields, or validation warnings.")

# Query cheques currently requiring review
queue_response = backend_client.list_cheques(review_required=True, limit=50)
pending_cheques = queue_response.get("items", [])

if not pending_cheques:
    st.success("🎉 All clear! The review queue is currently empty. No cheques require manual teller intervention.")
    st.info("Cheques that fail confidence thresholds or trigger validation warnings will appear here automatically.")
    st.stop()

st.warning(f"⚠️ **{len(pending_cheques)} Cheque(s) Pending Review**")

# Selector for review item
cheque_map = {c["id"]: f"{c['cheque_identifier']} | Conf: {c['overall_confidence']:.1%} | ID: {c['id']}" for c in pending_cheques}
selected_chq_id = st.selectbox(
    "Select Cheque to Review:",
    options=list(cheque_map.keys()),
    format_func=lambda x: cheque_map[x],
)

cheque = backend_client.get_cheque(selected_chq_id)
if not cheque:
    st.error("Selected cheque could not be loaded.")
    st.stop()

# Layout: Cheque Image on Left, Review Form on Right
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(f"Cheque: {cheque['cheque_identifier']}")
    img_path = Path(cheque["image_path"])
    if img_path.exists():
        cheque_img = Image.open(img_path)
        annotated_img = draw_bounding_boxes_on_cheque(
            cheque_img,
            cheque.get("extracted_fields", []),
            highlight_low_confidence=True,
        )
        st.image(annotated_img, use_container_width=True, caption="Detected Fields (Crimson = Review Flag)")
    else:
        st.error(f"Image not found on disk: {img_path}")

    # Reasons for Review
    st.markdown("#### Triggers Requiring Review")
    validations = cheque.get("validation_results", [])
    flagged_reasons = [v for v in validations if v["validation_status"] in ["INVALID", "WARNING"]]
    if flagged_reasons:
        for r in flagged_reasons:
            st.markdown(
                f"""
                <div class="alert-card alert-warning">
                    <strong>{r['check_type']}</strong>: {r['validation_reason']}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Overall confidence fell below threshold gate.")

with col_right:
    st.subheader("Field Audit & Correction")
    st.caption("Verify and correct any improperly recognized fields:")

    fields = cheque.get("extracted_fields", [])
    corrections: dict[str, str] = {}

    for f in fields:
        fname = f["field_name"]
        curr_val = f["normalized_value"]
        conf = f["confidence"]
        tier = f["confidence_tier"]
        badge_html = get_confidence_badge_html(conf, tier)

        st.markdown(
            f"<div style='display:flex; justify-content:space-between; margin-bottom:-10px;'><strong>{fname.upper()}</strong> {badge_html}</div>",
            unsafe_allow_html=True,
        )
        corrections[fname] = st.text_input(
            label=f"Value for {fname}",
            value=curr_val,
            label_visibility="collapsed",
            key=f"review_corr_{fname}",
        )

    st.divider()
    st.markdown("#### Audit Decision")
    c_acc, c_rej = st.columns(2)

    with c_acc:
        if st.button("✅ Approve & Verify Cheque", type="primary", use_container_width=True):
            # Check if any field was changed
            any_changed = any(corrections[f["field_name"]] != f["normalized_value"] for f in fields)
            decision = "CORRECTED" if any_changed else "ACCEPTED"

            for v in validations:
                backend_client.submit_review_resolution(
                    validation_id=v["id"],
                    resolution=decision,
                    reviewer_id=1,
                    corrected_fields=corrections if any_changed else None,
                )
            st.success(f"Cheque approved as {decision} and cleared for ledger posting.")
            st.rerun()

    with c_rej:
        if st.button("❌ Reject Cheque", use_container_width=True):
            for v in validations:
                backend_client.submit_review_resolution(
                    validation_id=v["id"],
                    resolution="REJECTED",
                    reviewer_id=1,
                )
            st.error("Cheque rejected and archived.")
            st.rerun()
