"""Cheque Ingestion and Pipeline Processing Page."""

from __future__ import annotations

import io
from pathlib import Path
from PIL import Image
import streamlit as st

from dashboard.components import BANKING_CSS, backend_client

st.markdown(BANKING_CSS, unsafe_allow_html=True)

st.title("📤 Cheque Ingestion & Processing")
st.markdown("Upload cheque images for automated field extraction, validation rule checks, and straight-through processing.")

# Sample Cheque Quick Selectors
SAMPLE_DIR = Path("artifacts/pipeline/test_cheques")
sample_files = list(SAMPLE_DIR.glob("*.png")) if SAMPLE_DIR.exists() else []

tab_upload, tab_sample = st.tabs(["Upload Cheque Image", "Select Pre-loaded Sample"])

selected_image_bytes = None
selected_filename = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a cheque image (PNG, JPG, JPEG, TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "webp"],
        help="Upload clear scans or photos of handwritten/printed cheques.",
    )
    if uploaded_file is not None:
        selected_image_bytes = uploaded_file.getvalue()
        selected_filename = uploaded_file.name

with tab_sample:
    if sample_files:
        chosen_sample = st.selectbox(
            "Select sample cheque from test suite:",
            options=sample_files,
            format_func=lambda p: p.name,
        )
        if chosen_sample:
            if st.button("Load Selected Sample"):
                with open(chosen_sample, "rb") as f:
                    selected_image_bytes = f.read()
                selected_filename = chosen_sample.name
    else:
        st.info("No pre-loaded sample cheques found in artifacts directory.")

# If an image is selected, show preview and actions
if selected_image_bytes:
    st.divider()
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Original Cheque Preview")
        preview_img = Image.open(io.BytesIO(selected_image_bytes))
        st.image(preview_img, use_container_width=True, caption=f"Selected: {selected_filename} ({preview_img.size[0]}x{preview_img.size[1]} px)")

    with col2:
        st.subheader("Ingestion & Analysis")
        st.markdown(
            """
            Click below to upload the cheque to secure storage and run:
            - **Spatial Field Detection** (Faster R-CNN)
            - **Character/Digit Recognition** (CNN Recognizer)
            - **OCR Text Extraction** (Tesseract)
            - **Compliance & Format Validation**
            """
        )

        auto_process = st.checkbox("Automatically run ML Pipeline after upload", value=True)

        if st.button("🚀 Ingest & Process Cheque", type="primary", use_container_width=True):
            with st.spinner("Uploading and analyzing cheque image..."):
                try:
                    # 1. Ingest image
                    upload_res = backend_client.upload_cheque(selected_filename or "cheque.png", selected_image_bytes)
                    cheque_id = upload_res["id"]
                    cheque_ident = upload_res["cheque_identifier"]
                    st.success(f"Cheque recorded successfully: **{cheque_ident}** (ID: {cheque_id})")

                    # 2. Process if requested
                    if auto_process:
                        with st.spinner("Executing ChequeSense pipeline & validation rules..."):
                            proc_res = backend_client.process_cheque(cheque_id)
                            st.session_state["current_cheque_id"] = cheque_id
                            st.success(f"Processing complete! Status: **{proc_res['status']}** (Confidence: {proc_res['overall_confidence']:.2%})")

                            if proc_res["review_required"]:
                                st.warning("⚠️ This cheque was flagged for human teller review.")

                            st.info("👉 Switch to the **Extraction Results** tab in the sidebar to inspect detected fields and bounding boxes.")
                    else:
                        st.session_state["current_cheque_id"] = cheque_id
                        st.info(f"Cheque saved. You can process it from the Cheque History page.")

                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")
