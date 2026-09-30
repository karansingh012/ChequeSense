"""Executive Banking Analytics & Telemetry Dashboard Page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import (
    BANKING_CSS,
    backend_client,
    format_currency_inr,
)

st.markdown(BANKING_CSS, unsafe_allow_html=True)

st.title("📊 Banking Analytics & Executive Dashboard")
st.markdown("Operational intelligence, straight-through processing (STP) performance, and financial clearing trends.")

# Refresh & interval controls
col_ctrl1, col_ctrl2 = st.columns([4, 1])
with col_ctrl1:
    interval = st.segmented_control("Aggregation Window:", ["day", "week", "hour"], default="day") or "day"
with col_ctrl2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# Fetch Analytics from Backend
with st.spinner("Compiling analytics and trends from database..."):
    summary = backend_client.get_analytics_summary()
    trends = backend_client.get_analytics_trends(interval=interval)

vol = summary.get("volume", {})
amt = summary.get("amounts", {})
rev = summary.get("reviews", {})
conf = summary.get("confidence", {})
val = summary.get("validation", {})
lat = summary.get("latency", {})
fields = summary.get("fields", [])

# 1. Executive Metric Cards
st.subheader("Key Operational Indicators (KPIs)")
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-title">Total Cheques</div>
            <div class="metric-value">{vol.get('total_cheques', 0):,}</div>
            <div class="metric-sub">{vol.get('verified_count', 0):,} Verified</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-title">Total Amount</div>
            <div class="metric-value">{format_currency_inr(amt.get('total_recognized_amount', 0.0))}</div>
            <div class="metric-sub">Mean: {format_currency_inr(amt.get('average_amount', 0.0))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    stp_rate = round(100.0 - rev.get("manual_review_rate_pct", 0.0), 1)
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-title">STP Rate</div>
            <div class="metric-value" style="color:#059669;">{stp_rate}%</div>
            <div class="metric-sub">Auto-Cleared</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-title">Review Rate</div>
            <div class="metric-value" style="color:#d97706;">{rev.get('manual_review_rate_pct', 0.0):.1f}%</div>
            <div class="metric-sub">{rev.get('review_required_count', 0)} in Queue</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m5:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-title">Avg Confidence</div>
            <div class="metric-value">{conf.get('mean_confidence', 0.0):.1%}</div>
            <div class="metric-sub">Median: {conf.get('median_confidence', 0.0):.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m6:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-title">Median Latency</div>
            <div class="metric-value">{lat.get('p50_latency_ms', 0.0):.0f} ms</div>
            <div class="metric-sub">p90: {lat.get('p90_latency_ms', 0.0):.0f} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# 2. Temporal Trends
st.subheader("Processing Volume & Monetary Clearing Trends")
c_tr1, c_tr2 = st.columns(2)

vol_trend_data = trends.get("volume_trends", [])
amt_trend_data = trends.get("amount_trends", [])

with c_tr1:
    st.markdown("#### Cheques Processed vs. Flagged for Review")
    if vol_trend_data:
        df_vol = pd.DataFrame(vol_trend_data)
        df_vol["Date"] = df_vol["period"]
        st.bar_chart(
            df_vol,
            x="Date",
            y=["verified_count", "review_required_count"],
            color=["#10b981", "#f59e0b"],
            use_container_width=True,
        )
    else:
        st.info("No volume trend data available yet.")

with c_tr2:
    st.markdown("#### Monetary Value Cleared (₹)")
    if amt_trend_data:
        df_amt = pd.DataFrame(amt_trend_data)
        df_amt["Date"] = df_amt["period"]
        st.line_chart(
            df_amt,
            x="Date",
            y="total_amount",
            color="#2563eb",
            use_container_width=True,
        )
    else:
        st.info("No monetary clearing trend data available yet.")

st.divider()

# 3. Model Confidence Distribution & Validation Gates
c_cd1, c_cd2 = st.columns(2)

with c_cd1:
    st.subheader("Model Confidence Tier Distribution")
    buckets = conf.get("distribution_buckets", {})
    if buckets:
        df_buckets = pd.DataFrame({
            "Confidence Tier": list(buckets.keys()),
            "Cheque Count": list(buckets.values()),
        })
        st.bar_chart(
            df_buckets,
            x="Confidence Tier",
            y="Cheque Count",
            color="#0ea5e9",
            use_container_width=True,
        )
    else:
        st.info("No confidence distribution data available.")

with c_cd2:
    st.subheader("Validation Rules & Gate Pass Rates")
    checks = val.get("checks_by_type", {})
    if checks:
        rows_val = []
        for ctype, cdata in checks.items():
            rows_val.append({
                "Rule Type": ctype,
                "Total Checks": cdata.get("total_checks", 0),
                "Valid": cdata.get("valid_count", 0),
                "Warnings / Invalid": cdata.get("invalid_count", 0) + cdata.get("warning_count", 0),
                "Pass Rate (%)": f"{cdata.get('pass_rate_pct', 0.0):.1f}%",
            })
        st.dataframe(pd.DataFrame(rows_val), use_container_width=True, hide_index=True)
    else:
        st.info("No validation check statistics available.")

st.divider()

# 4. Field-Level Recognition Accuracy Table
st.subheader("Field-Level Recognition Performance")
if fields:
    df_f = pd.DataFrame([
        {
            "Field Name": f["field_name"].upper(),
            "Total Extractions": f["total_extractions"],
            "Avg Confidence": f"{f['avg_confidence']:.2%}",
            "High Tier": f["high_tier_count"],
            "Medium Tier": f["medium_tier_count"],
            "Low Tier": f["low_tier_count"],
            "OCR Engine": f["ocr_method_count"],
            "CNN Recognizer": f["recognizer_method_count"],
        }
        for f in fields
    ])
    st.dataframe(df_f, use_container_width=True, hide_index=True)
else:
    st.info("No field-level metrics logged.")

# 5. Regulatory Notice
st.caption(
    "🔒 **Regulatory Compliance Notice:** Any statistical outliers identified by ChequeSense function strictly as operational "
    "REVIEW SIGNALS for secondary teller review. Statistical anomalies are never classified as fraud without independent human and legal investigation."
)
