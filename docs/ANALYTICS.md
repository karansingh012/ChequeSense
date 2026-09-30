# ChequeSense Banking Analytics Layer (Phase 8)

## 1. Overview & Architectural Principles

The ChequeSense Banking Analytics Layer computes operational business intelligence, straight-through processing (STP) ratios, monetary clearing volumes, validation gate effectiveness, and inference performance directly from the normalized PostgreSQL database.

### Core Principles & Invariants:
1. **Grounded Strictly in Database Schema**:
   All metrics and trends are computed **strictly from fields actually persisted** in the normalized PostgreSQL schema (`cheques`, `extracted_fields`, `predictions`, `validation_results`, and `processing_runs`). No ungrounded or synthetic synthetic assumptions are made.
2. **Deterministic & Reusable Querying**:
   Analytics are exposed as both **composable Python services** (via SQLAlchemy 2.0 ORM) and **reusable standard SQL queries** ready for integration with enterprise BI dashboards (Metabase, Apache Superset, Tableau, PowerBI).
3. **Leading-Zero & Monetary Integrity**:
   Numerical aggregations properly cast sanitized string representations of amounts without losing numerical precision or corrupting leading zeros on identifier fields.
4. **Strict Compliance Rule on Anomaly Detection**:
   > [!IMPORTANT]
   > **ANOMALY DETECTION IS A REVIEW SIGNAL ONLY.**
   > Under no circumstances does ChequeSense classify or label a cheque as "fraud" or "counterfeit" solely because it is a statistical anomaly.
   > High-value transactions, corporate payrolls, and atypical cheque layouts are routine, legitimate banking operations. Statistical outliers trigger an operational `ReviewSignal` for secondary human teller review, never an automated fraud accusation.

---

## 2. Key Performance Indicators (KPIs) & Metric Definitions

### 2.1 Volume & Straight-Through Processing (STP)
- **Total Processed Volume ($N$)**: Total count of cheques processed over the reporting period.
  $$\text{Total Cheques} = \sum \mathbf{1}$$
- **Straight-Through Processing (STP) Rate**: The percentage of cheques that cleared with high confidence without requiring any manual human teller intervention.
  $$\text{STP Rate (\%)} = \frac{\text{Auto-Cleared Cheques}}{\text{Total Cheques}} \times 100$$
- **Manual Review Rate**: The percentage of cheques flagged for teller verification.
  $$\text{Review Rate (\%)} = \frac{\text{Cheques with } \text{review\_required} = \text{TRUE}}{\text{Total Cheques}} \times 100$$
- **Status Breakdown**: Frequency distribution across operational statuses (`PROCESSED`, `VERIFIED`, `REVIEW_REQUIRED`, `INVALID`).

### 2.2 Monetary Clearing & Financial Metrics
Derived from `extracted_fields` where `field_name = 'amount'`:
- **Total Recognized Value**: $\sum \text{Amount}_i$ across all successfully extracted cheque amounts.
- **Mean Amount**: $\mu_{\text{amount}} = \frac{1}{n}\sum \text{Amount}_i$.
- **Median Amount**: 50th percentile of recognized amounts, providing resilience against extreme skew from large corporate cheques.
- **Range & Dispersion**: Minimum, maximum, and standard deviation ($\sigma_{\text{amount}}$).

### 2.3 Model Confidence Distribution
Derived from `cheques.overall_confidence`:
- **Summary Metrics**: Mean, median, minimum, maximum, and standard deviation.
- **Histogram Buckets**:
  - `< 0.50`: Severe ambiguity or degraded image.
  - `0.50 - 0.70`: Low confidence (always flagged for review).
  - `0.70 - 0.85`: Medium confidence (conditional review gate).
  - `0.85 - 0.95`: High confidence (straight-through eligible).
  - `>= 0.95`: Very high confidence (flawless extraction).

### 2.4 Field-Level Recognition Performance
Evaluates spatial detection and text extraction efficiency per field (`cheque_number`, `amount`, `date`, `payee_name`, etc.):
- **Extraction Volume**: Total occurrences extracted.
- **Composite Confidence**: Propagated confidence across detection and extraction.
- **Detection Confidence**: Spatial Faster R-CNN bounding box score.
- **Extraction Confidence**: OCR engine or Character CNN recognizer score.
- **Tier Breakdown**: High / Medium / Low distribution.
- **Engine Routing Breakdown**: Number of extractions routed through Tesseract OCR vs. custom digit CNN recognizer.

### 2.5 Validation Failure Analysis
Derived from `validation_results`:
- **Pass Rate**: Percentage of validation checks evaluated as `VALID`.
- **Gate Breakdown**: Pass/fail/warning counts across `DATE_VALIDITY`, `FORMAT`, `CONSISTENCY`, `REQUIRED_FIELD`, and `CONFIDENCE_GATE`.
- **Top Review Reasons**: Ranked frequency of specific compliance triggers (e.g. stale cheque $> 90$ days, missing signature, mismatched payee).

### 2.6 Pipeline Latency & SLA Telemetry
Derived from `processing_runs.total_latency_ms`:
- **Average Latency**: Mean execution time in milliseconds.
- **Percentiles**: p50 (median), p90, and p99 tail latencies.
- **SLA Compliance**: Verification whether 90% of cheques are processed under bank SLA thresholds (e.g., $< 250\text{ ms}$).

---

## 3. Reusable SQL Queries

The repository exposes standardized, performant SQL queries in [`src/analytics/queries.py`](file:///Users/karansingh/ChequeSense/src/analytics/queries.py#L22):

### 3.1 Daily Volume and Manual Review Trend
```sql
SELECT
    DATE_TRUNC('day', created_at) AS date_bucket,
    COUNT(*) AS total_processed,
    COUNT(CASE WHEN status = 'VERIFIED' THEN 1 END) AS verified_count,
    COUNT(CASE WHEN review_required THEN 1 END) AS review_required_count,
    ROUND(COUNT(CASE WHEN review_required THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS review_rate_pct,
    ROUND(AVG(overall_confidence)::numeric, 4) AS avg_confidence
FROM cheques
WHERE (:start_date IS NULL OR created_at >= :start_date)
  AND (:end_date IS NULL OR created_at <= :end_date)
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY date_bucket ASC;
```

### 3.2 Recognized Monetary Clearing Statistics
```sql
SELECT
    COUNT(*) AS amount_field_count,
    ROUND(SUM(NULLIF(regexp_replace(normalized_value, '[^0-9.]', '', 'g'), '')::numeric), 2) AS total_amount,
    ROUND(AVG(NULLIF(regexp_replace(normalized_value, '[^0-9.]', '', 'g'), '')::numeric), 2) AS avg_amount,
    MIN(NULLIF(regexp_replace(normalized_value, '[^0-9.]', '', 'g'), '')::numeric) AS min_amount,
    MAX(NULLIF(regexp_replace(normalized_value, '[^0-9.]', '', 'g'), '')::numeric) AS max_amount
FROM extracted_fields
WHERE field_name = 'amount'
  AND normalized_value ~ '^[0-9]+(\.[0-9]+)?$';
```

### 3.3 Field-Level Recognition & OCR Engine Breakdown
```sql
SELECT
    field_name,
    COUNT(*) AS occurrences,
    ROUND(AVG(confidence)::numeric, 4) AS avg_confidence,
    ROUND(AVG(detection_confidence)::numeric, 4) AS avg_detection_confidence,
    ROUND(AVG(extraction_confidence)::numeric, 4) AS avg_extraction_confidence,
    COUNT(CASE WHEN confidence_tier = 'HIGH' THEN 1 END) AS high_tier_count,
    COUNT(CASE WHEN confidence_tier = 'MEDIUM' THEN 1 END) AS medium_tier_count,
    COUNT(CASE WHEN confidence_tier = 'LOW' THEN 1 END) AS low_tier_count,
    COUNT(CASE WHEN extraction_method = 'ocr' THEN 1 END) AS ocr_count,
    COUNT(CASE WHEN extraction_method = 'recognizer' THEN 1 END) AS recognizer_count
FROM extracted_fields
GROUP BY field_name
ORDER BY occurrences DESC;
```

### 3.4 Pipeline Latency Percentiles
```sql
SELECT
    COUNT(*) AS total_runs,
    ROUND(AVG(total_latency_ms)::numeric, 2) AS avg_latency_ms,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_latency_ms)::numeric, 2) AS p50_latency_ms,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY total_latency_ms)::numeric, 2) AS p90_latency_ms,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_latency_ms)::numeric, 2) AS p99_latency_ms,
    ROUND(MIN(total_latency_ms)::numeric, 2) AS min_latency_ms,
    ROUND(MAX(total_latency_ms)::numeric, 2) AS max_latency_ms
FROM processing_runs;
```

---

## 4. Anomaly Detection as an Operational Review Signal

To balance risk management with banking compliance, ChequeSense implements statistical outlier detection using both **parametric (Z-score)** and **non-parametric (Interquartile Range IQR)** methods:

### 4.1 Amount Outlier Detection
1. **Minimum Sample Gate**: Requires at least 10 historical cheque amounts to ensure stable distribution statistics.
2. **Evaluation Metric**:
   - $Z = \frac{x - \mu}{\sigma}$
   - Outlier condition: $x > Q_3 + (2.5 \times \text{IQR})$ or $Z \ge 3.0$.
3. **Signal Generation**:
   Produces a [`ReviewSignal`](file:///Users/karansingh/ChequeSense/src/analytics/trends.py#L38) with:
   - `signal_type`: `"AMOUNT_OUTLIER_REVIEW_SIGNAL"`
   - `severity`: `"HIGH"` if $Z \ge 4.0$, else `"MEDIUM"`
   - `is_fraud_claim`: `False` (hardcoded invariant)
   - `message`: Advises tellers of atypical disbursement value for verification of account funds.

### 4.2 Latency Spike Detection
1. Detects pipeline slowdowns where inference latency exceeds $\mu_{\text{lat}} + 2.5\sigma_{\text{lat}}$.
2. Flags internal telemetry for hardware monitoring or model checkpoint review.

---

## 5. Python API & Report Generation Examples

### 5.1 Computing the Executive Dashboard
```python
from src.database.connection import db_session_scope
from src.analytics import calculate_executive_dashboard

with db_session_scope() as session:
    summary = calculate_executive_dashboard(session)
    print(f"Total Cheques: {summary.volume.total_cheques}")
    print(f"Review Rate: {summary.reviews.manual_review_rate_pct:.1f}%")
    print(f"Total Amount Cleared: Rs. {summary.amounts.total_recognized_amount:,.2f}")
    print(f"Median Latency: {summary.latency.p50_latency_ms:.1f} ms")
```

### 5.2 Generating Markdown Audit Briefings
```python
from src.analytics import generate_executive_report_markdown

with db_session_scope() as session:
    summary = calculate_executive_dashboard(session)
    markdown_doc = generate_executive_report_markdown(summary)
    with open("artifacts/analytics_briefing.md", "w") as f:
        f.write(markdown_doc)
```

### 5.3 Exporting Bank Reconciliation CSV
```python
from src.analytics import generate_cheque_reconciliation_csv

with db_session_scope() as session:
    csv_content = generate_cheque_reconciliation_csv(session, limit=1000)
    with open("artifacts/daily_ledger_reconciliation.csv", "w") as f:
        f.write(csv_content)
```

---

## 6. Testing & Quality Assurance

Unit and integration tests for the analytics layer are located in [`tests/test_analytics_layer.py`](file:///Users/karansingh/ChequeSense/tests/test_analytics_layer.py).

Test suite covers:
- SQL query syntax and completeness.
- Volume, status breakdown, and manual review rate calculations.
- Amount statistics (sum, mean, median, min, max, std).
- Confidence distribution and histogram bucket generation.
- Field recognition accuracy and OCR vs recognizer routing.
- Validation gate summaries and top review triggers.
- Latency percentiles (p50, p90, p99).
- Daily volume and monetary amount trends over time.
- Statistical anomaly detection (verifying that normal amounts pass and outliers trigger review signals without fraud accusations).
- Export formats (Markdown executive briefing, SLA evaluation JSON, CSV ledger exports).
