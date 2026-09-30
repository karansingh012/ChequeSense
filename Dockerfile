# ==============================================================================
# ChequeSense Multi-Service Production Dockerfile
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered production logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Install system dependencies:
# - tesseract-ocr & language packs for document OCR
# - libgl1 & libglib2.0-0 for OpenCV headless
# - curl & postgresql-client for healthchecks & DB diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install CPU-optimized PyTorch first for fast layer caching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create application non-root user and directories for secure execution
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/uploads /app/models /app/artifacts /app/logs && \
    chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser api/ /app/api/
COPY --chown=appuser:appuser dashboard/ /app/dashboard/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser alembic.ini /app/
COPY --chown=appuser:appuser README.md /app/

# Set environment paths and switch to non-root user
ENV PYTHONPATH=/app \
    UPLOAD_DIR=/app/data/uploads \
    DETECTOR_MODEL_PATH=/app/models/field_detector/best_model.pt \
    RECOGNIZER_MODEL_PATH=/app/models/recognizer/best_model.pt \
    TESSERACT_CMD=/usr/bin/tesseract

USER appuser

# Expose API (8000) and Streamlit Dashboard (8501)
EXPOSE 8000 8501

# Default entrypoint starts the FastAPI backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
