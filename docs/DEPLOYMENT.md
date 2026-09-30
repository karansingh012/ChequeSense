# ChequeSense — Deployment Guide

## Overview

ChequeSense is containerised using Docker and Docker Compose. The stack consists of three services:

| Service | Image | Port |
|---|---|---|
| `db` | `postgres:15-alpine` | 5432 (internal) |
| `api` | Custom (Python 3.11-slim) | 8000 |
| `dashboard` | Custom (same image) | 8501 |

---

## Prerequisites

- **Docker** 24.0+
- **Docker Compose** v2 (`docker compose`, not `docker-compose`)
- **Git**
- **Model weights** at `models/field_detector/best_model.pt` and `models/recognizer/best_model.pt`

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/<your-org>/ChequeSense.git
cd ChequeSense

cp .env.example .env
```

Edit `.env` and fill in all required values (see [Environment Variables](#environment-variables) below).

### 2. Place model weights

```bash
mkdir -p models/field_detector models/recognizer
# Copy your trained checkpoints:
cp /path/to/field_detector.pt models/field_detector/best_model.pt
cp /path/to/recognizer.pt      models/recognizer/best_model.pt
```

### 3. Build and start

```bash
docker compose build
docker compose up -d
```

### 4. Verify

```bash
# API health
curl http://localhost:8000/api/v1/health

# API docs
open http://localhost:8000/docs

# Dashboard
open http://localhost:8501
```

---

## Environment Variables

Copy `.env.example` to `.env` and set all values:

```bash
# PostgreSQL
POSTGRES_DB=chequesense
POSTGRES_USER=chequesense_user
POSTGRES_PASSWORD=<strong-password>

# Application database URL (used by API and dashboard)
DATABASE_URL=postgresql://chequesense_user:<strong-password>@db:5432/chequesense

# JWT Authentication
SECRET_KEY=<64-char-random-hex>
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Initial admin credentials (seeded at startup)
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=<strong-admin-password>

# Model paths (inside container)
DETECTOR_MODEL_PATH=/app/models/field_detector/best_model.pt
RECOGNIZER_MODEL_PATH=/app/models/recognizer/best_model.pt
DEVICE=cpu

# OCR
TESSERACT_CMD=/usr/bin/tesseract

# File upload
UPLOAD_DIR=/app/data/uploads
MAX_FILE_SIZE_MB=10

# API URL for dashboard
API_BASE_URL=http://api:8000
```

### Generating a Secure `SECRET_KEY`

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Services Detail

### `db` (PostgreSQL)

- Image: `postgres:15-alpine`
- Data volume: `postgres_data` (persists across restarts)
- Health check: `pg_isready -U $POSTGRES_USER`
- The API service waits for the DB health check before starting

### `api` (FastAPI)

- Built from `Dockerfile`
- Runs as non-root user `appuser` (UID 1000)
- Runs `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- On startup: runs Alembic migrations + seeds admin user
- Health check: `curl http://localhost:8000/api/v1/health`
- Model directory mounted read-only: `./models:/app/models:ro`
- Upload directory mounted with write access: `uploads_data:/app/data/uploads`

### `dashboard` (Streamlit)

- Same Docker image as the API
- Runs `streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0`
- Communicates with the API service via `http://api:8000` (internal Docker network)
- Health check: `curl http://localhost:8501/_stcore/health`

---

## Docker Compose Commands

```bash
# Build all service images
docker compose build

# Start all services in background
docker compose up -d

# Start and follow logs
docker compose up

# Follow logs for a specific service
docker compose logs -f api
docker compose logs -f dashboard
docker compose logs -f db

# Stop services (keep volumes)
docker compose down

# Stop services and remove all data (DESTRUCTIVE)
docker compose down -v

# Restart a single service
docker compose restart api

# View running containers
docker compose ps

# Open a shell in the API container
docker compose exec api bash

# Run Alembic migrations manually
docker compose exec api alembic upgrade head

# Check database connectivity from the API container
docker compose exec api python -c \
  "from src.database.connection import engine; engine.connect(); print('DB OK')"
```

---

## Volumes

| Volume | Purpose |
|---|---|
| `postgres_data` | PostgreSQL data directory (persists across `docker compose down`) |
| `uploads_data` | Uploaded cheque images (shared between API and dashboard) |
| `./models` (bind mount) | Read-only model weights — mounted from host, not baked into image |

---

## Production Recommendations

1. **HTTPS:** Place an nginx or Traefik reverse proxy in front of the API and dashboard. Configure TLS termination at the proxy layer.

2. **Secrets management:** For production, use Docker Secrets, HashiCorp Vault, or your cloud provider's secret manager instead of `.env` files.

3. **Database backups:** Configure regular `pg_dump` backups of the `postgres_data` volume.

4. **Resource limits:** Add `deploy.resources.limits` in `docker-compose.yml` to cap CPU and memory per service (especially for the API, which loads PyTorch).

5. **GPU acceleration:** If a CUDA-capable GPU is available, change `DEVICE=cuda` and use the `pytorch/pytorch:latest` CUDA base image. Update `torch` installation in the Dockerfile accordingly.

6. **Model storage:** In production, store model weights in object storage (AWS S3, GCS) and download them at container startup rather than mounting from the host.

7. **Log aggregation:** Configure Docker logging drivers to forward logs to a centralised logging system (Elasticsearch, CloudWatch, etc.).

---

## Troubleshooting

### API fails to start with "could not connect to server"

The database may not be ready. Wait 10–20 seconds and retry, or inspect the DB logs:

```bash
docker compose logs db
```

### API returns 500 on `/process` — "model file not found"

Check that the model weights are correctly mounted:

```bash
docker compose exec api ls /app/models/field_detector/
docker compose exec api ls /app/models/recognizer/
```

### Dashboard shows "Connection refused"

Ensure the API container is running and healthy before the dashboard tries to connect:

```bash
docker compose ps
docker compose logs api
```

### Database schema is out of date

Run migrations manually:

```bash
docker compose exec api alembic upgrade head
```

---

## Security Notes

- Never commit `.env` to version control (listed in `.gitignore` and `.dockerignore`).
- Dataset files are excluded from the Docker image via `.dockerignore`.
- Model weights should not contain training data or PII.
- The Docker image runs as a non-root user (`appuser`, UID 1000).
