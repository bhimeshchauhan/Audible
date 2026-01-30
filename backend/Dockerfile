# =============================================================================
# Production Dockerfile for FastAPI Backend
# Uses gunicorn with uvicorn workers for production performance
# =============================================================================

# -----------------------------------------------------------------------------
# Build stage: Install dependencies and download spaCy model
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# -----------------------------------------------------------------------------
# Production stage: Minimal runtime image
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS production

# Security: Run as non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Production defaults
    ENVIRONMENT=production \
    DEBUG=false \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000 \
    SERVER_WORKERS=4

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appgroup backend/ ./backend/
COPY --chown=appuser:appgroup requirements.txt .

# Create temp directory for uploads with proper permissions
RUN mkdir -p /tmp/uploads && chown appuser:appgroup /tmp/uploads

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run gunicorn with uvicorn workers
# - workers: Number of worker processes (configurable via SERVER_WORKERS)
# - worker-class: Use uvicorn's async worker
# - bind: Listen on all interfaces at port 8000
# - timeout: Worker timeout (important for long PDF processing)
# - graceful-timeout: Time to finish requests on shutdown
# - keep-alive: Keep-alive timeout for connections
CMD ["sh", "-c", "gunicorn backend.main:app \
    --workers ${SERVER_WORKERS:-4} \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile -"]
