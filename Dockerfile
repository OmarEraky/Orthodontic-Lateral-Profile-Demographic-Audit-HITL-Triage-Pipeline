# ==============================================================================
# Orthodontic Demographic Audit & HITL Triage Pipeline - Production Container
# Base: Python 3.11 Slim (Debian-based)
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface

# Install essential system libraries for OpenCV headless and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Create required working directories
RUN mkdir -p /app/Dataset /app/audit_outputs /app/.cache/torch /app/.cache/huggingface

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

# Install Python packages (using fast CPU-optimized PyTorch wheels & network timeout protection)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Copy application source code
COPY audit_side_profiles.py triage_server.py apply_triage_decisions.py ./

# Expose port for interactive Triage Dashboard
EXPOSE 8000

# Healthcheck to ensure triage server is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Default command: Start Live Triage Dashboard Server
CMD ["python3", "triage_server.py", "--host", "0.0.0.0", "--port", "8000"]
