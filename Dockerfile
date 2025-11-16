# ============================================================
# GroundedDINO-VL - CUDA 12.8 Runtime Image
# Multi-stage build: compile in "builder", run in "runtime"
# ============================================================

# -----------------------------
# Stage 1: Builder
# -----------------------------
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

# System dependencies: Python, build tools, OpenCV libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential git \
    libgl1 libglib2.0-0 \
    gcc \
    ca-certificates curl wget \
 && rm -rf /var/lib/apt/lists/*

# Make "python3.10" the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/groundeddino_vl

# Copy source into builder image
COPY . .

# Install Python dependencies and GroundedDINO-VL (with CUDA ops)
RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir .

# Sanity check: import inside builder
RUN python3 -c "import groundeddino_vl; print('GroundedDINO-VL import OK in builder image')"


# -----------------------------
# Stage 2: Runtime
# -----------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies: Python + OpenCV system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    libgl1 libglib2.0-0 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip

WORKDIR /workspace/groundeddino_vl
# Copy installed Python packages and entry points from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Optional: also copy the source tree for debugging/examples
COPY . .

# Create a test script to verify installation & environment inside the container
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'echo "========================================"' \
'echo " GroundedDINO-VL Docker Environment Check"' \
'echo "========================================"' \
'echo ""' \
'echo "Python version:"' \
'python3 --version' \
'echo ""' \
'echo "pip version:"' \
'python3 -m pip --version' \
'echo ""' \
'echo "Checking GroundedDINO-VL import..."' \
'python3 -c "import groundeddino_vl; print(\"✓ GroundedDINO-VL import successful\")"' \
'echo ""' \
'echo "✓ Installation test passed!"' \
> /usr/local/bin/test.sh && chmod +x /usr/local/bin/test.sh

# Healthcheck: ensure we can still import GroundedDINO-VL
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import groundeddino_vl; print('OK')" || exit 1

# Default command runs the test script (override in docker run as needed)
CMD ["/usr/local/bin/test.sh"]

# Labels (public repo friendly)
LABEL org.opencontainers.image.source="https://github.com/ghostcipher1/GroundedDINO-VL"
LABEL org.opencontainers.image.description="GroundedDINO-VL with PyTorch 2.7 + CUDA 12.8 support (Built from source)"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.vendor="ghostcipher1"
LABEL org.opencontainers.image.title="GroundedDINO-VL"
LABEL org.opencontainers.image.version="2025.11.0"
LABEL maintainer="ghostcipher1"

