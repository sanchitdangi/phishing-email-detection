# ─── Multi-Stage Docker Build ──────────────────────────────────────────────────
# Using python:3.11-slim as a lightweight, secure base.

# Stage 1: Build & Dependency Installation
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system compiler/libraries required for LightGBM and pip compilations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a wheel directory to enable cached multi-stage layer copy
RUN pip install --no-cache-dir --user -r requirements.txt


# Stage 2: Final Service Runtime Environment
FROM python:3.11-slim AS runner

WORKDIR /app

# LightGBM requires libgomp1 at runtime (without it, import lightgbm throws OSError)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python package binaries from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy codebase
COPY api/ ./api/
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY README.md .

# Port exposures
EXPOSE 8000
EXPOSE 8501

# Default runtime: will be overridden in docker-compose.yml for frontend/backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
