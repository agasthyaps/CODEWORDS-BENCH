# Build stage for UI
FROM node:20-slim AS ui-builder

WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm ci

COPY ui/ ./
RUN npm run build

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY words.txt ./words.txt

# Copy built UI from builder stage
COPY --from=ui-builder /app/ui/dist ./ui/dist

# Create data directory (will be mounted as volume in Railway)
RUN mkdir -p /app/benchmark_results

# Set environment variables
ENV BENCHMARK_DATA_DIR=/app/benchmark_results
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the API server (Railway injects PORT env var)
CMD uvicorn src.ui_api.app:app --host 0.0.0.0 --port ${PORT:-8000}
