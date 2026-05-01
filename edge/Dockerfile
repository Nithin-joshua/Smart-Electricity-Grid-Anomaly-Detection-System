# ==============================================================================
# Dockerfile: Smart Grid Edge Processor
# Builds a minimal production image for the MQTT stream consumer and
# anomaly detection inference engine.
# ==============================================================================

# --- Stage: Runtime ---
FROM python:3.11-slim AS runtime

# Metadata labels for container registry and operational tooling.
LABEL maintainer="smart-grid-team"
LABEL service="edge-processor"
LABEL description="Real-time MQTT stream processor with TFLite anomaly inference"

# Prevent .pyc files and enable unbuffered logging for docker logs.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the application working directory inside the container.
WORKDIR /app

# --- Dependency Installation (cached layer) ---
# Copy requirements first for Docker layer caching efficiency.
COPY edge/requirements.txt /app/edge/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/edge/requirements.txt

# --- Application Code ---
# Copy only the modules needed by the edge processor.
COPY config/ /app/config/
COPY edge/ /app/edge/
COPY utils/ /app/utils/
COPY model/artifacts/ /app/model/artifacts/

# --- Security: Non-root user ---
RUN groupadd --gid 1000 griduser \
    && useradd --uid 1000 --gid griduser --shell /bin/bash --create-home griduser
USER griduser

# --- Entrypoint ---
# Default: launch the edge stream processor.
CMD ["python", "-m", "edge.edge_stream_processor"]
