# ==============================================================================
# Dockerfile: Smart Grid Meter Simulator
# Builds a minimal production image for the MQTT meter publisher.
# ==============================================================================

# --- Stage: Runtime ---
FROM python:3.11-slim AS runtime

# Metadata labels for container registry and operational tooling.
LABEL maintainer="smart-grid-team"
LABEL service="meter-simulator"
LABEL description="Residential smart meter fleet simulator with MQTT publishing"

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# so logs appear immediately in docker logs (critical for real-time monitoring).
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the application working directory inside the container.
WORKDIR /app

# --- Dependency Installation (cached layer) ---
# Copy requirements first to leverage Docker layer caching.
# This layer only rebuilds when dependencies change, not on every code edit.
COPY simulator/requirements.txt /app/simulator/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/simulator/requirements.txt

# --- Application Code ---
# Copy only the modules needed by the simulator (minimal attack surface).
COPY config/ /app/config/
COPY simulator/ /app/simulator/
COPY utils/ /app/utils/

# --- Security: Non-root user ---
# Run the application as a non-privileged user to limit container escape impact.
RUN groupadd --gid 1000 griduser \
    && useradd --uid 1000 --gid griduser --shell /bin/bash --create-home griduser
USER griduser

# --- Entrypoint ---
# Default: launch the simulator in live MQTT streaming mode.
# Override via docker-compose command or docker run args.
CMD ["python", "-m", "simulator.cli_entrypoint", "--mode", "live"]
