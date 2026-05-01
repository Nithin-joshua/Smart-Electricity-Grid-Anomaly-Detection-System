# ==============================================================================
# FILE: config/mqtt_settings.py
# PURPOSE: MQTT broker connection and topic configuration for the smart grid
#          telemetry pipeline. Centralizes all broker-related parameters.
# ==============================================================================

"""
MQTT Broker Settings
--------------------
Defines the connection parameters and topic hierarchy for the MQTT-based
telemetry transport layer. The topic structure follows a hierarchical
namespace convention (grid/meter/{meter_id}/reading) that enables:

1. Per-meter subscription granularity for edge processors.
2. Wildcard subscriptions (grid/meter/+/reading or grid/meter/#) for
   fleet-wide monitoring dashboards.
3. Clean topic-based routing in downstream consumers (InfluxDB, Grafana).

QoS Level 1 guarantees at-least-once delivery — an appropriate trade-off
between reliability and throughput for smart meter telemetry where
occasional duplicates are acceptable but message loss is not.
"""

import os
from typing import Final

# ==============================================================================
# BROKER CONNECTION
# ==============================================================================

# MQTT broker hostname — defaults to localhost for local development.
# In production, this resolves to the Mosquitto container hostname
# defined in docker-compose.yml via the MQTT_HOST environment variable.
MQTT_HOST: Final[str] = os.environ.get("MQTT_HOST", "localhost")

# MQTT broker listener port — standard unencrypted MQTT port.
# For TLS-secured deployments, override to 8883 via environment variable.
MQTT_PORT: Final[int] = int(os.environ.get("MQTT_PORT", "1883"))

# ==============================================================================
# TOPIC HIERARCHY
# ==============================================================================

# Topic template for individual meter readings.
# The {meter_id} placeholder is substituted at publish time with the
# meter's unique identifier (e.g., "M_023").
# Resulting topic example: grid/meter/M_023/reading
MQTT_TOPIC_TEMPLATE: Final[str] = "grid/meter/{meter_id}/reading"

# ==============================================================================
# QUALITY OF SERVICE
# ==============================================================================

# QoS 1 = At-least-once delivery.
# The broker acknowledges receipt with a PUBACK. If the publisher does
# not receive PUBACK within the retry window, it re-transmits the message.
# This prevents silent data loss on unreliable networks while avoiding
# the overhead of QoS 2 (exactly-once) which requires a 4-step handshake.
MQTT_QOS: Final[int] = 1

# ==============================================================================
# CLIENT IDENTITY
# ==============================================================================

# Base prefix for MQTT client IDs. Each publisher instance appends a
# unique suffix (PID or UUID) to prevent client ID collisions when
# multiple simulator instances connect to the same broker.
MQTT_CLIENT_ID_PREFIX: Final[str] = "smart_grid_sim_"

# ==============================================================================
# RECONNECTION POLICY
# ==============================================================================

# Maximum number of reconnection attempts before the publisher
# transitions to a fatal error state and halts the simulation.
MQTT_MAX_RECONNECT_ATTEMPTS: Final[int] = 10

# Initial delay (seconds) between reconnection attempts.
# Uses exponential backoff: delay = base * 2^attempt, capped at 60s.
MQTT_RECONNECT_BASE_DELAY_SECONDS: Final[float] = 1.0

# Upper bound (seconds) for exponential backoff delay.
MQTT_RECONNECT_MAX_DELAY_SECONDS: Final[float] = 60.0
