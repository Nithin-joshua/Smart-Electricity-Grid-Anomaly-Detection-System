# ==============================================================================
# FILE: simulator/mqtt_stream_publisher.py
# PURPOSE: MQTT transport layer for publishing smart meter telemetry.
#          Handles broker connection, reconnection with exponential backoff,
#          JSON serialization, and per-meter topic routing.
# ==============================================================================

"""
MQTT Meter Stream Publisher
----------------------------
Manages the MQTT client lifecycle for the smart grid simulation layer.
Publishes JSON-encoded meter readings to per-meter topics using QoS 1
(at-least-once delivery) to prevent silent data loss.

Connection resilience is achieved through exponential backoff reconnection
with a configurable retry ceiling. All connection state transitions are
logged for operational observability.
"""

import json
import time
import uuid
from typing import Dict, Any, Optional

import paho.mqtt.client as mqtt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.mqtt_settings import (
    MQTT_HOST,
    MQTT_PORT,
    MQTT_TOPIC_TEMPLATE,
    MQTT_QOS,
    MQTT_CLIENT_ID_PREFIX,
    MQTT_MAX_RECONNECT_ATTEMPTS,
    MQTT_RECONNECT_BASE_DELAY_SECONDS,
    MQTT_RECONNECT_MAX_DELAY_SECONDS,
)
from utils.logging_setup import get_module_logger

# Module-scoped logger for MQTT transport events.
mqtt_publisher_logger = get_module_logger("mqtt_stream_publisher")


class MQTTMeterStreamPublisher:
    """
    Publishes smart meter readings to an MQTT broker.

    Encapsulates all MQTT client management including:
    - Unique client ID generation (prevents broker-side collisions)
    - Asynchronous connection with callback-based state tracking
    - Exponential backoff reconnection on connection loss
    - JSON serialization and per-meter topic routing
    - Graceful shutdown with clean disconnect
    """

    def __init__(self) -> None:
        """
        Initialize the MQTT publisher with a unique client identity.

        The client ID combines a fixed prefix with a UUID4 suffix to
        ensure uniqueness even when multiple simulator instances connect
        to the same broker (e.g., during parallel testing).
        """
        # Generate a globally unique client ID for this publisher instance.
        unique_client_suffix: str = uuid.uuid4().hex[:8]
        self._mqtt_client_id: str = f"{MQTT_CLIENT_ID_PREFIX}{unique_client_suffix}"

        # Instantiate the Paho MQTT client with the generated ID.
        # Using MQTTv311 (3.1.1) for broad broker compatibility.
        self._mqtt_client: mqtt.Client = mqtt.Client(
            client_id=self._mqtt_client_id,
            protocol=mqtt.MQTTv311,
        )

        # Connection state flag — used to gate publish attempts.
        self._is_connected_to_broker: bool = False

        # Register callback handlers for connection lifecycle events.
        self._mqtt_client.on_connect = self._on_broker_connection_established
        self._mqtt_client.on_disconnect = self._on_broker_connection_lost

        mqtt_publisher_logger.info(
            "MQTT publisher initialized | client_id=%s", self._mqtt_client_id
        )

    def _on_broker_connection_established(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Dict[str, int],
        result_code: int,
    ) -> None:
        """
        Callback invoked when the MQTT client successfully connects.

        Result code 0 indicates successful connection. Non-zero codes
        map to specific failure reasons (auth failure, protocol error, etc.).

        Args:
            client: The MQTT client instance (unused, required by Paho API).
            userdata: User-defined data (unused).
            flags: Response flags from the broker.
            result_code: 0 = success, non-zero = failure reason.
        """
        if result_code == 0:
            self._is_connected_to_broker = True
            mqtt_publisher_logger.info(
                "Connected to MQTT broker | host=%s:%d | client_id=%s",
                MQTT_HOST,
                MQTT_PORT,
                self._mqtt_client_id,
            )
        else:
            self._is_connected_to_broker = False
            mqtt_publisher_logger.error(
                "MQTT connection REJECTED | result_code=%d | host=%s:%d",
                result_code,
                MQTT_HOST,
                MQTT_PORT,
            )

    def _on_broker_connection_lost(
        self, client: mqtt.Client, userdata: Any, result_code: int
    ) -> None:
        """
        Callback invoked when the MQTT connection is unexpectedly lost.

        A non-zero result code indicates an unexpected disconnect
        (network failure, broker restart). A zero result code indicates
        a clean disconnect initiated by the client.

        Args:
            client: The MQTT client instance.
            userdata: User-defined data (unused).
            result_code: 0 = clean disconnect, non-zero = unexpected loss.
        """
        self._is_connected_to_broker = False
        if result_code != 0:
            mqtt_publisher_logger.warning(
                "Unexpected MQTT disconnect | result_code=%d | will attempt reconnection",
                result_code,
            )

    def connect_to_broker(self) -> bool:
        """
        Establish connection to the MQTT broker with retry logic.

        Uses exponential backoff to avoid thundering-herd reconnection
        storms when the broker restarts and all clients reconnect simultaneously.

        Backoff formula: delay = min(base * 2^attempt, max_delay)

        Returns:
            True if connection was established, False if all retries exhausted.
        """
        for reconnect_attempt in range(MQTT_MAX_RECONNECT_ATTEMPTS):
            try:
                mqtt_publisher_logger.info(
                    "Connecting to MQTT broker | attempt=%d/%d | host=%s:%d",
                    reconnect_attempt + 1,
                    MQTT_MAX_RECONNECT_ATTEMPTS,
                    MQTT_HOST,
                    MQTT_PORT,
                )
                self._mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)

                # Start the network loop in a background thread.
                # This handles PINGREQ/PINGRESP, reconnection, and
                # asynchronous message delivery acknowledgments.
                self._mqtt_client.loop_start()

                # Brief wait for the on_connect callback to fire.
                time.sleep(1.0)

                if self._is_connected_to_broker:
                    return True

            except (ConnectionRefusedError, OSError, mqtt.MQTTException) as connection_error:
                mqtt_publisher_logger.error(
                    "MQTT connection attempt FAILED | attempt=%d | error=%s",
                    reconnect_attempt + 1,
                    str(connection_error),
                )

            # Exponential backoff with ceiling.
            backoff_delay_seconds: float = min(
                MQTT_RECONNECT_BASE_DELAY_SECONDS * (2 ** reconnect_attempt),
                MQTT_RECONNECT_MAX_DELAY_SECONDS,
            )
            mqtt_publisher_logger.info(
                "Retrying in %.1f seconds...", backoff_delay_seconds
            )
            time.sleep(backoff_delay_seconds)

        mqtt_publisher_logger.error(
            "MQTT connection FAILED after %d attempts — halting publisher",
            MQTT_MAX_RECONNECT_ATTEMPTS,
        )
        return False

    def publish_meter_reading(self, meter_reading_payload: Dict[str, Any]) -> bool:
        """
        Publish a single meter reading to the broker.

        The reading is JSON-serialized and published to the meter-specific
        topic derived from the meter_id field in the payload. QoS 1
        ensures the broker acknowledges receipt.

        Args:
            meter_reading_payload: Dictionary conforming to the MQTT
                                    payload schema (must contain 'meter_id').

        Returns:
            True if the message was successfully queued for delivery,
            False if the client is disconnected or publish failed.
        """
        if not self._is_connected_to_broker:
            mqtt_publisher_logger.warning(
                "Cannot publish — not connected to broker | meter_id=%s",
                meter_reading_payload.get("meter_id", "UNKNOWN"),
            )
            return False

        # Construct the per-meter topic from the template.
        meter_specific_topic: str = MQTT_TOPIC_TEMPLATE.format(
            meter_id=meter_reading_payload["meter_id"]
        )

        # Serialize the payload to compact JSON (no pretty-printing).
        serialized_payload: str = json.dumps(
            meter_reading_payload, separators=(",", ":")
        )

        try:
            # Publish with QoS 1 (at-least-once) and get the result info.
            publish_result: mqtt.MQTTMessageInfo = self._mqtt_client.publish(
                topic=meter_specific_topic,
                payload=serialized_payload,
                qos=MQTT_QOS,
            )

            # Check if the message was accepted by the client library.
            if publish_result.rc == mqtt.MQTT_ERR_SUCCESS:
                mqtt_publisher_logger.debug(
                    "Published reading | topic=%s | seq=%s",
                    meter_specific_topic,
                    meter_reading_payload.get("seq_no"),
                )
                return True
            else:
                mqtt_publisher_logger.error(
                    "Publish FAILED | topic=%s | rc=%d",
                    meter_specific_topic,
                    publish_result.rc,
                )
                return False

        except (mqtt.MQTTException, ValueError) as publish_error:
            mqtt_publisher_logger.error(
                "Publish EXCEPTION | topic=%s | error=%s",
                meter_specific_topic,
                str(publish_error),
            )
            return False

    def disconnect_from_broker(self) -> None:
        """
        Gracefully disconnect from the MQTT broker.

        Stops the network loop thread and sends a DISCONNECT packet
        to the broker. This ensures the broker marks the client's
        session as cleanly terminated (no LWT message triggered).
        """
        mqtt_publisher_logger.info(
            "Disconnecting from MQTT broker | client_id=%s", self._mqtt_client_id
        )
        self._mqtt_client.loop_stop()
        self._mqtt_client.disconnect()
        self._is_connected_to_broker = False
