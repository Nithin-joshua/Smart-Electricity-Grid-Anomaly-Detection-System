# ==============================================================================
# FILE: simulator/anomaly_injection_engine.py
# PURPOSE: Dynamic anomaly injection for testing anomaly detection pipelines.
#          Supports 6 anomaly types that corrupt meter readings in realistic
#          ways, each with configurable duration and activation triggers.
# ==============================================================================

"""
Grid Anomaly Injection Engine
------------------------------
Provides a controlled mechanism to inject realistic grid anomalies into
the simulated meter data stream. Each anomaly type models a real-world
failure or theft scenario observed in distribution network operations.

Supported anomaly types:
- energy_theft_drop:    Revenue protection scenario — tampered meter
- extreme_spike:        Faulty CT clamp or meter firmware bug
- flatline_failure:     Frozen meter (communication/hardware fault)
- sustained_overload:   Illegal load beyond contracted capacity
- nocturnal_spike:      Cannabis farm or crypto mining (night-only)
- voltage_instability:  Transformer tap changer fault or loose neutral

Each anomaly mutates the reading payload in-place and tags it with
metadata for downstream evaluation (ground truth labels for ML training).
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logging_setup import get_module_logger
from utils.time_utils import extract_hour_from_timestamp

# Module-scoped logger for anomaly injection events.
anomaly_logger = get_module_logger("anomaly_injection_engine")


class GridAnomalyType(Enum):
    """Enumeration of supported anomaly injection scenarios."""
    ENERGY_THEFT_DROP = "energy_theft_drop"
    EXTREME_SPIKE = "extreme_spike"
    FLATLINE_FAILURE = "flatline_failure"
    SUSTAINED_OVERLOAD = "sustained_overload"
    NOCTURNAL_SPIKE = "nocturnal_spike"
    VOLTAGE_INSTABILITY = "voltage_instability"


@dataclass
class AnomalyActivationState:
    """
    Tracks the lifecycle of an active anomaly injection.

    Fields:
        anomaly_type: Which anomaly scenario is active.
        activation_time: UTC datetime when the anomaly was triggered.
        duration_minutes: How long the anomaly persists before auto-deactivation.
        flatline_frozen_kwh: Cached kWh value for flatline scenarios
                             (locks the reading to a constant value).
    """
    anomaly_type: GridAnomalyType
    activation_time: datetime
    duration_minutes: float
    flatline_frozen_kwh: Optional[float] = None


class GridAnomalyInjectionEngine:
    """
    Controls anomaly injection lifecycle for a single smart meter.

    Each meter in the fleet gets its own injection engine instance,
    allowing independent anomaly scheduling per meter. The engine
    manages activation, duration tracking, and payload mutation.
    """

    def __init__(self, associated_meter_id: str) -> None:
        """
        Initialize the injection engine for a specific meter.

        Args:
            associated_meter_id: The meter ID this engine is bound to
                                  (e.g., 'M_023'). Used for log correlation.
        """
        self.associated_meter_id: str = associated_meter_id

        # Currently active anomaly state, or None if operating normally.
        self._active_anomaly_state: Optional[AnomalyActivationState] = None

    @property
    def is_anomaly_active(self) -> bool:
        """Check whether an anomaly is currently injected on this meter."""
        return self._active_anomaly_state is not None

    def activate_anomaly(
        self,
        anomaly_type: GridAnomalyType,
        activation_time: datetime,
        duration_minutes: float = 30.0,
    ) -> None:
        """
        Activate an anomaly injection on this meter.

        If an anomaly is already active, it is replaced (last-write-wins).
        This simplifies scheduling logic and avoids complex state machines.

        Args:
            anomaly_type: The anomaly scenario to inject.
            activation_time: UTC datetime marking the start of the anomaly.
            duration_minutes: How long (in minutes) the anomaly persists.
        """
        self._active_anomaly_state = AnomalyActivationState(
            anomaly_type=anomaly_type,
            activation_time=activation_time,
            duration_minutes=duration_minutes,
            flatline_frozen_kwh=None,  # Set lazily on first flatline reading
        )
        anomaly_logger.info(
            "Anomaly ACTIVATED on %s | type=%s | duration=%.1f min",
            self.associated_meter_id,
            anomaly_type.value,
            duration_minutes,
        )

    def deactivate_anomaly(self) -> None:
        """Manually deactivate any active anomaly on this meter."""
        if self._active_anomaly_state is not None:
            anomaly_logger.info(
                "Anomaly DEACTIVATED on %s | type=%s",
                self.associated_meter_id,
                self._active_anomaly_state.anomaly_type.value,
            )
            self._active_anomaly_state = None

    def _check_anomaly_expiration(self, current_timestamp: datetime) -> None:
        """
        Auto-deactivate the anomaly if its duration has elapsed.

        This is called before every reading to ensure anomalies
        don't persist beyond their configured window.
        """
        if self._active_anomaly_state is None:
            return

        elapsed_since_activation = (
            current_timestamp - self._active_anomaly_state.activation_time
        )
        anomaly_duration_limit = timedelta(
            minutes=self._active_anomaly_state.duration_minutes
        )

        if elapsed_since_activation >= anomaly_duration_limit:
            anomaly_logger.info(
                "Anomaly EXPIRED on %s | type=%s | ran for %.1f min",
                self.associated_meter_id,
                self._active_anomaly_state.anomaly_type.value,
                self._active_anomaly_state.duration_minutes,
            )
            self._active_anomaly_state = None

    def apply_active_anomaly(
        self, reading_payload: Dict[str, Any], reading_timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Mutate a meter reading payload if an anomaly is currently active.

        Each anomaly type applies a specific transformation to the reading
        values (kWh, voltage, current) that mimics its real-world signature.
        The payload is also tagged with anomaly metadata for ground-truth
        labeling in ML evaluation datasets.

        Args:
            reading_payload: The original meter reading dictionary.
            reading_timestamp: UTC datetime of this reading (for time checks).

        Returns:
            The (potentially mutated) reading payload with anomaly tags.
        """
        # Check if the active anomaly has expired before applying.
        self._check_anomaly_expiration(reading_timestamp)

        if self._active_anomaly_state is None:
            # No active anomaly — pass through the reading unchanged.
            reading_payload["anomaly_injected"] = False
            reading_payload["anomaly_type"] = None
            return reading_payload

        anomaly_type = self._active_anomaly_state.anomaly_type

        # Dispatch to the appropriate mutation handler.
        if anomaly_type == GridAnomalyType.ENERGY_THEFT_DROP:
            reading_payload = self._apply_energy_theft_drop(reading_payload)

        elif anomaly_type == GridAnomalyType.EXTREME_SPIKE:
            reading_payload = self._apply_extreme_spike(reading_payload)

        elif anomaly_type == GridAnomalyType.FLATLINE_FAILURE:
            reading_payload = self._apply_flatline_failure(reading_payload)

        elif anomaly_type == GridAnomalyType.SUSTAINED_OVERLOAD:
            reading_payload = self._apply_sustained_overload(reading_payload)

        elif anomaly_type == GridAnomalyType.NOCTURNAL_SPIKE:
            reading_payload = self._apply_nocturnal_spike(
                reading_payload, reading_timestamp
            )

        elif anomaly_type == GridAnomalyType.VOLTAGE_INSTABILITY:
            reading_payload = self._apply_voltage_instability(reading_payload)

        # Tag the payload with ground-truth anomaly metadata.
        reading_payload["anomaly_injected"] = True
        reading_payload["anomaly_type"] = anomaly_type.value

        return reading_payload

    # ------------------------------------------------------------------
    # ANOMALY MUTATION HANDLERS
    # ------------------------------------------------------------------

    def _apply_energy_theft_drop(
        self, reading_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate energy theft by reducing reported kWh by ~90%.

        Real-world signature: A tampered meter (magnet, bypass wire, or
        firmware hack) reports drastically lower consumption while actual
        load remains unchanged. Current stays normal but kWh drops.
        """
        original_kwh = reading_payload["kwh"]
        # Reduce consumption to ~10% of actual (theft concealment).
        stolen_kwh = round(original_kwh * 0.10, 2)
        reading_payload["kwh"] = stolen_kwh
        # Recalculate current to match the falsified consumption.
        reading_payload["current"] = round(
            stolen_kwh * 1000.0 / reading_payload["voltage"], 2
        )
        return reading_payload

    def _apply_extreme_spike(
        self, reading_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate an extreme consumption spike (×4.5).

        Real-world signature: Faulty current transformer (CT) clamp
        producing amplified readings, or a meter firmware bug causing
        accumulator overflow and wrap-around to high values.
        """
        spike_multiplier = 4.5
        spiked_kwh = round(reading_payload["kwh"] * spike_multiplier, 2)
        reading_payload["kwh"] = spiked_kwh
        reading_payload["current"] = round(
            spiked_kwh * 1000.0 / reading_payload["voltage"], 2
        )
        return reading_payload

    def _apply_flatline_failure(
        self, reading_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate a frozen/stuck meter reporting constant values.

        Real-world signature: Hardware fault (stuck register), frozen
        communication module, or corrupted firmware causing the meter
        to repeat its last valid reading indefinitely.
        """
        # Capture the first reading's kWh as the "frozen" value.
        if self._active_anomaly_state.flatline_frozen_kwh is None:
            self._active_anomaly_state.flatline_frozen_kwh = reading_payload["kwh"]

        frozen_value = self._active_anomaly_state.flatline_frozen_kwh
        reading_payload["kwh"] = frozen_value
        reading_payload["voltage"] = 230.0  # Perfectly stable (unrealistic)
        reading_payload["current"] = round(
            frozen_value * 1000.0 / 230.0, 2
        )
        return reading_payload

    def _apply_sustained_overload(
        self, reading_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate sustained consumption overload (×2.5).

        Real-world signature: Illegal load exceeding the contracted
        supply capacity — e.g., unauthorized EV charging station,
        industrial equipment on a residential supply, or space heater
        banks during extreme cold weather events.
        """
        overload_multiplier = 2.5
        overloaded_kwh = round(reading_payload["kwh"] * overload_multiplier, 2)
        reading_payload["kwh"] = overloaded_kwh
        reading_payload["current"] = round(
            overloaded_kwh * 1000.0 / reading_payload["voltage"], 2
        )
        return reading_payload

    def _apply_nocturnal_spike(
        self, reading_payload: Dict[str, Any], reading_timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Simulate suspicious nocturnal consumption spikes (1 AM–5 AM only).

        Real-world signature: Cannabis cultivation facility or
        cryptocurrency mining operation that runs high-draw equipment
        only during off-peak hours to avoid detection. Consumption
        appears normal during daytime but spikes dramatically at night.
        """
        current_hour = extract_hour_from_timestamp(reading_timestamp)

        # Only apply the spike during the nocturnal window (1 AM to 5 AM).
        if 1.0 <= current_hour <= 5.0:
            nocturnal_multiplier = 3.5
            spiked_kwh = round(reading_payload["kwh"] * nocturnal_multiplier, 2)
            reading_payload["kwh"] = spiked_kwh
            reading_payload["current"] = round(
                spiked_kwh * 1000.0 / reading_payload["voltage"], 2
            )
        # Outside the window, readings pass through unchanged.

        return reading_payload

    def _apply_voltage_instability(
        self, reading_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate voltage instability with high-magnitude fluctuations.

        Real-world signature: Faulty transformer tap changer, loose
        neutral connection, or upstream grid disturbance causing voltage
        to swing ±15V instead of the normal ±3V range.
        """
        # Inject high-noise voltage (σ=15V vs. normal σ=3V).
        unstable_voltage = round(230.0 + random.gauss(0.0, 15.0), 1)
        reading_payload["voltage"] = unstable_voltage
        # Recalculate current with the distorted voltage.
        reading_payload["current"] = round(
            reading_payload["kwh"] * 1000.0 / max(unstable_voltage, 1.0), 2
        )
        return reading_payload
