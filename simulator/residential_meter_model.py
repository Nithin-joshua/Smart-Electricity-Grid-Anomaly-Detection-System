# ==============================================================================
# FILE: simulator/residential_meter_model.py
# PURPOSE: Simulates a single residential smart electricity meter with
#          realistic diurnal load profiles, voltage variation, and
#          current derivation. Each instance represents one household.
# ==============================================================================

"""
Residential Energy Meter Node
------------------------------
Models a single smart meter installed at a residential premises.
The load profile follows a sinusoidal diurnal curve modulated by:
- Household-specific baseline consumption (randomized per meter)
- Time-of-day effects (morning/evening peaks)
- Weekend occupancy multipliers
- Gaussian noise representing stochastic appliance switching

The mathematical model:
    base_load(t) = baseline_kwh
                   * (1 + 0.5 * sin(2π * (hour - wake_hour) / 24))
                   * weekend_factor
                   + N(0, noise_std)
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any

# Relative imports from project packages
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.simulation_config import (
    BASELINE_KWH_RANGE,
    WAKE_HOUR_RANGE,
    WEEKEND_FACTOR_RANGE,
    NOISE_STD_RANGE,
    NOMINAL_VOLTAGE,
    VOLTAGE_NOISE_STD,
)
from utils.time_utils import (
    format_timestamp_iso8601,
    extract_hour_from_timestamp,
    is_weekend_day,
)


@dataclass
class HouseholdLoadProfile:
    """
    Encapsulates the randomized electrical characteristics of a single
    household. Each meter gets a unique profile to prevent fleet-wide
    correlation artifacts in the simulated data.
    """
    # Average consumption in kWh per reading interval under normal conditions.
    baseline_kwh: float

    # Hour of day (24h) when the household's consumption ramp begins.
    # Shifts the sinusoidal peak to model early vs. late risers.
    wake_hour: int

    # Multiplier applied on weekends to model increased occupancy.
    weekend_consumption_factor: float

    # Standard deviation of the Gaussian noise added to base load.
    # Models stochastic appliance switching (kettles, microwaves, etc.).
    consumption_noise_std: float


class ResidentialEnergyMeterNode:
    """
    Simulates a single residential smart electricity meter.

    Each instance maintains:
    - A unique meter ID (e.g., 'M_023')
    - A randomized household load profile
    - A monotonically increasing sequence number for reading ordering
    - Methods to synthesize realistic kWh, voltage, and current values
    """

    def __init__(self, meter_index: int) -> None:
        """
        Initialize a meter node with a unique ID and randomized profile.

        Args:
            meter_index: Zero-based index used to generate the meter ID.
                         Padded to 3 digits (e.g., index 23 → 'M_023').
        """
        # Unique meter identifier — zero-padded for consistent string sorting
        # and topic routing in MQTT subscriptions.
        self.meter_id: str = f"M_{meter_index:03d}"

        # Monotonically increasing sequence number.
        # Used by downstream consumers to detect gaps (missed readings)
        # and duplicates (QoS 1 redelivery).
        self._reading_sequence_number: int = 0

        # Generate a randomized household profile for this meter instance.
        # Each parameter is independently sampled from its configured range
        # to create diverse but realistic consumption patterns across the fleet.
        self._household_profile: HouseholdLoadProfile = self._generate_random_household_profile()

    def _generate_random_household_profile(self) -> HouseholdLoadProfile:
        """
        Sample a unique household load profile from configured ranges.

        Uses uniform random sampling for each parameter independently.
        This ensures the fleet exhibits realistic heterogeneity — some
        households are low-consumption flats, others are high-draw family homes.

        Returns:
            A HouseholdLoadProfile with randomized parameters.
        """
        return HouseholdLoadProfile(
            baseline_kwh=random.uniform(*BASELINE_KWH_RANGE),
            wake_hour=random.randint(*WAKE_HOUR_RANGE),
            weekend_consumption_factor=random.uniform(*WEEKEND_FACTOR_RANGE),
            consumption_noise_std=random.uniform(*NOISE_STD_RANGE),
        )

    def synthesize_residential_load_profile(self, reading_timestamp: datetime) -> float:
        """
        Compute the instantaneous kWh consumption for this household.

        The load model uses a sinusoidal curve to capture the diurnal
        pattern of residential electricity usage:
        - Trough in early morning (2–5 AM) when occupants sleep
        - Morning peak around wake_hour (breakfast, hot water)
        - Evening peak ~12 hours later (cooking, lighting, entertainment)

        Mathematical formulation:
            load = baseline * (1 + 0.5 * sin(2π * (hour - wake) / 24))
                   * weekend_factor + N(0, σ)

        The sine function is phase-shifted by wake_hour so each household's
        peak aligns with its occupants' actual activity pattern.

        Args:
            reading_timestamp: UTC datetime of the simulated reading.

        Returns:
            Instantaneous consumption in kWh (clamped to non-negative).
        """
        fractional_hour: float = extract_hour_from_timestamp(reading_timestamp)
        profile: HouseholdLoadProfile = self._household_profile

        # Phase-shifted sinusoidal component models the diurnal demand curve.
        # The 0.5 amplitude means consumption varies ±50% around baseline.
        diurnal_phase_radians: float = (
            2.0 * math.pi * (fractional_hour - profile.wake_hour) / 24.0
        )
        diurnal_modulation_factor: float = 1.0 + 0.5 * math.sin(diurnal_phase_radians)

        # Apply weekend multiplier only on Saturday/Sunday.
        # On weekdays, the factor is 1.0 (no adjustment).
        effective_weekend_factor: float = (
            profile.weekend_consumption_factor if is_weekend_day(reading_timestamp) else 1.0
        )

        # Gaussian noise simulates stochastic appliance switching events.
        # Each reading gets independent noise — no temporal correlation.
        appliance_switching_noise: float = random.gauss(0.0, profile.consumption_noise_std)

        # Combine all components into the final consumption value.
        consumption_kwh: float = (
            profile.baseline_kwh
            * diurnal_modulation_factor
            * effective_weekend_factor
            + appliance_switching_noise
        )

        # Clamp to zero — negative consumption is physically impossible
        # for a standard residential meter (no solar/battery export modeled).
        return max(0.0, consumption_kwh)

    def compute_voltage_variation(self) -> float:
        """
        Simulate instantaneous grid voltage with realistic fluctuation.

        Real-world residential voltage varies around the nominal 230V
        due to transformer tap positions, feeder impedance, and aggregate
        neighborhood load. The fluctuation is modeled as Gaussian noise
        with σ=3V, producing a typical range of ~221V to ~239V.

        Returns:
            Simulated voltage in Volts.
        """
        voltage_fluctuation: float = random.gauss(0.0, VOLTAGE_NOISE_STD)
        return round(NOMINAL_VOLTAGE + voltage_fluctuation, 1)

    def derive_current_from_consumption(
        self, consumption_kwh: float, measured_voltage: float
    ) -> float:
        """
        Derive the instantaneous current draw from consumption and voltage.

        Uses the power relationship: I = P / V
        Where P (watts) is derived from kWh assuming the reading interval
        represents instantaneous power (a simplification appropriate for
        5-second sampling intervals where energy ≈ power × dt).

        The factor of 1000 converts kWh → Wh → W (assuming 1-hour basis),
        giving a representative current magnitude for the consumption level.

        Args:
            consumption_kwh: Current consumption reading in kWh.
            measured_voltage: Current voltage reading in Volts.

        Returns:
            Derived current in Amperes, rounded to 2 decimal places.
        """
        # Guard against division by zero (physically impossible but defensive)
        if measured_voltage <= 0:
            return 0.0

        # Convert kWh to Watts (assuming 1-hour basis) then derive current.
        instantaneous_power_watts: float = consumption_kwh * 1000.0
        derived_current_amps: float = instantaneous_power_watts / measured_voltage

        return round(derived_current_amps, 2)

    def generate_meter_reading(self, reading_timestamp: datetime) -> Dict[str, Any]:
        """
        Produce a complete, structured meter reading payload.

        Orchestrates all sub-computations (load, voltage, current) and
        assembles them into the standardized JSON-serializable dictionary
        that matches the MQTT payload specification.

        The sequence number is incremented atomically to ensure monotonic
        ordering for downstream gap detection.

        Args:
            reading_timestamp: UTC datetime for this reading.

        Returns:
            Dictionary matching the MQTT payload schema:
            {
                "meter_id": "M_023",
                "timestamp": "2026-03-28T14:32:10Z",
                "kwh": 3.47,
                "voltage": 229.8,
                "current": 15.12,
                "seq_no": 98432
            }
        """
        # Step 1: Synthesize the consumption value from the diurnal model.
        consumption_kwh_reading: float = self.synthesize_residential_load_profile(
            reading_timestamp
        )

        # Step 2: Simulate grid voltage with realistic fluctuation.
        voltage_reading: float = self.compute_voltage_variation()

        # Step 3: Derive current from the consumption and voltage values.
        current_reading: float = self.derive_current_from_consumption(
            consumption_kwh_reading, voltage_reading
        )

        # Step 4: Increment the sequence counter for ordering guarantees.
        self._reading_sequence_number += 1

        # Step 5: Assemble the payload in the standardized MQTT format.
        meter_reading_payload: Dict[str, Any] = {
            "meter_id": self.meter_id,
            "timestamp": format_timestamp_iso8601(reading_timestamp),
            "kwh": round(consumption_kwh_reading, 2),
            "voltage": voltage_reading,
            "current": current_reading,
            "seq_no": self._reading_sequence_number,
        }

        return meter_reading_payload
