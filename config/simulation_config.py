# ==============================================================================
# FILE: config/simulation_config.py
# PURPOSE: Central configuration for the smart meter simulation layer.
#          All tunable parameters governing meter behavior, fleet sizing,
#          and timing are defined here as typed constants.
# ==============================================================================

"""
Simulation Configuration Constants
-----------------------------------
Defines the operational parameters for the residential smart meter fleet
simulation. Values are calibrated against typical UK/EU residential load
profiles to ensure realistic waveform generation.

Key design decisions:
- Parameter ranges use tuples (min, max) so each meter instance can sample
  a unique household profile, preventing fleet-wide correlation artifacts.
- Weekend factors > 1.0 model the empirical observation that residential
  consumption rises on non-workdays due to occupancy increases.
"""

from typing import Final, Tuple

# ==============================================================================
# FLEET SIZING
# ==============================================================================

# Total number of independent smart meter nodes in the simulated fleet.
# 50 meters provides a statistically meaningful sample while keeping
# per-tick computation under 10ms on commodity hardware.
NUMBER_OF_SIMULATED_METERS: Final[int] = 50

# ==============================================================================
# TIMING
# ==============================================================================

# Interval (in seconds) between consecutive MQTT publications per meter.
# 5 seconds mirrors the reporting cadence of real-world smart meters
# deployed by UK DNOs (Distribution Network Operators) like UKPN and WPD.
PUBLISH_INTERVAL_SECONDS: Final[float] = 5.0

# ==============================================================================
# HOUSEHOLD LOAD PROFILE PARAMETER RANGES
# ==============================================================================

# Baseline consumption range in kWh per reading interval.
# Lower bound (1.5 kWh) represents a single-occupant flat with LED lighting.
# Upper bound (6.0 kWh) represents a family home with electric heating and
# high-draw appliances (oven, dryer, EV charger cycling).
BASELINE_KWH_RANGE: Final[Tuple[float, float]] = (1.5, 6.0)

# Wake hour range — the hour (24h clock) at which the household's
# diurnal consumption curve begins its morning ramp-up.
# Early risers start at 05:00; late risers at 08:00.
# This shifts the sinusoidal peak of the load profile per household.
WAKE_HOUR_RANGE: Final[Tuple[int, int]] = (5, 8)

# Weekend consumption multiplier range.
# Values > 1.0 reflect increased daytime occupancy on weekends.
# 0.8 accounts for households that leave for weekend trips.
# 1.3 accounts for households with full-day cooking, entertainment, etc.
WEEKEND_FACTOR_RANGE: Final[Tuple[float, float]] = (0.8, 1.3)

# Gaussian noise standard deviation range applied to the base load signal.
# This models the stochastic nature of appliance switching events.
# Lower bound: minimal noise (e.g., stable thermostat-controlled loads).
# Upper bound: higher noise (e.g., frequent kettle/microwave cycling).
NOISE_STD_RANGE: Final[Tuple[float, float]] = (0.03, 0.08)

# ==============================================================================
# VOLTAGE PARAMETERS
# ==============================================================================

# Nominal grid voltage (V) — standard for most EU/UK residential supplies.
# The actual delivered voltage fluctuates around this nominal value.
NOMINAL_VOLTAGE: Final[float] = 230.0

# Standard deviation of voltage fluctuations around the nominal value.
# Real-world residential feeders exhibit ~2-4V variation under normal load.
VOLTAGE_NOISE_STD: Final[float] = 3.0

# ==============================================================================
# SIMULATION MODE DEFAULTS
# ==============================================================================

# Default number of simulated days for fast-forward dataset generation.
# 7 days captures a full weekly cycle including weekday/weekend transitions.
DEFAULT_FASTFORWARD_DAYS: Final[int] = 7

# Output filename for fast-forward CSV dataset export.
DEFAULT_FASTFORWARD_OUTPUT_PATH: Final[str] = "data/raw/simulated_readings.csv"
