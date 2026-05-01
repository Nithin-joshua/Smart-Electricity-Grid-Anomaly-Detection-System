# ==============================================================================
# FILE: utils/time_utils.py
# PURPOSE: Timestamp generation and ISO-8601 formatting utilities.
# ==============================================================================

"""
Time Utilities — UTC timestamp ops for the smart grid telemetry pipeline.
All timestamps use UTC to eliminate timezone ambiguity across subsystems.
"""

from datetime import datetime, timezone, timedelta


def generate_utc_timestamp() -> datetime:
    """Generate the current UTC timestamp (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def format_timestamp_iso8601(utc_timestamp: datetime) -> str:
    """Format datetime to ISO-8601 with 'Z' suffix: '2026-03-28T14:32:10Z'."""
    return utc_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_simulated_timestamp(
    simulation_start_time: datetime,
    elapsed_ticks: int,
    tick_interval_seconds: float,
) -> datetime:
    """Compute synthetic timestamp for fast-forward mode based on tick count."""
    elapsed_duration = timedelta(seconds=elapsed_ticks * tick_interval_seconds)
    return simulation_start_time + elapsed_duration


def extract_hour_from_timestamp(utc_timestamp: datetime) -> float:
    """Extract fractional hour-of-day (e.g., 14.5 = 2:30 PM) for load model."""
    return utc_timestamp.hour + utc_timestamp.minute / 60.0 + utc_timestamp.second / 3600.0


def is_weekend_day(utc_timestamp: datetime) -> bool:
    """Return True if timestamp falls on Saturday (5) or Sunday (6)."""
    return utc_timestamp.weekday() >= 5
