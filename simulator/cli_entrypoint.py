# ==============================================================================
# FILE: simulator/cli_entrypoint.py
# PURPOSE: Command-line interface for the smart grid simulation system.
#          Provides argparse-based control over simulation mode, fleet size,
#          timing, and anomaly injection from the terminal.
# ==============================================================================

"""
CLI Entrypoint
--------------
Production CLI for launching the smart meter fleet simulation.

Supports two execution modes:
- live:        Real-time MQTT publishing (infinite loop until Ctrl+C)
- fastforward: Rapid CSV dataset generation (completes in seconds/minutes)

Usage examples:
    python -m simulator.cli_entrypoint --mode live
    python -m simulator.cli_entrypoint --mode live --meters 10 --interval 2
    python -m simulator.cli_entrypoint --mode fastforward --days 30
    python -m simulator.cli_entrypoint --mode live --inject-anomaly M_005:extreme_spike:15
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.simulation_config import (
    NUMBER_OF_SIMULATED_METERS,
    PUBLISH_INTERVAL_SECONDS,
    DEFAULT_FASTFORWARD_DAYS,
    DEFAULT_FASTFORWARD_OUTPUT_PATH,
)
from simulator.meter_fleet_orchestrator import SmartMeterFleetOrchestrator
from simulator.anomaly_injection_engine import GridAnomalyType
from utils.logging_setup import configure_grid_logging, get_module_logger

# Module-scoped logger for CLI events.
cli_logger = get_module_logger("cli_entrypoint")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct the argument parser with all supported CLI flags.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="smart-grid-simulator",
        description=(
            "Smart Electricity Grid Meter Simulation System — "
            "Generates realistic residential consumption data via MQTT "
            "or batch CSV export."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m simulator.cli_entrypoint --mode live\n"
            "  python -m simulator.cli_entrypoint --mode live --meters 10 --interval 2\n"
            "  python -m simulator.cli_entrypoint --mode fastforward --days 30\n"
            "  python -m simulator.cli_entrypoint --mode live "
            "--inject-anomaly M_005:extreme_spike:15\n"
        ),
    )

    # --- Required arguments ---

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["live", "fastforward"],
        help=(
            "Simulation execution mode. "
            "'live' publishes to MQTT in real-time. "
            "'fastforward' generates a CSV dataset without delays."
        ),
    )

    # --- Optional arguments ---

    parser.add_argument(
        "--meters",
        type=int,
        default=NUMBER_OF_SIMULATED_METERS,
        help=f"Number of smart meters to simulate (default: {NUMBER_OF_SIMULATED_METERS}).",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=PUBLISH_INTERVAL_SECONDS,
        help=f"Publish interval in seconds for live mode (default: {PUBLISH_INTERVAL_SECONDS}).",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_FASTFORWARD_DAYS,
        help=f"Number of days to simulate in fastforward mode (default: {DEFAULT_FASTFORWARD_DAYS}).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_FASTFORWARD_OUTPUT_PATH,
        help=f"Output CSV path for fastforward mode (default: {DEFAULT_FASTFORWARD_OUTPUT_PATH}).",
    )

    parser.add_argument(
        "--inject-anomaly",
        type=str,
        default=None,
        help=(
            "Inject an anomaly on a specific meter at startup. "
            "Format: METER_ID:ANOMALY_TYPE:DURATION_MINUTES "
            "Example: M_005:extreme_spike:15 "
            f"Valid types: {[t.value for t in GridAnomalyType]}"
        ),
    )

    parser.add_argument(
        "--random-anomalies",
        action="store_true",
        default=False,
        help="Enable random anomaly injection during fastforward simulation.",
    )

    return parser


def parse_anomaly_injection_argument(
    anomaly_spec_string: str,
) -> tuple:
    """
    Parse the --inject-anomaly argument into its components.

    Expected format: METER_ID:ANOMALY_TYPE:DURATION_MINUTES
    Example: M_005:extreme_spike:15

    Args:
        anomaly_spec_string: Raw CLI argument string.

    Returns:
        Tuple of (meter_id, anomaly_type_name, duration_minutes).

    Raises:
        ValueError: If the format is invalid.
    """
    parts = anomaly_spec_string.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid anomaly spec format: '{anomaly_spec_string}'. "
            "Expected format: METER_ID:ANOMALY_TYPE:DURATION_MINUTES"
        )

    target_meter_id: str = parts[0]
    anomaly_type_name: str = parts[1]
    anomaly_duration_minutes: float = float(parts[2])

    return target_meter_id, anomaly_type_name, anomaly_duration_minutes


def main() -> None:
    """
    Main entry point — parses arguments and launches the simulation.
    """
    # Initialize logging before anything else.
    configure_grid_logging()

    # Parse CLI arguments.
    argument_parser = build_argument_parser()
    parsed_args = argument_parser.parse_args()

    cli_logger.info(
        "CLI invoked | mode=%s | meters=%d | interval=%.1fs",
        parsed_args.mode,
        parsed_args.meters,
        parsed_args.interval,
    )

    # Initialize the fleet orchestrator with the specified configuration.
    fleet_orchestrator = SmartMeterFleetOrchestrator(
        fleet_size=parsed_args.meters,
        publish_interval_seconds=parsed_args.interval,
    )

    # Handle anomaly injection if specified.
    if parsed_args.inject_anomaly is not None:
        try:
            meter_id, anomaly_type, duration = parse_anomaly_injection_argument(
                parsed_args.inject_anomaly
            )
            injection_success = fleet_orchestrator.inject_anomaly_on_meter(
                meter_id=meter_id,
                anomaly_type_name=anomaly_type,
                duration_minutes=duration,
            )
            if injection_success:
                cli_logger.info(
                    "Anomaly injected via CLI | meter=%s | type=%s | duration=%.1f min",
                    meter_id,
                    anomaly_type,
                    duration,
                )
            else:
                cli_logger.error("Failed to inject anomaly — check meter ID and type")
        except ValueError as parse_error:
            cli_logger.error("Anomaly spec parse error: %s", str(parse_error))
            sys.exit(1)

    # Dispatch to the appropriate simulation mode.
    if parsed_args.mode == "live":
        cli_logger.info("Launching LIVE stream mode...")
        fleet_orchestrator.run_live_stream()

    elif parsed_args.mode == "fastforward":
        cli_logger.info("Launching FAST-FORWARD simulation mode...")
        output_path = fleet_orchestrator.run_fast_forward_simulation(
            simulation_days=parsed_args.days,
            output_csv_path=parsed_args.output,
            inject_random_anomalies=parsed_args.random_anomalies,
        )
        cli_logger.info("Dataset generated at: %s", output_path)


if __name__ == "__main__":
    main()
