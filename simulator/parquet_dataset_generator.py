# ==============================================================================
# FILE: simulator/parquet_dataset_generator.py
# PURPOSE: Fast-forward dataset generation with per-meter Parquet output.
#          Creates a clean, anomaly-free historical dataset for training
#          the LSTM autoencoder in Milestone 2.
# ==============================================================================

"""
Parquet Dataset Generator
--------------------------
Extends the Milestone 1 fast-forward simulation capability to produce
training-grade datasets stored as Apache Parquet files.

Key design decisions:
- **One Parquet file per meter**: Avoids loading 25M+ rows into RAM.
  Each file is ~20MB (518K rows × 6 columns), easily fitting in memory
  for per-meter sliding window extraction.
- **No anomaly injection**: Training data must be purely normal consumption
  patterns so the autoencoder learns the "healthy" distribution only.
- **Columnar Parquet format**: 5-10× compression over CSV, faster reads,
  and native support for typed columns (float64, int64, string).
- **Direct integration with Milestone 1**: Reuses ResidentialEnergyMeterNode
  for data synthesis — identical physics model, just different output format.

Usage:
    python -m simulator.parquet_dataset_generator --meters 50 --days 30
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

import pandas as pd

# Ensure project root is importable regardless of invocation path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.training_config import (
    TRAINING_FLEET_SIZE,
    TRAINING_SIMULATION_DAYS,
    TRAINING_SAMPLING_INTERVAL_SECONDS,
    NORMAL_DATA_OUTPUT_DIR,
)
from simulator.residential_meter_model import ResidentialEnergyMeterNode
from utils.logging_setup import configure_grid_logging, get_module_logger

# Module-scoped logger for dataset generation events.
dataset_gen_logger = get_module_logger("parquet_dataset_generator")


def generate_single_meter_dataset(
    meter_node: ResidentialEnergyMeterNode,
    simulation_start_utc: datetime,
    total_ticks: int,
    tick_interval_seconds: float,
) -> pd.DataFrame:
    """
    Generate the complete time-series dataset for a single meter.

    Synthesizes readings at every tick from simulation_start_utc, covering
    the full simulation window. No anomalies are injected — this produces
    a clean "normal behavior" dataset for unsupervised training.

    Args:
        meter_node: The ResidentialEnergyMeterNode instance to sample.
        simulation_start_utc: UTC datetime marking the simulation epoch.
        total_ticks: Number of 5-second intervals to simulate.
        tick_interval_seconds: Seconds between consecutive readings.

    Returns:
        DataFrame with columns [meter_id, timestamp, kwh, voltage, current, seq_no]
        and total_ticks rows. Timestamp column is datetime64[ns, UTC].
    """
    # Pre-allocate lists for columnar construction — faster than row-by-row append.
    # Each list will hold total_ticks elements (e.g., 518,400 for 30 days).
    meter_id_column: List[str] = []
    timestamp_column: List[datetime] = []
    kwh_column: List[float] = []
    voltage_column: List[float] = []
    current_column: List[float] = []
    seq_no_column: List[int] = []

    for tick_index in range(total_ticks):
        # Compute the simulated timestamp for this tick.
        # timedelta arithmetic is cheaper than repeated datetime.now() calls.
        tick_timestamp: datetime = simulation_start_utc + timedelta(
            seconds=tick_index * tick_interval_seconds
        )

        # Generate a reading using the Milestone 1 residential meter model.
        # This includes the diurnal load profile, voltage fluctuation,
        # and current derivation — all physics-based, no anomalies.
        reading_payload: Dict[str, Any] = meter_node.generate_meter_reading(
            reading_timestamp=tick_timestamp
        )

        # Append to columnar lists (faster than pd.DataFrame.append or concat).
        meter_id_column.append(reading_payload["meter_id"])
        timestamp_column.append(tick_timestamp)
        kwh_column.append(reading_payload["kwh"])
        voltage_column.append(reading_payload["voltage"])
        current_column.append(reading_payload["current"])
        seq_no_column.append(reading_payload["seq_no"])

    # Construct DataFrame from pre-allocated columns in a single operation.
    # This is 10-50× faster than iterative row appends.
    meter_dataframe: pd.DataFrame = pd.DataFrame({
        "meter_id": meter_id_column,
        "timestamp": pd.to_datetime(timestamp_column, utc=True),
        "kwh": kwh_column,
        "voltage": voltage_column,
        "current": current_column,
        "seq_no": seq_no_column,
    })

    return meter_dataframe


def generate_training_dataset(
    fleet_size: int = TRAINING_FLEET_SIZE,
    simulation_days: int = TRAINING_SIMULATION_DAYS,
    sampling_interval_seconds: float = TRAINING_SAMPLING_INTERVAL_SECONDS,
    output_directory: str = NORMAL_DATA_OUTPUT_DIR,
) -> str:
    """
    Generate the full training dataset: one Parquet file per meter.

    Orchestrates the end-to-end generation pipeline:
    1. Create N meter nodes with unique household profiles.
    2. For each meter, simulate `simulation_days` of normal readings.
    3. Write each meter's data as a separate Parquet file.

    The per-meter file strategy ensures:
    - Memory usage stays under 500MB (one meter loaded at a time).
    - Downstream loading can parallelize across meters.
    - Individual meter files can be re-generated without reprocessing all.

    Args:
        fleet_size: Number of independent meters to simulate.
        simulation_days: Days of simulated history to generate.
        sampling_interval_seconds: Seconds between readings.
        output_directory: Directory to write Parquet files into.

    Returns:
        Absolute path to the output directory containing Parquet files.
    """
    # Ensure the output directory exists.
    os.makedirs(output_directory, exist_ok=True)

    # Calculate total ticks for the full simulation window.
    total_seconds: float = simulation_days * 24 * 3600
    total_ticks: int = int(total_seconds / sampling_interval_seconds)

    # Anchor simulation to midnight UTC, N days before now.
    # This produces realistic timestamps that align with UTC day boundaries.
    simulation_start_utc: datetime = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=simulation_days)

    dataset_gen_logger.info(
        "Starting Parquet dataset generation | meters=%d | days=%d | "
        "ticks_per_meter=%d | total_readings=%d | output=%s",
        fleet_size,
        simulation_days,
        total_ticks,
        fleet_size * total_ticks,
        output_directory,
    )

    generation_start_time: float = time.time()
    total_rows_written: int = 0

    for meter_index in range(fleet_size):
        # Create a fresh meter node for each index.
        # Each gets a unique randomized household profile.
        meter_node: ResidentialEnergyMeterNode = ResidentialEnergyMeterNode(
            meter_index=meter_index
        )

        meter_start_time: float = time.time()

        # Generate the full time-series for this meter.
        meter_dataframe: pd.DataFrame = generate_single_meter_dataset(
            meter_node=meter_node,
            simulation_start_utc=simulation_start_utc,
            total_ticks=total_ticks,
            tick_interval_seconds=sampling_interval_seconds,
        )

        # Write to Parquet with snappy compression.
        # Snappy gives ~3× compression with minimal CPU overhead — ideal for
        # large time-series datasets where I/O is the bottleneck, not compute.
        parquet_filename: str = f"meter_{meter_node.meter_id}.parquet"
        parquet_filepath: str = os.path.join(output_directory, parquet_filename)

        meter_dataframe.to_parquet(
            parquet_filepath,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        meter_elapsed: float = time.time() - meter_start_time
        total_rows_written += len(meter_dataframe)

        dataset_gen_logger.info(
            "Meter %s complete | rows=%d | file=%s | time=%.1fs",
            meter_node.meter_id,
            len(meter_dataframe),
            parquet_filename,
            meter_elapsed,
        )

    total_elapsed: float = time.time() - generation_start_time

    dataset_gen_logger.info(
        "Dataset generation COMPLETE | total_rows=%d | total_files=%d | "
        "total_time=%.1fs | output=%s",
        total_rows_written,
        fleet_size,
        total_elapsed,
        os.path.abspath(output_directory),
    )

    return os.path.abspath(output_directory)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct the CLI argument parser for standalone dataset generation.

    Returns:
        Configured ArgumentParser with --meters, --days, and --output flags.
    """
    parser = argparse.ArgumentParser(
        prog="parquet-dataset-generator",
        description=(
            "Generate normal-behavior training datasets for the LSTM autoencoder. "
            "Outputs one Parquet file per meter with no anomaly injection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m simulator.parquet_dataset_generator --meters 50 --days 30\n"
            "  python -m simulator.parquet_dataset_generator --meters 5 --days 1 "
            "--output data/test_data\n"
        ),
    )

    parser.add_argument(
        "--meters",
        type=int,
        default=TRAINING_FLEET_SIZE,
        help=f"Number of meters to simulate (default: {TRAINING_FLEET_SIZE}).",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=TRAINING_SIMULATION_DAYS,
        help=f"Days of simulated history (default: {TRAINING_SIMULATION_DAYS}).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=NORMAL_DATA_OUTPUT_DIR,
        help=f"Output directory for Parquet files (default: {NORMAL_DATA_OUTPUT_DIR}).",
    )

    return parser


def main() -> None:
    """
    CLI entry point for standalone dataset generation.
    """
    configure_grid_logging()

    parser = build_argument_parser()
    args = parser.parse_args()

    dataset_gen_logger.info(
        "CLI invoked | meters=%d | days=%d | output=%s",
        args.meters,
        args.days,
        args.output,
    )

    generate_training_dataset(
        fleet_size=args.meters,
        simulation_days=args.days,
        output_directory=args.output,
    )


if __name__ == "__main__":
    main()
