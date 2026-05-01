# ==============================================================================
# FILE: simulator/meter_fleet_orchestrator.py
# PURPOSE: Orchestrates the full fleet of simulated smart meters, managing
#          their lifecycle, anomaly engines, and simulation loop execution
#          in both live-stream and fast-forward modes.
# ==============================================================================

"""
Smart Meter Fleet Orchestrator
-------------------------------
Central coordinator for the simulated smart meter fleet. Responsible for:
1. Instantiating N meter nodes with unique household profiles.
2. Binding an anomaly injection engine to each meter.
3. Running the simulation loop in either:
   - LIVE mode:  Real-time MQTT publishing with configurable intervals.
   - FAST-FORWARD mode: Rapid dataset generation to CSV without delays.

The orchestrator decouples simulation logic from transport (MQTT) and
storage (CSV), enabling independent testing of each subsystem.
"""

import csv
import time
import random
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.simulation_config import (
    NUMBER_OF_SIMULATED_METERS,
    PUBLISH_INTERVAL_SECONDS,
    DEFAULT_FASTFORWARD_DAYS,
    DEFAULT_FASTFORWARD_OUTPUT_PATH,
)
from simulator.residential_meter_model import ResidentialEnergyMeterNode
from simulator.anomaly_injection_engine import (
    GridAnomalyInjectionEngine,
    GridAnomalyType,
)
from simulator.mqtt_stream_publisher import MQTTMeterStreamPublisher
from utils.logging_setup import get_module_logger, configure_grid_logging
from utils.time_utils import (
    generate_utc_timestamp,
    compute_simulated_timestamp,
)

# Module-scoped logger for fleet orchestration events.
orchestrator_logger = get_module_logger("meter_fleet_orchestrator")


class SmartMeterFleetOrchestrator:
    """
    Manages the lifecycle and execution of a fleet of simulated smart meters.

    The orchestrator is the top-level coordination layer that:
    - Creates meter instances and their paired anomaly engines
    - Manages the MQTT publisher connection
    - Drives the simulation tick loop
    - Handles graceful shutdown on KeyboardInterrupt
    """

    def __init__(
        self,
        fleet_size: int = NUMBER_OF_SIMULATED_METERS,
        publish_interval_seconds: float = PUBLISH_INTERVAL_SECONDS,
    ) -> None:
        """
        Initialize the fleet with the specified number of meters.

        Args:
            fleet_size: Number of independent smart meters to simulate.
            publish_interval_seconds: Seconds between consecutive readings.
        """
        # Initialize the logging subsystem before any other operations.
        configure_grid_logging()

        self._fleet_size: int = fleet_size
        self._publish_interval_seconds: float = publish_interval_seconds

        # Instantiate the meter fleet — each meter gets a unique index
        # which determines its meter_id (e.g., M_000 through M_049).
        self._meter_fleet: List[ResidentialEnergyMeterNode] = [
            ResidentialEnergyMeterNode(meter_index=idx)
            for idx in range(self._fleet_size)
        ]

        # Bind an anomaly injection engine to each meter.
        # This 1:1 pairing allows independent anomaly scheduling per meter.
        self._anomaly_engines: Dict[str, GridAnomalyInjectionEngine] = {
            meter.meter_id: GridAnomalyInjectionEngine(
                associated_meter_id=meter.meter_id
            )
            for meter in self._meter_fleet
        }

        # MQTT publisher instance — initialized lazily in live mode.
        self._mqtt_publisher: Optional[MQTTMeterStreamPublisher] = None

        orchestrator_logger.info(
            "Fleet orchestrator initialized | fleet_size=%d | interval=%.1fs",
            self._fleet_size,
            self._publish_interval_seconds,
        )

    def inject_anomaly_on_meter(
        self,
        meter_id: str,
        anomaly_type_name: str,
        duration_minutes: float = 30.0,
    ) -> bool:
        """
        Inject an anomaly on a specific meter by its ID.

        Args:
            meter_id: Target meter identifier (e.g., 'M_023').
            anomaly_type_name: String name of the anomaly type
                               (must match GridAnomalyType enum values).
            duration_minutes: How long the anomaly persists.

        Returns:
            True if the anomaly was successfully activated, False if the
            meter_id or anomaly_type_name is invalid.
        """
        if meter_id not in self._anomaly_engines:
            orchestrator_logger.error(
                "Cannot inject anomaly — unknown meter_id=%s", meter_id
            )
            return False

        try:
            anomaly_type = GridAnomalyType(anomaly_type_name)
        except ValueError:
            orchestrator_logger.error(
                "Cannot inject anomaly — unknown type='%s' | valid types: %s",
                anomaly_type_name,
                [t.value for t in GridAnomalyType],
            )
            return False

        activation_timestamp = generate_utc_timestamp()
        self._anomaly_engines[meter_id].activate_anomaly(
            anomaly_type=anomaly_type,
            activation_time=activation_timestamp,
            duration_minutes=duration_minutes,
        )
        return True

    def _generate_fleet_readings_for_tick(
        self, tick_timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """
        Generate one reading from every meter in the fleet for a single tick.

        For each meter:
        1. Generate the raw reading from the residential load model.
        2. Pass the reading through the anomaly engine (may mutate it).
        3. Collect the final payload.

        Args:
            tick_timestamp: The UTC timestamp for this simulation tick.

        Returns:
            List of reading payloads (one per meter) for this tick.
        """
        fleet_readings_batch: List[Dict[str, Any]] = []

        for meter_node in self._meter_fleet:
            # Step 1: Generate the base reading from the diurnal model.
            raw_meter_reading: Dict[str, Any] = meter_node.generate_meter_reading(
                reading_timestamp=tick_timestamp
            )

            # Step 2: Pass through the anomaly engine (no-op if no anomaly active).
            anomaly_engine = self._anomaly_engines[meter_node.meter_id]
            final_reading: Dict[str, Any] = anomaly_engine.apply_active_anomaly(
                reading_payload=raw_meter_reading,
                reading_timestamp=tick_timestamp,
            )

            fleet_readings_batch.append(final_reading)

        return fleet_readings_batch

    def run_live_stream(self) -> None:
        """
        Execute the simulation in LIVE mode with real-time MQTT publishing.

        This mode:
        1. Connects to the MQTT broker.
        2. Enters an infinite loop, generating readings every N seconds.
        3. Publishes each reading to the meter-specific MQTT topic.
        4. Handles KeyboardInterrupt for graceful shutdown.

        The loop runs indefinitely until interrupted (Ctrl+C).
        """
        orchestrator_logger.info(
            "Starting LIVE stream mode | fleet_size=%d | interval=%.1fs",
            self._fleet_size,
            self._publish_interval_seconds,
        )

        # Initialize and connect the MQTT publisher.
        self._mqtt_publisher = MQTTMeterStreamPublisher()
        broker_connected: bool = self._mqtt_publisher.connect_to_broker()

        if not broker_connected:
            orchestrator_logger.error(
                "Failed to connect to MQTT broker — aborting live stream"
            )
            return

        tick_counter: int = 0

        try:
            while True:
                tick_timestamp = generate_utc_timestamp()
                tick_counter += 1

                # Generate readings for the entire fleet.
                fleet_readings = self._generate_fleet_readings_for_tick(tick_timestamp)

                # Publish each reading to its meter-specific topic.
                published_count: int = 0
                for reading_payload in fleet_readings:
                    success = self._mqtt_publisher.publish_meter_reading(reading_payload)
                    if success:
                        published_count += 1

                orchestrator_logger.info(
                    "Tick #%d | published %d/%d readings | timestamp=%s",
                    tick_counter,
                    published_count,
                    len(fleet_readings),
                    tick_timestamp.isoformat(),
                )

                # Sleep for the configured interval before the next tick.
                time.sleep(self._publish_interval_seconds)

        except KeyboardInterrupt:
            orchestrator_logger.info(
                "Live stream interrupted by user (Ctrl+C) — shutting down gracefully"
            )
        finally:
            # Ensure clean MQTT disconnection regardless of exit reason.
            if self._mqtt_publisher is not None:
                self._mqtt_publisher.disconnect_from_broker()
            orchestrator_logger.info(
                "Live stream stopped | total_ticks=%d", tick_counter
            )

    def run_fast_forward_simulation(
        self,
        simulation_days: int = DEFAULT_FASTFORWARD_DAYS,
        output_csv_path: str = DEFAULT_FASTFORWARD_OUTPUT_PATH,
        inject_random_anomalies: bool = False,
    ) -> str:
        """
        Execute the simulation in FAST-FORWARD mode for dataset generation.

        Generates readings for N days of simulated time without real-time
        delays, writing all data to a CSV file. This mode is used to
        create training datasets for the LSTM autoencoder.

        Args:
            simulation_days: Number of days to simulate.
            output_csv_path: File path for the output CSV.
            inject_random_anomalies: If True, randomly inject anomalies
                                      on ~5% of meter-tick combinations.

        Returns:
            The absolute path to the generated CSV file.
        """
        orchestrator_logger.info(
            "Starting FAST-FORWARD simulation | days=%d | meters=%d | output=%s",
            simulation_days,
            self._fleet_size,
            output_csv_path,
        )

        # Calculate the total number of ticks for the simulation window.
        total_seconds_to_simulate: float = simulation_days * 24 * 3600
        total_ticks: int = int(total_seconds_to_simulate / self._publish_interval_seconds)

        # Define the simulation start time (midnight UTC today minus N days).
        simulation_start_utc: datetime = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=simulation_days)

        # Ensure the output directory exists.
        output_directory = os.path.dirname(output_csv_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        # CSV column headers matching the MQTT payload schema plus anomaly tags.
        csv_field_names: List[str] = [
            "meter_id", "timestamp", "kwh", "voltage", "current",
            "seq_no", "anomaly_injected", "anomaly_type",
        ]

        total_readings_written: int = 0

        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_output_file:
            csv_writer = csv.DictWriter(csv_output_file, fieldnames=csv_field_names)
            csv_writer.writeheader()

            for tick_index in range(total_ticks):
                # Compute the simulated timestamp for this tick.
                tick_timestamp = compute_simulated_timestamp(
                    simulation_start_time=simulation_start_utc,
                    elapsed_ticks=tick_index,
                    tick_interval_seconds=self._publish_interval_seconds,
                )

                # Optionally inject random anomalies (~5% probability per meter per tick).
                if inject_random_anomalies and random.random() < 0.001:
                    # Pick a random meter and anomaly type.
                    random_meter = random.choice(self._meter_fleet)
                    random_anomaly = random.choice(list(GridAnomalyType))
                    self._anomaly_engines[random_meter.meter_id].activate_anomaly(
                        anomaly_type=random_anomaly,
                        activation_time=tick_timestamp,
                        duration_minutes=random.uniform(5.0, 60.0),
                    )

                # Generate fleet readings for this tick.
                fleet_readings = self._generate_fleet_readings_for_tick(tick_timestamp)

                # Write each reading to the CSV.
                for reading in fleet_readings:
                    csv_writer.writerow(reading)
                    total_readings_written += 1

                # Progress logging every 10% of total ticks.
                if tick_index > 0 and tick_index % (total_ticks // 10) == 0:
                    progress_percent = (tick_index / total_ticks) * 100
                    orchestrator_logger.info(
                        "Fast-forward progress: %.0f%% | tick %d/%d | readings=%d",
                        progress_percent,
                        tick_index,
                        total_ticks,
                        total_readings_written,
                    )

        orchestrator_logger.info(
            "Fast-forward simulation COMPLETE | total_readings=%d | output=%s",
            total_readings_written,
            output_csv_path,
        )

        return os.path.abspath(output_csv_path)
