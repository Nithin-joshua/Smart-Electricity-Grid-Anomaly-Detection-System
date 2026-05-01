# ==============================================================================
# FILE: utils/logging_setup.py
# PURPOSE: Centralized logging configuration for the smart grid simulation
#          system. Provides structured, consistent log formatting across
#          all modules with appropriate severity routing.
# ==============================================================================

"""
Logging Setup
-------------
Configures a unified logging infrastructure that ensures every module in
the smart grid simulation system emits logs in a consistent, parseable
format. Structured logs are critical for:

1. Correlating events across the simulation, MQTT, and anomaly subsystems.
2. Post-mortem debugging of anomaly injection timing.
3. Integration with log aggregation tools (ELK, Loki) in production.

The format includes ISO-8601 timestamps, module origin, severity level,
and the log message — enabling grep-based filtering and time-series
analysis of operational events.
"""

import logging
import sys
from typing import Optional


def configure_grid_logging(
    log_level: int = logging.INFO,
    log_format: Optional[str] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Initialize and return a configured logger for the smart grid system.

    This function sets up a logger with:
    - A StreamHandler directing output to stdout (container-friendly).
    - A structured format string including timestamp, module, and level.
    - Suppression of duplicate handler attachment on repeated calls.

    Args:
        log_level: The minimum severity level to emit. Defaults to INFO.
                   Use logging.DEBUG for development diagnostics.
        log_format: Optional override for the log line format string.
                    If None, uses the production-standard structured format.
        logger_name: Optional name for the logger. If None, configures the
                     root logger (affects all modules).

    Returns:
        A configured logging.Logger instance ready for use.
    """

    # Use the production-standard format if no custom format is provided.
    # Format fields:
    #   %(asctime)s   — ISO-8601 timestamp for temporal correlation
    #   %(name)s      — Module/logger name for origin tracing
    #   %(levelname)s — Severity level for filtering (INFO, WARNING, ERROR)
    #   %(message)s   — The actual log payload
    effective_log_format: str = log_format or (
        "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s"
    )

    # Retrieve or create the named logger instance.
    # Using a named logger (vs. root) prevents cross-contamination with
    # third-party library logs (e.g., paho-mqtt's internal logging).
    grid_logger: logging.Logger = logging.getLogger(logger_name or "smart_grid")
    grid_logger.setLevel(log_level)

    # Guard against duplicate handler attachment.
    # This occurs when configure_grid_logging() is called multiple times
    # (e.g., during test setUp/tearDown cycles or module re-imports).
    if not grid_logger.handlers:
        # Direct log output to stdout rather than stderr.
        # This is the standard practice for containerized applications
        # because Docker and Kubernetes capture stdout as the primary log stream.
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Apply the structured format to the handler.
        structured_formatter: logging.Formatter = logging.Formatter(
            fmt=effective_log_format,
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        console_handler.setFormatter(structured_formatter)

        grid_logger.addHandler(console_handler)

    return grid_logger


def get_module_logger(module_name: str) -> logging.Logger:
    """
    Retrieve a child logger scoped to a specific module.

    Child loggers inherit the configuration of the parent 'smart_grid'
    logger, ensuring consistent formatting without redundant setup.
    The hierarchical naming (e.g., 'smart_grid.simulator.meter_model')
    enables granular log filtering in production.

    Args:
        module_name: The dot-separated module identifier.
                     Example: "simulator.residential_meter_model"

    Returns:
        A child logger inheriting the smart_grid configuration.
    """
    return logging.getLogger(f"smart_grid.{module_name}")
