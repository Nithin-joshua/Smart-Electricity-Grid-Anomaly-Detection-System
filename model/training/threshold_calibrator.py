# ==============================================================================
# FILE: model/training/threshold_calibrator.py
# PURPOSE: Calibrates the anomaly detection threshold by computing
#          reconstruction error statistics on the validation dataset.
# ==============================================================================

"""
Threshold Calibrator
----------------------
After the autoencoder is trained, this module computes the anomaly
detection threshold using the validation dataset.

Algorithm:
    1. Run the trained model on validation sequences.
    2. Compute per-sample MAE: error_i = mean(|input_i - reconstruction_i|)
    3. Compute threshold = mean(errors) + 3 * std(errors)

The 3-sigma threshold assumes reconstruction errors are approximately
normally distributed for normal data. Under this assumption:
    - 99.73% of normal samples fall below the threshold.
    - Only ~0.27% of normal data triggers false positives.

The threshold is saved as JSON for consumption by the edge inference engine.
"""

import json
import os
import sys
from typing import Dict, Any, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Suppress TF logs before import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import Model

from config.training_config import (
    ANOMALY_THRESHOLD_SIGMA,
    THRESHOLD_JSON_PATH,
    ERROR_DISTRIBUTION_PLOT_PATH,
    PLOTS_DIR,
)
from utils.logging_setup import get_module_logger

calibration_logger = get_module_logger("threshold_calibrator")


def compute_reconstruction_errors(
    autoencoder_model: Model,
    validation_sequences: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute per-sample reconstruction errors on the validation set.

    For each input window, the error is the Mean Absolute Error (MAE)
    averaged across all timesteps and features:
        error_i = mean(|input_i - reconstruction_i|)

    This produces a scalar error per sample, suitable for thresholding.

    Args:
        autoencoder_model: The trained LSTM autoencoder.
        validation_sequences: 3D array (n_samples, window_size, n_features).
        batch_size: Batch size for model.predict() to manage memory.

    Returns:
        1D array of per-sample reconstruction errors, shape (n_samples,).
    """
    calibration_logger.info(
        "Computing reconstruction errors | n_samples=%d | batch_size=%d",
        validation_sequences.shape[0], batch_size,
    )

    # Generate reconstructions using the trained autoencoder.
    reconstructed_sequences: np.ndarray = autoencoder_model.predict(
        validation_sequences,
        batch_size=batch_size,
        verbose=0,
    )

    # Compute per-sample MAE: average absolute difference across
    # all timesteps (axis=1) and all features (axis=2).
    # Shape: (n_samples, 60, 3) → (n_samples,)
    per_sample_mae: np.ndarray = np.mean(
        np.abs(validation_sequences - reconstructed_sequences),
        axis=(1, 2),
    )

    calibration_logger.info(
        "Reconstruction errors computed | min=%.6f | max=%.6f | "
        "mean=%.6f | std=%.6f",
        float(np.min(per_sample_mae)),
        float(np.max(per_sample_mae)),
        float(np.mean(per_sample_mae)),
        float(np.std(per_sample_mae)),
    )

    return per_sample_mae


def calibrate_anomaly_threshold(
    reconstruction_errors: np.ndarray,
    sigma_multiplier: float = ANOMALY_THRESHOLD_SIGMA,
) -> Dict[str, float]:
    """
    Compute the anomaly detection threshold from reconstruction errors.

    Formula:
        threshold = mean(errors) + sigma_multiplier * std(errors)

    Under Gaussian assumption with sigma_multiplier=3:
        - 99.73% of normal samples fall below threshold.
        - False positive rate ≈ 0.27%.

    Args:
        reconstruction_errors: 1D array of per-sample MAE values.
        sigma_multiplier: Number of std deviations above mean (default: 3.0).

    Returns:
        Dictionary with keys: mean, std, threshold, sigma_multiplier.
    """
    error_mean: float = float(np.mean(reconstruction_errors))
    error_std: float = float(np.std(reconstruction_errors))
    threshold_value: float = error_mean + sigma_multiplier * error_std

    calibration_logger.info(
        "Threshold calibrated | mean=%.6f | std=%.6f | "
        "sigma=%.1f | threshold=%.6f",
        error_mean, error_std, sigma_multiplier, threshold_value,
    )

    return {
        "mean": error_mean,
        "std": error_std,
        "sigma_multiplier": sigma_multiplier,
        "threshold": threshold_value,
    }


def save_threshold_json(
    threshold_params: Dict[str, float],
    output_path: str = THRESHOLD_JSON_PATH,
) -> None:
    """
    Persist threshold parameters to JSON for edge inference deployment.

    The edge processor loads this file at startup to determine the
    decision boundary for anomaly classification.

    Args:
        threshold_params: Dictionary from calibrate_anomaly_threshold().
        output_path: File path for the JSON output.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(threshold_params, json_file, indent=2)

    calibration_logger.info("Threshold saved to %s", output_path)


def plot_error_distribution(
    reconstruction_errors: np.ndarray,
    threshold_params: Dict[str, float],
    output_path: str = ERROR_DISTRIBUTION_PLOT_PATH,
) -> None:
    """
    Generate a histogram of reconstruction errors with threshold overlay.

    This visualization helps assess:
    - Whether the error distribution is approximately Gaussian.
    - How many normal samples exceed the threshold (false positive preview).
    - The separation between the bulk of the distribution and the threshold.

    Args:
        reconstruction_errors: 1D array of per-sample MAE values.
        threshold_params: Dictionary from calibrate_anomaly_threshold().
        output_path: File path for the PNG plot.
    """
    # Import matplotlib here to avoid startup overhead when not plotting.
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless environments.
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of reconstruction errors.
    ax.hist(
        reconstruction_errors,
        bins=100,
        density=True,
        alpha=0.7,
        color="#2196F3",
        edgecolor="#1565C0",
        label="Reconstruction Error Distribution",
    )

    # Vertical lines for mean and threshold.
    ax.axvline(
        x=threshold_params["mean"],
        color="#4CAF50",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {threshold_params['mean']:.6f}",
    )
    ax.axvline(
        x=threshold_params["threshold"],
        color="#F44336",
        linestyle="-",
        linewidth=2.5,
        label=(
            f"Threshold = {threshold_params['threshold']:.6f} "
            f"(μ + {threshold_params['sigma_multiplier']:.0f}σ)"
        ),
    )

    ax.set_xlabel("Reconstruction Error (MAE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Reconstruction Error Distribution — Anomaly Threshold Calibration",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    calibration_logger.info("Error distribution plot saved to %s", output_path)


def run_full_calibration(
    autoencoder_model: Model,
    validation_sequences: np.ndarray,
    sigma_multiplier: float = ANOMALY_THRESHOLD_SIGMA,
    threshold_output_path: str = THRESHOLD_JSON_PATH,
    plot_output_path: str = ERROR_DISTRIBUTION_PLOT_PATH,
    batch_size: int = 64,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Execute the complete threshold calibration pipeline.

    Orchestrates: predict → compute errors → calibrate → save → plot.

    Args:
        autoencoder_model: Trained LSTM autoencoder.
        validation_sequences: 3D validation data array.
        sigma_multiplier: Sigma multiplier for threshold.
        threshold_output_path: Path for threshold JSON.
        plot_output_path: Path for error distribution plot.
        batch_size: Prediction batch size.

    Returns:
        (threshold_params, reconstruction_errors)
    """
    calibration_logger.info("Starting full calibration pipeline...")

    # Step 1: Compute reconstruction errors.
    reconstruction_errors = compute_reconstruction_errors(
        autoencoder_model, validation_sequences, batch_size,
    )

    # Step 2: Calibrate threshold.
    threshold_params = calibrate_anomaly_threshold(
        reconstruction_errors, sigma_multiplier,
    )

    # Step 3: Save threshold JSON.
    save_threshold_json(threshold_params, threshold_output_path)

    # Step 4: Generate error distribution plot.
    plot_error_distribution(
        reconstruction_errors, threshold_params, plot_output_path,
    )

    calibration_logger.info("Calibration COMPLETE | threshold=%.6f",
                           threshold_params["threshold"])

    return threshold_params, reconstruction_errors
