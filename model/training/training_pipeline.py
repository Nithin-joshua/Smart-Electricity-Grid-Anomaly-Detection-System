# ==============================================================================
# FILE: model/training/training_pipeline.py
# PURPOSE: End-to-end training pipeline orchestrator for the LSTM autoencoder.
#          CLI-driven: handles data loading, training, calibration, and export.
# ==============================================================================

"""
Training Pipeline
-------------------
Production CLI for the complete ML training workflow:

    python -m model.training.training_pipeline --data_path data/normal_data --epochs 50

Pipeline stages:
    1. Build training/validation sequences from Parquet data.
    2. Construct and compile the LSTM autoencoder.
    3. Train with EarlyStopping and ModelCheckpoint callbacks.
    4. Generate training loss visualization.
    5. Calibrate anomaly detection threshold on validation data.
    6. Export model as TF SavedModel (for TFLite conversion in Milestone 3).
    7. Save all artifacts (threshold.json, scaler_params.json, plots, logs).
"""

import argparse
import os
import sys
import time
import json
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Suppress TF verbose logs before import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config.training_config import (
    NORMAL_DATA_OUTPUT_DIR,
    SLIDING_WINDOW_SIZE,
    SLIDING_WINDOW_STRIDE,
    FEATURE_COUNT,
    TRAINING_MAX_EPOCHS,
    TRAINING_BATCH_SIZE,
    VALIDATION_SPLIT_RATIO,
    EARLY_STOPPING_PATIENCE,
    ANOMALY_THRESHOLD_SIGMA,
    ARTIFACTS_BASE_DIR,
    SAVED_MODEL_DIR,
    CHECKPOINT_DIR,
    BEST_CHECKPOINT_PATH,
    THRESHOLD_JSON_PATH,
    SCALER_PARAMS_JSON_PATH,
    TRAINING_LOGS_DIR,
    TRAINING_LOG_CSV_PATH,
    PLOTS_DIR,
    LOSS_CURVE_PLOT_PATH,
    ERROR_DISTRIBUTION_PLOT_PATH,
)
from model.training.grid_sequence_builder import build_training_sequences
from model.training.lstm_autoencoder_model import build_lstm_autoencoder
from model.training.threshold_calibrator import run_full_calibration
from utils.logging_setup import configure_grid_logging, get_module_logger

pipeline_logger = get_module_logger("training_pipeline")


def create_artifact_directories() -> None:
    """Create all required output directories for training artifacts."""
    directories = [
        ARTIFACTS_BASE_DIR, SAVED_MODEL_DIR, CHECKPOINT_DIR,
        TRAINING_LOGS_DIR, PLOTS_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        pipeline_logger.debug("Ensured directory: %s", directory)


def build_training_callbacks(
    patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_path: str = BEST_CHECKPOINT_PATH,
    log_csv_path: str = TRAINING_LOG_CSV_PATH,
) -> list:
    """
    Construct Keras training callbacks.

    Callbacks:
    1. EarlyStopping: Halts training when val_loss stops improving.
       Restores the best weights to avoid overfitting on the final epoch.
    2. ModelCheckpoint: Saves the model with lowest val_loss.
       If training is interrupted, the best checkpoint is preserved.
    3. CSVLogger: Writes epoch-level metrics to a CSV file for
       post-hoc analysis and reproducibility auditing.

    Args:
        patience: Epochs to wait for val_loss improvement.
        checkpoint_path: Path for the best model checkpoint.
        log_csv_path: Path for the epoch metrics CSV.

    Returns:
        List of configured Keras Callback instances.
    """
    # Ensure parent directories exist.
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    csv_logger = keras.callbacks.CSVLogger(
        filename=log_csv_path,
        append=False,
    )

    pipeline_logger.info(
        "Callbacks configured | EarlyStopping(patience=%d) | "
        "ModelCheckpoint(%s) | CSVLogger(%s)",
        patience, checkpoint_path, log_csv_path,
    )

    return [early_stopping, model_checkpoint, csv_logger]


def plot_training_loss_curve(
    training_history: keras.callbacks.History,
    output_path: str = LOSS_CURVE_PLOT_PATH,
) -> None:
    """
    Generate and save the training/validation loss curve plot.

    This plot shows:
    - Training loss convergence across epochs.
    - Validation loss for overfitting detection.
    - The epoch where EarlyStopping triggered (if applicable).

    Args:
        training_history: The History object returned by model.fit().
        output_path: File path for the PNG plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs_range = range(1, len(training_history.history["loss"]) + 1)

    ax.plot(
        epochs_range,
        training_history.history["loss"],
        color="#2196F3",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Training Loss",
    )

    ax.plot(
        epochs_range,
        training_history.history["val_loss"],
        color="#F44336",
        linewidth=2,
        marker="s",
        markersize=4,
        label="Validation Loss",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MAE)", fontsize=12)
    ax.set_title(
        "LSTM Autoencoder Training — Loss Curve",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate the best epoch.
    best_epoch = int(np.argmin(training_history.history["val_loss"])) + 1
    best_val_loss = min(training_history.history["val_loss"])
    ax.axvline(x=best_epoch, color="#4CAF50", linestyle="--", alpha=0.7)
    ax.annotate(
        f"Best: epoch {best_epoch}\nval_loss = {best_val_loss:.6f}",
        xy=(best_epoch, best_val_loss),
        xytext=(best_epoch + 1, best_val_loss * 1.1),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#4CAF50"),
        color="#4CAF50",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pipeline_logger.info("Loss curve saved to %s", output_path)


def run_training_pipeline(
    data_path: str = NORMAL_DATA_OUTPUT_DIR,
    epochs: int = TRAINING_MAX_EPOCHS,
    batch_size: int = TRAINING_BATCH_SIZE,
    window_size: int = SLIDING_WINDOW_SIZE,
    window_stride: int = SLIDING_WINDOW_STRIDE,
    patience: int = EARLY_STOPPING_PATIENCE,
    max_meters: Optional[int] = None,
) -> None:
    """
    Execute the complete training pipeline end-to-end.

    Orchestrates all stages from raw data to deployable artifacts.

    Args:
        data_path: Directory containing per-meter Parquet files.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size for gradient updates.
        window_size: Timesteps per sliding window.
        window_stride: Stride between consecutive windows.
        patience: EarlyStopping patience.
        max_meters: Optional meter limit for quick debugging runs.
    """
    pipeline_start_time = time.time()

    pipeline_logger.info("=" * 70)
    pipeline_logger.info("SMART GRID LSTM AUTOENCODER — TRAINING PIPELINE")
    pipeline_logger.info("=" * 70)
    pipeline_logger.info(
        "Config | data=%s | epochs=%d | batch=%d | window=%d | stride=%d",
        data_path, epochs, batch_size, window_size, window_stride,
    )

    # --- Step 0: Create artifact directories ---
    create_artifact_directories()

    # --- Step 1: Build training/validation sequences ---
    pipeline_logger.info("-" * 50)
    pipeline_logger.info("STEP 1: Building training sequences...")
    pipeline_logger.info("-" * 50)

    try:
        train_sequences, val_sequences, feature_normalizer = build_training_sequences(
            data_directory=data_path,
            window_size=window_size,
            window_stride=window_stride,
            scaler_output_path=SCALER_PARAMS_JSON_PATH,
            max_meters=max_meters,
        )
    except (FileNotFoundError, ValueError) as data_error:
        pipeline_logger.error("Data loading failed: %s", str(data_error))
        pipeline_logger.error(
            "Fix: Run dataset generator first — "
            "python -m simulator.parquet_dataset_generator --meters 50 --days 30"
        )
        sys.exit(1)

    pipeline_logger.info(
        "Sequences ready | train=%s | val=%s",
        train_sequences.shape, val_sequences.shape,
    )

    # Validate shapes match the expected model input.
    expected_features = FEATURE_COUNT
    if train_sequences.shape[2] != expected_features:
        pipeline_logger.error(
            "Feature count mismatch: expected %d, got %d. "
            "Check TRAINING_FEATURE_COLUMNS in training_config.py.",
            expected_features, train_sequences.shape[2],
        )
        sys.exit(1)

    # --- Step 2: Build the LSTM autoencoder ---
    pipeline_logger.info("-" * 50)
    pipeline_logger.info("STEP 2: Building LSTM autoencoder model...")
    pipeline_logger.info("-" * 50)

    autoencoder_model = build_lstm_autoencoder(
        window_size=window_size,
        feature_count=expected_features,
    )

    # --- Step 3: Train the model ---
    pipeline_logger.info("-" * 50)
    pipeline_logger.info("STEP 3: Training model...")
    pipeline_logger.info("-" * 50)

    training_callbacks = build_training_callbacks(
        patience=patience,
        checkpoint_path=BEST_CHECKPOINT_PATH,
        log_csv_path=TRAINING_LOG_CSV_PATH,
    )

    # For autoencoder training, input == target.
    # The model learns to reconstruct its own input.
    training_history = autoencoder_model.fit(
        x=train_sequences,
        y=train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_sequences, val_sequences),
        callbacks=training_callbacks,
        verbose=1,
    )

    actual_epochs = len(training_history.history["loss"])
    best_val_loss = min(training_history.history["val_loss"])

    pipeline_logger.info(
        "Training complete | epochs_run=%d/%d | best_val_loss=%.6f",
        actual_epochs, epochs, best_val_loss,
    )

    # --- Step 4: Plot training loss curve ---
    pipeline_logger.info("-" * 50)
    pipeline_logger.info("STEP 4: Generating loss visualization...")
    pipeline_logger.info("-" * 50)

    plot_training_loss_curve(training_history, LOSS_CURVE_PLOT_PATH)

    # --- Step 5: Calibrate anomaly threshold ---
    pipeline_logger.info("-" * 50)
    pipeline_logger.info("STEP 5: Calibrating anomaly threshold...")
    pipeline_logger.info("-" * 50)

    threshold_params, reconstruction_errors = run_full_calibration(
        autoencoder_model=autoencoder_model,
        validation_sequences=val_sequences,
        sigma_multiplier=ANOMALY_THRESHOLD_SIGMA,
        threshold_output_path=THRESHOLD_JSON_PATH,
        plot_output_path=ERROR_DISTRIBUTION_PLOT_PATH,
        batch_size=batch_size,
    )

    # --- Step 6: Export SavedModel ---
    pipeline_logger.info("-" * 50)
    pipeline_logger.info("STEP 6: Exporting TF SavedModel...")
    pipeline_logger.info("-" * 50)

    # Keras 3.x requires model.export() for TF SavedModel format.
    # model.save() now only supports .keras extension.
    # We save BOTH formats:
    #   - .keras for checkpointing and reloading in Python
    #   - SavedModel for TFLite conversion in Milestone 3
    keras_model_path = os.path.join(ARTIFACTS_BASE_DIR, "grid_autoencoder.keras")
    autoencoder_model.save(keras_model_path)
    pipeline_logger.info("Keras model saved to %s", keras_model_path)

    # Export TF SavedModel for TFLite/TFServing deployment.
    autoencoder_model.export(SAVED_MODEL_DIR)
    pipeline_logger.info("TF SavedModel exported to %s", SAVED_MODEL_DIR)

    # Verify the saved Keras model can be reloaded.
    verification_model = keras.models.load_model(keras_model_path)
    pipeline_logger.info(
        "Model verified | input_shape=%s | params=%d",
        verification_model.input_shape, verification_model.count_params(),
    )

    # --- Step 7: Final summary ---
    total_elapsed = time.time() - pipeline_start_time

    pipeline_logger.info("=" * 70)
    pipeline_logger.info("TRAINING PIPELINE COMPLETE")
    pipeline_logger.info("=" * 70)
    pipeline_logger.info("Total time: %.1f seconds (%.1f minutes)",
                        total_elapsed, total_elapsed / 60)
    pipeline_logger.info("Artifacts generated:")
    pipeline_logger.info("  SavedModel:      %s", SAVED_MODEL_DIR)
    pipeline_logger.info("  Best checkpoint: %s", BEST_CHECKPOINT_PATH)
    pipeline_logger.info("  Threshold JSON:  %s", THRESHOLD_JSON_PATH)
    pipeline_logger.info("  Scaler params:   %s", SCALER_PARAMS_JSON_PATH)
    pipeline_logger.info("  Training logs:   %s", TRAINING_LOG_CSV_PATH)
    pipeline_logger.info("  Loss curve plot: %s", LOSS_CURVE_PLOT_PATH)
    pipeline_logger.info("  Error dist plot: %s", ERROR_DISTRIBUTION_PLOT_PATH)
    pipeline_logger.info("Threshold: %.6f (mean=%.6f + %.0f*std=%.6f)",
                        threshold_params["threshold"],
                        threshold_params["mean"],
                        threshold_params["sigma_multiplier"],
                        threshold_params["std"])
    pipeline_logger.info(
        "Model input shape: (1, %d, %d) - ready for TFLite conversion",
        window_size, expected_features,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the training pipeline."""
    parser = argparse.ArgumentParser(
        prog="training-pipeline",
        description=(
            "Smart Grid LSTM Autoencoder Training Pipeline — "
            "Trains on normal consumption data and calibrates anomaly threshold."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m model.training.training_pipeline "
            "--data_path data/normal_data --epochs 50\n"
            "  python -m model.training.training_pipeline "
            "--data_path data/normal_data --epochs 5 --max_meters 3\n"
        ),
    )

    parser.add_argument(
        "--data_path", type=str, default=NORMAL_DATA_OUTPUT_DIR,
        help=f"Path to per-meter Parquet files (default: {NORMAL_DATA_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=TRAINING_MAX_EPOCHS,
        help=f"Maximum training epochs (default: {TRAINING_MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=TRAINING_BATCH_SIZE,
        help=f"Mini-batch size (default: {TRAINING_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--window_size", type=int, default=SLIDING_WINDOW_SIZE,
        help=f"Sliding window size in timesteps (default: {SLIDING_WINDOW_SIZE}).",
    )
    parser.add_argument(
        "--window_stride", type=int, default=SLIDING_WINDOW_STRIDE,
        help=f"Sliding window stride (default: {SLIDING_WINDOW_STRIDE}).",
    )
    parser.add_argument(
        "--patience", type=int, default=EARLY_STOPPING_PATIENCE,
        help=f"EarlyStopping patience in epochs (default: {EARLY_STOPPING_PATIENCE}).",
    )
    parser.add_argument(
        "--max_meters", type=int, default=None,
        help="Limit number of meters to load (for debugging).",
    )

    return parser


def main() -> None:
    """CLI entry point for the training pipeline."""
    configure_grid_logging()

    parser = build_argument_parser()
    args = parser.parse_args()

    pipeline_logger.info(
        "CLI invoked | data=%s | epochs=%d | batch=%d | window=%d",
        args.data_path, args.epochs, args.batch_size, args.window_size,
    )

    run_training_pipeline(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        window_size=args.window_size,
        window_stride=args.window_stride,
        patience=args.patience,
        max_meters=args.max_meters,
    )


if __name__ == "__main__":
    main()
