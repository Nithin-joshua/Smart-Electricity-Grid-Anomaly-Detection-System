# ==============================================================================
# FILE: config/training_config.py
# PURPOSE: Centralized configuration for the LSTM autoencoder training pipeline.
#          All hyperparameters, file paths, feature definitions, and model
#          architecture constants are defined here to enable single-point tuning.
# ==============================================================================

"""
Training Configuration Constants
----------------------------------
Defines every tunable parameter for the Milestone 2 ML pipeline:

- Dataset generation (meters, days, output paths)
- Feature selection and normalization strategy
- Sliding window dimensions
- LSTM autoencoder architecture (layer sizes, latent dim)
- Training loop settings (epochs, batch size, early stopping)
- Threshold calibration sigma multiplier
- Output artifact paths

Design rationale:
    Centralizing hyperparameters here (rather than scattering across modules)
    ensures reproducibility and simplifies hyperparameter sweeps. Every module
    imports from this single source of truth.
"""

import os
from typing import Final, Tuple, List

# ==============================================================================
# PROJECT ROOT — computed relative to this file's location
# All artifact paths are anchored to this root for portability.
# ==============================================================================

# Navigate from config/ up one level to the project root.
_CONFIG_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: Final[str] = os.path.abspath(os.path.join(_CONFIG_DIR, ".."))

# ==============================================================================
# DATASET GENERATION SETTINGS
# ==============================================================================

# Number of smart meters to simulate for the training corpus.
# 50 meters provides sufficient diversity in household load profiles
# to train a generalizable autoencoder without overfitting to a single
# consumption pattern.
TRAINING_FLEET_SIZE: Final[int] = 50

# Duration (in days) of simulated history.
# 30 days captures 4 full weekly cycles, ensuring the model learns both
# weekday and weekend consumption patterns, plus month-boundary effects.
TRAINING_SIMULATION_DAYS: Final[int] = 30

# Sampling interval in seconds — must match the simulator's publish interval.
# 5-second granularity provides 17,280 readings/meter/day, which is
# sufficient temporal resolution for detecting sub-minute anomalies.
TRAINING_SAMPLING_INTERVAL_SECONDS: Final[float] = 5.0

# Directory where per-meter Parquet files are stored.
# One file per meter (e.g., meter_M_000.parquet) to enable parallel loading
# and avoid loading the entire 25M+ row dataset into RAM at once.
NORMAL_DATA_OUTPUT_DIR: Final[str] = os.path.join(PROJECT_ROOT, "data", "normal_data")

# ==============================================================================
# FEATURE CONFIGURATION
# ==============================================================================

# Ordered list of numerical features extracted from each meter reading.
# These are the input channels to the LSTM autoencoder.
#   - kwh:     Instantaneous consumption — primary signal for load anomalies
#   - voltage: Grid voltage — detects voltage sag/swell anomalies
#   - current: Derived current — correlated with kwh, provides redundancy
TRAINING_FEATURE_COLUMNS: Final[List[str]] = ["kwh", "voltage", "current"]

# Number of features per timestep (must match len(TRAINING_FEATURE_COLUMNS)).
# Used to define the model's input shape: (window_size, FEATURE_COUNT).
FEATURE_COUNT: Final[int] = len(TRAINING_FEATURE_COLUMNS)

# ==============================================================================
# SLIDING WINDOW CONFIGURATION
# ==============================================================================

# Number of consecutive timesteps in each input window.
# 60 timesteps × 5-second intervals = 5 minutes of context.
# This window captures short-term consumption patterns (appliance cycles,
# heating ramp-ups) while remaining small enough for efficient training.
SLIDING_WINDOW_SIZE: Final[int] = 60

# Step size between consecutive windows.
# stride=1 produces maximum overlap for dense training signal.
# Increase to 10-30 to reduce dataset size if memory is constrained.
SLIDING_WINDOW_STRIDE: Final[int] = 1

# ==============================================================================
# LSTM AUTOENCODER ARCHITECTURE
# ==============================================================================

# Encoder LSTM layer sizes — progressively compress temporal information.
# 64 → 32 captures both fine-grained and coarse temporal patterns.
ENCODER_LSTM_UNITS_L1: Final[int] = 64
ENCODER_LSTM_UNITS_L2: Final[int] = 32

# Latent space dimensionality — the bottleneck representation.
# 16 dimensions compress the 60×3 = 180-dimensional input by 11×,
# forcing the encoder to learn only the most salient normal patterns.
LATENT_SPACE_DIMENSION: Final[int] = 16

# Decoder LSTM layer sizes — mirror the encoder for symmetric reconstruction.
DECODER_LSTM_UNITS_L1: Final[int] = 32
DECODER_LSTM_UNITS_L2: Final[int] = 64

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

# Maximum number of training epochs.
# 50 epochs is a safe upper bound — EarlyStopping typically halts at 15-25.
TRAINING_MAX_EPOCHS: Final[int] = 50

# Mini-batch size for gradient updates.
# 64 balances gradient noise (smaller → noisier) vs. memory usage (larger → more RAM).
# On CPU, 64 keeps per-batch computation under 2 seconds.
TRAINING_BATCH_SIZE: Final[int] = 64

# Fraction of training data reserved for validation.
# 10% provides a meaningful validation signal while preserving 90% for training.
VALIDATION_SPLIT_RATIO: Final[float] = 0.10

# EarlyStopping patience — epochs to wait for val_loss improvement before halting.
# 5 epochs prevents premature stopping from normal loss fluctuations
# while avoiding wasteful computation on plateaued models.
EARLY_STOPPING_PATIENCE: Final[int] = 5

# Learning rate for the Adam optimizer.
# 1e-3 is the Adam default and works well for LSTM autoencoders.
ADAM_LEARNING_RATE: Final[float] = 1e-3

# ==============================================================================
# THRESHOLD CALIBRATION
# ==============================================================================

# Number of standard deviations above mean reconstruction error for anomaly threshold.
# 3σ corresponds to the 99.73% confidence interval under Gaussian assumption,
# meaning only ~0.27% of normal data would trigger a false positive.
ANOMALY_THRESHOLD_SIGMA: Final[float] = 3.0

# ==============================================================================
# OUTPUT ARTIFACT PATHS
# ==============================================================================

# Base directory for all training artifacts.
ARTIFACTS_BASE_DIR: Final[str] = os.path.join(PROJECT_ROOT, "model", "artifacts")

# TensorFlow SavedModel directory — used for TFLite conversion in Milestone 3.
SAVED_MODEL_DIR: Final[str] = os.path.join(ARTIFACTS_BASE_DIR, "saved_model")

# Keras checkpoint directory — stores the best model weights during training.
CHECKPOINT_DIR: Final[str] = os.path.join(ARTIFACTS_BASE_DIR, "checkpoints")
BEST_CHECKPOINT_PATH: Final[str] = os.path.join(CHECKPOINT_DIR, "best_autoencoder.keras")

# Anomaly threshold JSON — consumed by the edge inference engine.
THRESHOLD_JSON_PATH: Final[str] = os.path.join(ARTIFACTS_BASE_DIR, "threshold.json")

# Feature scaler parameters — needed to normalize real-time data at inference.
SCALER_PARAMS_JSON_PATH: Final[str] = os.path.join(ARTIFACTS_BASE_DIR, "scaler_params.json")

# Training log CSV — epoch-level metrics for post-hoc analysis.
TRAINING_LOGS_DIR: Final[str] = os.path.join(ARTIFACTS_BASE_DIR, "training_logs")
TRAINING_LOG_CSV_PATH: Final[str] = os.path.join(TRAINING_LOGS_DIR, "epoch_metrics.csv")

# Visualization plots directory.
PLOTS_DIR: Final[str] = os.path.join(ARTIFACTS_BASE_DIR, "plots")
LOSS_CURVE_PLOT_PATH: Final[str] = os.path.join(PLOTS_DIR, "training_loss_curve.png")
ERROR_DISTRIBUTION_PLOT_PATH: Final[str] = os.path.join(
    PLOTS_DIR, "reconstruction_error_distribution.png"
)
