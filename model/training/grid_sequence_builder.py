# ==============================================================================
# FILE: model/training/grid_sequence_builder.py
# PURPOSE: Orchestrates the full data preprocessing pipeline: loading,
#          normalization, windowing, and train/validation splitting.
# ==============================================================================

"""
Grid Sequence Builder
-----------------------
End-to-end data preprocessing orchestrator for the LSTM autoencoder.

Pipeline: Load Parquet → Extract features → Normalize → Window → Split
"""

import os
import sys
from typing import Tuple, Optional, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.training_config import (
    NORMAL_DATA_OUTPUT_DIR,
    TRAINING_FEATURE_COLUMNS,
    SLIDING_WINDOW_SIZE,
    SLIDING_WINDOW_STRIDE,
    VALIDATION_SPLIT_RATIO,
    SCALER_PARAMS_JSON_PATH,
)
from model.training.dataset_loader import iter_meter_dataframes
from model.training.ml_utils.normalization import FeatureNormalizer
from model.training.ml_utils.windowing_engine import create_sliding_windows
from utils.logging_setup import get_module_logger

sequence_builder_logger = get_module_logger("grid_sequence_builder")


def build_training_sequences(
    data_directory: str = NORMAL_DATA_OUTPUT_DIR,
    window_size: int = SLIDING_WINDOW_SIZE,
    window_stride: int = SLIDING_WINDOW_STRIDE,
    validation_ratio: float = VALIDATION_SPLIT_RATIO,
    scaler_output_path: str = SCALER_PARAMS_JSON_PATH,
    max_meters: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, FeatureNormalizer]:
    """
    Build train/validation sequence tensors from raw Parquet data.

    Stages:
    1. Load all meter features (per-meter for memory efficiency)
    2. Concatenate, shuffle, and split into train/val
    3. Fit normalizer on training ONLY (prevents data leakage)
    4. Normalize both sets and extract sliding windows
    5. Save scaler params for inference deployment

    Args:
        data_directory: Path to per-meter Parquet files.
        window_size: Timesteps per sliding window.
        window_stride: Step between consecutive windows.
        validation_ratio: Fraction reserved for validation.
        scaler_output_path: Path to save normalization params.
        max_meters: Optional meter limit for debugging.

    Returns:
        (train_sequences, val_sequences, feature_normalizer)
    """
    sequence_builder_logger.info(
        "Starting sequence building | data_dir=%s | window=%d | stride=%d",
        data_directory, window_size, window_stride,
    )

    # STAGE 1: Load per-meter feature arrays
    all_meter_features: List[np.ndarray] = []
    meters_loaded: int = 0

    for meter_df in iter_meter_dataframes(data_directory, max_meters):
        meter_features = meter_df[TRAINING_FEATURE_COLUMNS].values.astype(np.float64)
        all_meter_features.append(meter_features)
        meters_loaded += 1

    if meters_loaded == 0:
        raise ValueError(f"No meter data loaded from {data_directory}.")

    sequence_builder_logger.info(
        "Stage 1 complete | meters=%d | total_rows=%d",
        meters_loaded, sum(len(mf) for mf in all_meter_features),
    )

    # STAGE 2: Concatenate and split (BEFORE normalization to prevent leakage)
    combined_features = np.vstack(all_meter_features)
    total_timesteps = combined_features.shape[0]
    val_size = int(total_timesteps * validation_ratio)
    train_size = total_timesteps - val_size

    # Shuffle to break temporal autocorrelation between train/val
    shuffle_indices = np.random.permutation(total_timesteps)
    combined_features = combined_features[shuffle_indices]

    train_features_raw = combined_features[:train_size]
    val_features_raw = combined_features[train_size:]

    sequence_builder_logger.info(
        "Stage 2 complete | train_rows=%d | val_rows=%d", train_size, val_size,
    )

    # STAGE 3: Fit normalizer on training data ONLY
    feature_normalizer = FeatureNormalizer(feature_names=TRAINING_FEATURE_COLUMNS)
    feature_normalizer.fit_on_training_data(train_features_raw)

    train_normalized = feature_normalizer.transform(train_features_raw)
    val_normalized = feature_normalizer.transform(val_features_raw)

    feature_normalizer.save_scaler_parameters(scaler_output_path)
    sequence_builder_logger.info("Stage 3 complete | scaler saved to %s", scaler_output_path)

    # STAGE 4: Extract sliding windows
    train_sequences = create_sliding_windows(train_normalized, window_size, window_stride)
    val_sequences = create_sliding_windows(val_normalized, window_size, window_stride)

    sequence_builder_logger.info(
        "COMPLETE | train=%s (%.1f MB) | val=%s (%.1f MB)",
        train_sequences.shape, train_sequences.nbytes / 1e6,
        val_sequences.shape, val_sequences.nbytes / 1e6,
    )

    return train_sequences, val_sequences, feature_normalizer
