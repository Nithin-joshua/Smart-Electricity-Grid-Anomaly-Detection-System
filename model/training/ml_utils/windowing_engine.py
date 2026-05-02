# ==============================================================================
# FILE: model/training/utils/windowing_engine.py
# PURPOSE: Sliding window extraction for time-series sequence modeling.
#          Converts flat normalized arrays into overlapping 3D tensors
#          suitable for LSTM autoencoder input.
# ==============================================================================

"""
Sliding Window Engine
-----------------------
Transforms a 2D normalized feature array of shape (N, F) into a 3D
tensor of overlapping windows with shape (num_windows, window_size, F).

Design decisions:
- **NumPy stride_tricks**: Creates windowed views without copying data.
  For a 500K-row meter, this produces ~500K windows using zero additional
  memory (views share the underlying buffer). The copy is made only at
  the end when converting to a contiguous array for TensorFlow.
- **Configurable stride**: stride=1 maximizes training data density but
  increases dataset size. stride=30 would reduce by 30× for faster
  prototyping experiments.
- **Input validation**: Rejects arrays shorter than window_size to
  prevent silent shape errors that would crash during model.fit().
"""

import numpy as np
from typing import Tuple
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from utils.logging_setup import get_module_logger

# Module-scoped logger for windowing operations.
windowing_logger = get_module_logger("windowing_engine")


def create_sliding_windows(
    normalized_features: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Extract overlapping sliding windows from a 2D feature array.

    Given an input array of shape (N, F) where:
        N = number of timesteps
        F = number of features per timestep

    Produces an output array of shape (num_windows, window_size, F) where:
        num_windows = (N - window_size) // stride + 1

    The windowing operation uses NumPy's stride_tricks for zero-copy
    view creation, making it memory-efficient for large datasets.

    Args:
        normalized_features: 2D array of shape (N, F) containing the
                             normalized time-series. Must be contiguous
                             in memory (C-order).
        window_size: Number of consecutive timesteps per window.
                     Must be > 0 and <= N.
        stride: Step size between consecutive window start positions.
                stride=1 produces maximum overlap.
                stride=window_size produces non-overlapping windows.

    Returns:
        3D array of shape (num_windows, window_size, F) containing
        the extracted windows. This is a contiguous copy (not a view)
        to ensure compatibility with TensorFlow's memory layout requirements.

    Raises:
        ValueError: If input dimensions are invalid or array is too short.
    """
    # --- Input validation ---

    if normalized_features.ndim != 2:
        raise ValueError(
            f"Expected 2D input array (timesteps, features), "
            f"got {normalized_features.ndim}D with shape {normalized_features.shape}"
        )

    num_timesteps: int = normalized_features.shape[0]
    num_features: int = normalized_features.shape[1]

    if window_size <= 0:
        raise ValueError(
            f"window_size must be positive, got {window_size}"
        )

    if window_size > num_timesteps:
        raise ValueError(
            f"Input array has {num_timesteps} timesteps but window_size={window_size}. "
            f"Cannot extract windows longer than the input sequence. "
            f"Either reduce window_size or provide more data."
        )

    if stride <= 0:
        raise ValueError(
            f"stride must be positive, got {stride}"
        )

    # --- Sliding window extraction via stride_tricks ---

    # Calculate the number of windows that fit with the given stride.
    num_windows: int = (num_timesteps - window_size) // stride + 1

    if num_windows <= 0:
        raise ValueError(
            f"No valid windows can be extracted: "
            f"timesteps={num_timesteps}, window_size={window_size}, stride={stride}"
        )

    # Ensure the input is contiguous in memory for stride_tricks.
    # Non-contiguous arrays (e.g., from column slicing) would produce
    # incorrect results with as_strided.
    if not normalized_features.flags["C_CONTIGUOUS"]:
        normalized_features = np.ascontiguousarray(normalized_features)

    # Compute the byte strides for the windowed view.
    # Original strides: (bytes_per_row, bytes_per_element)
    # Windowed strides: (stride * bytes_per_row, bytes_per_row, bytes_per_element)
    #
    # The first dimension steps by `stride` rows between windows.
    # The second dimension steps by 1 row within each window.
    # The third dimension steps by 1 element within each feature vector.
    bytes_per_element: int = normalized_features.strides[1]
    bytes_per_row: int = normalized_features.strides[0]

    windowed_view: np.ndarray = np.lib.stride_tricks.as_strided(
        normalized_features,
        shape=(num_windows, window_size, num_features),
        strides=(stride * bytes_per_row, bytes_per_row, bytes_per_element),
    )

    # Create a contiguous copy of the view.
    # TensorFlow requires contiguous memory layout, and the strided view
    # shares memory with the original array (modifying one affects the other).
    # The copy also prevents memory access violations if the original array
    # is garbage collected while the windows are still in use.
    windowed_sequences: np.ndarray = np.array(windowed_view, dtype=np.float32)

    windowing_logger.info(
        "Sliding windows extracted | input_shape=(%d, %d) | "
        "window_size=%d | stride=%d | output_shape=%s",
        num_timesteps,
        num_features,
        window_size,
        stride,
        windowed_sequences.shape,
    )

    return windowed_sequences


def compute_window_count(
    num_timesteps: int,
    window_size: int,
    stride: int = 1,
) -> int:
    """
    Pre-compute the number of windows without actually creating them.

    Useful for memory planning and progress bar estimation before
    committing to the full windowing operation.

    Args:
        num_timesteps: Total number of timesteps in the input.
        window_size: Number of timesteps per window.
        stride: Step size between windows.

    Returns:
        Number of windows that would be produced.
    """
    if num_timesteps < window_size:
        return 0
    return (num_timesteps - window_size) // stride + 1
