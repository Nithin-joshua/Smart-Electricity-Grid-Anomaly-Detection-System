# ==============================================================================
# FILE: model/training/lstm_autoencoder_model.py
# PURPOSE: LSTM Autoencoder architecture for smart grid anomaly detection.
#          Learns to reconstruct normal consumption patterns; anomalies
#          produce high reconstruction error at inference time.
# ==============================================================================

"""
LSTM Autoencoder Model
------------------------
Implements the encoder-decoder architecture for unsupervised anomaly detection.

Architecture (strict per spec):
    Encoder: LSTM(64) → LSTM(32) → Dense(16) [latent vector]
    Decoder: RepeatVector(60) → LSTM(32) → LSTM(64) → TimeDistributed(Dense(3))

Why autoencoder for anomaly detection?
    The model is trained ONLY on normal data. It learns to compress and
    reconstruct the typical consumption waveform. At inference, anomalous
    patterns (theft, spikes, voltage sag) produce high reconstruction error
    because they deviate from the learned normal distribution.

Why LSTM (not Transformer)?
    - LSTMs are efficient on small windows (60 timesteps) where attention
      overhead dominates.
    - Deterministic inference on CPU (no attention matrix allocation).
    - TFLite conversion is well-supported for LSTM layers.
"""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.training_config import (
    SLIDING_WINDOW_SIZE,
    FEATURE_COUNT,
    ENCODER_LSTM_UNITS_L1,
    ENCODER_LSTM_UNITS_L2,
    LATENT_SPACE_DIMENSION,
    DECODER_LSTM_UNITS_L1,
    DECODER_LSTM_UNITS_L2,
    ADAM_LEARNING_RATE,
)
from utils.logging_setup import get_module_logger

# Suppress TensorFlow info/warning logs before import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

autoencoder_logger = get_module_logger("lstm_autoencoder_model")


def build_lstm_autoencoder(
    window_size: int = SLIDING_WINDOW_SIZE,
    feature_count: int = FEATURE_COUNT,
    encoder_units_l1: int = ENCODER_LSTM_UNITS_L1,
    encoder_units_l2: int = ENCODER_LSTM_UNITS_L2,
    latent_dim: int = LATENT_SPACE_DIMENSION,
    decoder_units_l1: int = DECODER_LSTM_UNITS_L1,
    decoder_units_l2: int = DECODER_LSTM_UNITS_L2,
    learning_rate: float = ADAM_LEARNING_RATE,
) -> Model:
    """
    Construct and compile the LSTM autoencoder model.

    Architecture:
        Input:  (batch, window_size, feature_count) = (batch, 60, 3)

        Encoder:
            LSTM(64, return_sequences=True)   → (batch, 60, 64)
            LSTM(32, return_sequences=False)  → (batch, 32)
            Dense(16)                         → (batch, 16) [latent vector]

        Decoder:
            RepeatVector(60)                  → (batch, 60, 16)
            LSTM(32, return_sequences=True)   → (batch, 60, 32)
            LSTM(64, return_sequences=True)   → (batch, 60, 64)
            TimeDistributed(Dense(3))         → (batch, 60, 3)

    The model reconstructs its input. During training on normal data,
    it learns the manifold of healthy consumption patterns.

    Args:
        window_size: Timesteps per input window (default: 60).
        feature_count: Features per timestep (default: 3).
        encoder_units_l1: Units in encoder LSTM layer 1 (default: 64).
        encoder_units_l2: Units in encoder LSTM layer 2 (default: 32).
        latent_dim: Dimensionality of the latent space (default: 16).
        decoder_units_l1: Units in decoder LSTM layer 1 (default: 32).
        decoder_units_l2: Units in decoder LSTM layer 2 (default: 64).
        learning_rate: Adam optimizer learning rate (default: 1e-3).

    Returns:
        Compiled Keras Model ready for training.
    """
    autoencoder_logger.info(
        "Building LSTM autoencoder | input=(%d, %d) | "
        "encoder=[%d, %d] | latent=%d | decoder=[%d, %d]",
        window_size, feature_count,
        encoder_units_l1, encoder_units_l2,
        latent_dim,
        decoder_units_l1, decoder_units_l2,
    )

    # --- Input Layer ---
    # Shape: (batch_size, 60, 3) — 5 minutes of [kwh, voltage, current]
    encoder_input = keras.Input(
        shape=(window_size, feature_count),
        name="grid_sequence_input",
    )

    # --- Encoder ---
    # Layer 1: LSTM(64) with return_sequences=True
    # Returns the full output sequence so the next LSTM layer can process
    # the temporal dynamics at each timestep (not just the final state).
    encoder_lstm_1 = layers.LSTM(
        units=encoder_units_l1,
        return_sequences=True,
        name="encoder_lstm_64",
    )(encoder_input)

    # Layer 2: LSTM(32) with return_sequences=False
    # Collapses the temporal dimension into a single fixed-length vector.
    # This is the information bottleneck — only the most salient patterns
    # survive compression to 32 dimensions.
    encoder_lstm_2 = layers.LSTM(
        units=encoder_units_l2,
        return_sequences=False,
        name="encoder_lstm_32",
    )(encoder_lstm_1)

    # Layer 3: Dense(16) — the latent representation
    # Further compresses from 32 → 16 dimensions. This creates a tight
    # bottleneck that forces the model to learn only the most essential
    # features of normal consumption patterns.
    latent_vector = layers.Dense(
        units=latent_dim,
        activation="relu",
        name="latent_vector",
    )(encoder_lstm_2)

    # --- Decoder ---
    # RepeatVector: Broadcasts the latent vector across the time dimension.
    # Shape: (batch, 16) → (batch, 60, 16)
    # This gives each decoder timestep access to the same compressed state.
    decoder_repeat = layers.RepeatVector(
        n=window_size,
        name="repeat_latent",
    )(latent_vector)

    # Layer 4: LSTM(32) — mirrors encoder layer 2
    # Begins temporal reconstruction from the repeated latent vector.
    decoder_lstm_1 = layers.LSTM(
        units=decoder_units_l1,
        return_sequences=True,
        name="decoder_lstm_32",
    )(decoder_repeat)

    # Layer 5: LSTM(64) — mirrors encoder layer 1
    # Refines the temporal reconstruction at higher dimensionality.
    decoder_lstm_2 = layers.LSTM(
        units=decoder_units_l2,
        return_sequences=True,
        name="decoder_lstm_64",
    )(decoder_lstm_1)

    # Layer 6: TimeDistributed(Dense(3)) — final reconstruction
    # Projects each timestep's 64-dim hidden state back to 3 features.
    # TimeDistributed applies the same Dense layer independently at each
    # timestep, maintaining temporal structure in the output.
    decoder_output = layers.TimeDistributed(
        layers.Dense(feature_count),
        name="reconstruction_output",
    )(decoder_lstm_2)

    # --- Assemble Model ---
    autoencoder_model = Model(
        inputs=encoder_input,
        outputs=decoder_output,
        name="grid_lstm_autoencoder",
    )

    # --- Compile ---
    # Loss: MAE (Mean Absolute Error)
    # MAE is more robust to outliers than MSE. Since our training data is
    # "normal only," MAE produces a smoother loss landscape and more
    # interpretable reconstruction error at inference time.
    autoencoder_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
    )

    # Log model summary for verification.
    autoencoder_model.summary(print_fn=autoencoder_logger.info)

    autoencoder_logger.info(
        "Model compiled | loss=MAE | optimizer=Adam(lr=%.1e) | params=%d",
        learning_rate,
        autoencoder_model.count_params(),
    )

    return autoencoder_model


def load_trained_autoencoder(model_path: str) -> Model:
    """
    Load a previously saved autoencoder model from disk.

    Supports both SavedModel format and .keras checkpoint format.

    Args:
        model_path: Path to the saved model directory or .keras file.

    Returns:
        Loaded and compiled Keras Model.

    Raises:
        FileNotFoundError: If the model path doesn't exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Train the model first using training_pipeline.py."
        )

    autoencoder_logger.info("Loading model from %s", model_path)
    loaded_model = keras.models.load_model(model_path)
    autoencoder_logger.info(
        "Model loaded | params=%d | input_shape=%s",
        loaded_model.count_params(),
        loaded_model.input_shape,
    )

    return loaded_model
