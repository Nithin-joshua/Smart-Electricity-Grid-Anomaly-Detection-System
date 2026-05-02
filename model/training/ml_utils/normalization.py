# ==============================================================================
# FILE: model/training/utils/normalization.py
# PURPOSE: Feature normalization for the LSTM autoencoder training pipeline.
#          Implements z-score (StandardScaler) normalization with serializable
#          parameters for inference-time reconstruction.
# ==============================================================================

"""
Feature Normalization Engine
------------------------------
Wraps scikit-learn's StandardScaler with domain-specific methods for the
smart grid anomaly detection pipeline.

Why z-score (not min-max)?
    - LSTM gradients are sensitive to input scale. Z-score centers data at 0
      with unit variance, which aligns with the LSTM's tanh activation range.
    - Min-max normalization is brittle: a single outlier during training can
      compress the entire feature range, reducing signal-to-noise ratio.
    - Z-score is invertible and its parameters (mean, std) are stable across
      reasonably-sized training sets.

The scaler parameters (mean and std per feature) are serialized to JSON
so the edge inference engine can apply identical normalization at runtime
without requiring scikit-learn as a dependency.
"""

import json
import os
from typing import Dict, Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from utils.logging_setup import get_module_logger

# Module-scoped logger for normalization operations.
normalization_logger = get_module_logger("normalization")


class FeatureNormalizer:
    """
    Z-score normalization engine for smart grid feature vectors.

    Normalizes each feature independently to zero mean and unit variance:
        x_normalized = (x - mean) / std

    The scaler is fit on training data only (never on validation/test)
    to prevent data leakage. Parameters are then applied identically
    to validation data and serialized for inference-time use.

    Attributes:
        _scaler: The underlying StandardScaler instance.
        _is_fitted: Whether the scaler has been fit to training data.
        _feature_names: Ordered list of feature column names.
    """

    def __init__(self, feature_names: list) -> None:
        """
        Initialize the normalizer with the expected feature ordering.

        Args:
            feature_names: Ordered list of feature column names.
                           Example: ["kwh", "voltage", "current"]
        """
        self._scaler: StandardScaler = StandardScaler()
        self._is_fitted: bool = False
        self._feature_names: list = feature_names

        normalization_logger.info(
            "FeatureNormalizer initialized | features=%s", feature_names
        )

    def fit_on_training_data(self, training_features: np.ndarray) -> None:
        """
        Compute normalization parameters (mean, std) from training data.

        IMPORTANT: This must be called ONLY on training data, never on
        validation or test data. Fitting on the full dataset would leak
        information about the validation distribution into the model.

        Args:
            training_features: 2D array of shape (n_samples, n_features).
                               Each column corresponds to a feature in
                               self._feature_names order.

        Raises:
            ValueError: If input dimensions don't match expected feature count.
        """
        expected_feature_count: int = len(self._feature_names)

        if training_features.ndim != 2:
            raise ValueError(
                f"Expected 2D input array, got {training_features.ndim}D. "
                f"Shape: {training_features.shape}"
            )

        if training_features.shape[1] != expected_feature_count:
            raise ValueError(
                f"Feature count mismatch: expected {expected_feature_count} "
                f"features ({self._feature_names}), got {training_features.shape[1]}"
            )

        self._scaler.fit(training_features)
        self._is_fitted = True

        # Log the computed parameters for debugging and reproducibility.
        normalization_logger.info(
            "Scaler fitted | samples=%d | means=%s | stds=%s",
            training_features.shape[0],
            np.round(self._scaler.mean_, 4).tolist(),
            np.round(self._scaler.scale_, 4).tolist(),
        )

    def transform(self, feature_array: np.ndarray) -> np.ndarray:
        """
        Apply the fitted normalization to a feature array.

        Uses the mean and std computed during fit_on_training_data().
        Can be applied to training, validation, or real-time data.

        Args:
            feature_array: 2D array of shape (n_samples, n_features).

        Returns:
            Normalized array of the same shape, with each feature
            transformed to zero mean and unit variance.

        Raises:
            RuntimeError: If called before fit_on_training_data().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureNormalizer has not been fitted. "
                "Call fit_on_training_data() first."
            )

        return self._scaler.transform(feature_array)

    def inverse_transform(self, normalized_array: np.ndarray) -> np.ndarray:
        """
        Reverse the normalization to recover original-scale values.

        Used to convert model reconstruction outputs back to physical
        units (kWh, Volts, Amps) for interpretable error analysis.

        Args:
            normalized_array: 2D array in normalized space.

        Returns:
            Array in original feature space.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureNormalizer has not been fitted. "
                "Call fit_on_training_data() first."
            )

        return self._scaler.inverse_transform(normalized_array)

    def get_scaler_parameters(self) -> Dict[str, Any]:
        """
        Extract serializable scaler parameters for inference deployment.

        Returns a dictionary mapping feature names to their normalization
        parameters (mean and std). This is consumed by the edge inference
        engine to normalize real-time data without scikit-learn.

        Returns:
            Dictionary with structure:
            {
                "feature_names": ["kwh", "voltage", "current"],
                "means": {"kwh": 3.45, "voltage": 230.0, "current": 15.2},
                "stds": {"kwh": 1.12, "voltage": 3.05, "current": 4.87}
            }

        Raises:
            RuntimeError: If called before fitting.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Cannot extract parameters from unfitted scaler."
            )

        means: Dict[str, float] = {
            name: float(mean_val)
            for name, mean_val in zip(self._feature_names, self._scaler.mean_)
        }
        stds: Dict[str, float] = {
            name: float(std_val)
            for name, std_val in zip(self._feature_names, self._scaler.scale_)
        }

        return {
            "feature_names": self._feature_names,
            "means": means,
            "stds": stds,
        }

    def save_scaler_parameters(self, output_json_path: str) -> None:
        """
        Persist scaler parameters to a JSON file for inference deployment.

        The edge inference engine loads these parameters at startup to
        apply identical normalization to real-time meter readings.

        Args:
            output_json_path: File path to write the JSON parameters.
        """
        scaler_params: Dict[str, Any] = self.get_scaler_parameters()

        # Ensure the output directory exists.
        output_dir: str = os.path.dirname(output_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(scaler_params, json_file, indent=2)

        normalization_logger.info(
            "Scaler parameters saved to %s", output_json_path
        )

    @staticmethod
    def load_scaler_parameters(json_path: str) -> Dict[str, Any]:
        """
        Load previously saved scaler parameters from a JSON file.

        Used by the edge inference engine or for resuming training
        with consistent normalization.

        Args:
            json_path: Path to the scaler parameters JSON file.

        Returns:
            Dictionary with feature_names, means, and stds.

        Raises:
            FileNotFoundError: If the JSON file doesn't exist.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Scaler parameters file not found: {json_path}"
            )

        with open(json_path, "r", encoding="utf-8") as json_file:
            scaler_params: Dict[str, Any] = json.load(json_file)

        normalization_logger.info(
            "Scaler parameters loaded from %s | features=%s",
            json_path,
            scaler_params.get("feature_names", "unknown"),
        )

        return scaler_params
