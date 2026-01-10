# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Conformalized Quantile Regression

This module implements conformal prediction for obtaining quantile estimates
from standard regression models (e.g., XGBoost with TreeLite).

The approach:
1. Train a model to predict the mean (standard regression)
2. Calibrate with a held-out set to learn the residual distribution
3. At prediction time, add the appropriate quantile of residuals to get quantile prediction

This allows using TreeLite for fast inference while still getting valid quantile predictions.

References:
- "Conformalized Quantile Regression" (Romano et al., 2019)
- "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
"""

import numpy as np
import logging
from typing import Optional, Tuple
from collections import deque


class ConformalQuantilePredictor:
    """
    Wraps a standard regression model to provide quantile predictions
    using conformal prediction.

    Usage:
        # Training
        model = train_xgboost_model(X_train, y_train)  # Standard regression
        cqp = ConformalQuantilePredictor(quantile=0.9)
        cqp.calibrate(model, X_calibration, y_calibration)

        # Prediction
        mean_pred = model.predict(X_new)
        quantile_pred = cqp.conformalize(mean_pred)
    """

    def __init__(self, quantile: float = 0.9, max_calibration_samples: int = 5000):
        """
        Args:
            quantile: Target quantile to predict (e.g., 0.9 for P90)
            max_calibration_samples: Maximum calibration samples to keep
        """
        self.quantile = quantile
        self.max_calibration_samples = max_calibration_samples

        # Store calibration residuals (absolute errors)
        self.calibration_residuals = deque(maxlen=max_calibration_samples)

        # Cached quantile value (updated when calibration changes)
        self._cached_quantile_value: Optional[float] = None
        self._cache_dirty = True

        logging.info(f"Initialized ConformalQuantilePredictor for {quantile:.0%} quantile")

    def calibrate(self, predictions: np.ndarray, actuals: np.ndarray):
        """
        Calibrate the conformal predictor using a calibration set.

        Args:
            predictions: Model predictions on calibration set (mean predictions)
            actuals: Actual values for calibration set
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        # Compute absolute residuals
        residuals = np.abs(actuals - predictions)

        # Add to calibration set
        for r in residuals:
            self.calibration_residuals.append(float(r))

        self._cache_dirty = True

        logging.info(
            f"Calibrated with {len(residuals)} samples. "
            f"Total calibration samples: {len(self.calibration_residuals)}"
        )

    def add_online_sample(self, prediction: float, actual: float):
        """
        Add a single sample for online calibration.

        Args:
            prediction: Model prediction (mean)
            actual: Actual observed value
        """
        residual = abs(actual - prediction)
        self.calibration_residuals.append(residual)
        self._cache_dirty = True

    def conformalize(self, mean_prediction: float) -> float:
        """
        Convert a mean prediction to a quantile prediction.

        Args:
            mean_prediction: Model's mean prediction

        Returns:
            Quantile prediction (mean + appropriate residual quantile)
        """
        if len(self.calibration_residuals) == 0:
            logging.warning("No calibration data available. Returning mean prediction.")
            return mean_prediction

        # Update cached quantile if needed
        if self._cache_dirty:
            self._update_quantile_cache()

        # Return mean + quantile of residuals
        return mean_prediction + self._cached_quantile_value

    def conformalize_batch(self, mean_predictions: np.ndarray) -> np.ndarray:
        """
        Convert batch of mean predictions to quantile predictions.

        Args:
            mean_predictions: Array of model's mean predictions

        Returns:
            Array of quantile predictions
        """
        if len(self.calibration_residuals) == 0:
            logging.warning("No calibration data available. Returning mean predictions.")
            return mean_predictions

        # Update cached quantile if needed
        if self._cache_dirty:
            self._update_quantile_cache()

        # Add same quantile to all predictions
        return mean_predictions + self._cached_quantile_value

    def _update_quantile_cache(self):
        """Update the cached quantile value from calibration residuals."""
        if len(self.calibration_residuals) == 0:
            self._cached_quantile_value = 0.0
        else:
            residuals_array = np.array(list(self.calibration_residuals))
            self._cached_quantile_value = float(np.quantile(residuals_array, self.quantile))

        self._cache_dirty = False

        logging.debug(
            f"Updated quantile cache: {self.quantile:.0%} quantile = "
            f"{self._cached_quantile_value:.2f} (from {len(self.calibration_residuals)} samples)"
        )

    def get_coverage_stats(self, predictions: np.ndarray, actuals: np.ndarray) -> dict:
        """
        Evaluate coverage of conformalized predictions.

        Args:
            predictions: Mean predictions
            actuals: Actual values

        Returns:
            Dictionary with coverage statistics
        """
        quantile_preds = self.conformalize_batch(predictions)

        # Coverage: percentage of actuals below quantile prediction
        coverage = np.mean(actuals <= quantile_preds) * 100

        # Violation rate: percentage of actuals above quantile prediction
        violation_rate = np.mean(actuals > quantile_preds) * 100

        # Average width: how much we're adding to mean predictions
        avg_width = np.mean(quantile_preds - predictions)

        return {
            "coverage_percent": coverage,
            "violation_rate_percent": violation_rate,
            "target_coverage_percent": self.quantile * 100,
            "target_violation_percent": (1 - self.quantile) * 100,
            "average_interval_width": avg_width,
            "calibration_samples": len(self.calibration_residuals),
            "quantile_adjustment": self._cached_quantile_value
        }

    def get_state(self) -> dict:
        """Get current state for serialization."""
        return {
            "quantile": self.quantile,
            "calibration_residuals": list(self.calibration_residuals),
            "max_calibration_samples": self.max_calibration_samples
        }

    @classmethod
    def from_state(cls, state: dict) -> 'ConformalQuantilePredictor':
        """Restore from serialized state."""
        cqp = cls(
            quantile=state["quantile"],
            max_calibration_samples=state["max_calibration_samples"]
        )
        cqp.calibration_residuals = deque(
            state["calibration_residuals"],
            maxlen=state["max_calibration_samples"]
        )
        cqp._cache_dirty = True
        return cqp


# Example usage and testing
if __name__ == "__main__":
    # Simulate training data with known quantiles
    np.random.seed(42)

    # Generate synthetic data: y = 2*x + noise
    # where noise has a specific distribution
    n_samples = 1000
    x = np.random.uniform(0, 10, n_samples)

    # True model: y = 2*x + noise (with heavy tail)
    # P90 of noise distribution is at ~1.28 std devs for normal
    noise_std = 5.0
    noise = np.random.normal(0, noise_std, n_samples)
    y_true = 2 * x + noise

    # Simulate a model that predicts the mean perfectly
    y_pred_mean = 2 * x  # Perfect mean predictions

    # Split into calibration and test
    split = 800
    y_pred_cal = y_pred_mean[:split]
    y_true_cal = y_true[:split]
    y_pred_test = y_pred_mean[split:]
    y_true_test = y_true[split:]

    # Create and calibrate conformal predictor for P90
    cqp = ConformalQuantilePredictor(quantile=0.9)
    cqp.calibrate(y_pred_cal, y_true_cal)

    # Get P90 predictions on test set
    y_pred_p90 = cqp.conformalize_batch(y_pred_test)

    # Evaluate coverage
    stats = cqp.get_coverage_stats(y_pred_test, y_true_test)

    print("Conformal Quantile Regression Results:")
    print(f"  Target: P90 (90% coverage)")
    print(f"  Actual coverage: {stats['coverage_percent']:.1f}%")
    print(f"  Violation rate: {stats['violation_rate_percent']:.1f}%")
    print(f"  Quantile adjustment: +{stats['quantile_adjustment']:.2f}")
    print(f"  Calibration samples: {stats['calibration_samples']}")

    # Should be close to 90% coverage
    assert 85 <= stats['coverage_percent'] <= 95, "Coverage should be close to 90%"
    print("\n✓ Conformal prediction working correctly!")
