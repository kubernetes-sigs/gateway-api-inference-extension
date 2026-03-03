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
1. Train a model to predict the mean (standard regression with objective=reg:squarederror)
2. Calibrate with a held-out set to learn the residual distribution
3. At prediction time, add the appropriate quantile of residuals to get quantile prediction

This allows using TreeLite for fast inference while still getting valid quantile predictions.

References:
- "Conformalized Quantile Regression" (Romano et al., 2019)
- "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
"""

import logging
import math
from collections import deque

import numpy as np


class ConformalQuantilePredictor:
    """
    Wraps a standard regression model to provide quantile predictions
    using conformal prediction.

    Usage:
        # Training
        model = train_xgboost_model(X_train, y_train)  # Standard regression
        cqp = ConformalQuantilePredictor(quantile=0.90)
        cqp.calibrate(model, X_calibration, y_calibration)

        # Prediction
        mean_pred = model.predict(X_new)
        quantile_pred = cqp.conformalize(mean_pred)
    """

    def __init__(self, quantile: float = 0.90, max_calibration_samples: int = 5000):
        """
        Args:
            quantile: Target quantile to predict (e.g., 0.90 for P90)
            max_calibration_samples: Maximum calibration samples to keep
        """
        self.quantile = quantile
        self.max_calibration_samples = max_calibration_samples

        # Store calibration residuals (signed: actual - predicted)
        self.calibration_residuals = deque(maxlen=max_calibration_samples)

        # Cached quantile value (updated when calibration changes)
        self._cached_quantile_value: float | None = None
        self._cache_dirty = True

        # Auto-calibrated alpha (set by _auto_calibrate)
        self._calibrated_alpha: float | None = None

        # Track original sample count (used by v3 compact format where residuals are empty)
        self._calibration_samples_count = 0

        # Track whether we've logged "no calibration data" warning (to avoid spam)
        self._logged_no_calibration_warning = False

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Initialized ConformalQuantilePredictor for {quantile:.0%} quantile")

    def calibrate(self, predictions: np.ndarray, actuals: np.ndarray):
        """
        Calibrate the conformal predictor using a calibration set.

        Args:
            predictions: Model predictions on calibration set (mean predictions)
            actuals: Actual values for calibration set
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        # Compute signed residuals (actual - predicted) for one-sided upper bound
        residuals = actuals - predictions

        # Add to calibration set
        for r in residuals:
            self.calibration_residuals.append(float(r))

        self._cache_dirty = True
        self._calibration_samples_count = len(self.calibration_residuals)

        # Reset warning flag now that we have calibration data
        self._logged_no_calibration_warning = False

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                f"Calibrated with {len(residuals)} samples. "
                f"Total calibration samples: {len(self.calibration_residuals)}"
            )

        # Auto-calibrate alpha for accurate coverage
        self._auto_calibrate()

    def _auto_calibrate(self):
        """
        Compute conformal quantile level using the standard order statistic formula.

        For n calibration residuals and target coverage c, the conformal index is:
            k = ceil(c * (n + 1))
        clamped to [1, n]. Setting alpha = k / (n + 1) makes np.quantile()
        select approximately the k-th order statistic, providing a finite-sample
        coverage guarantee >= c on exchangeable future data.

        Reference: "A Gentle Introduction to Conformal Prediction and
        Distribution-Free Uncertainty Quantification" (Angelopoulos & Bates, 2021)
        """
        n = len(self.calibration_residuals)
        if n < 2:
            self._calibrated_alpha = self.quantile
            self._cache_dirty = True
            logging.debug(f"Auto-calibration: using raw quantile {self.quantile} as alpha ({n} samples < 2)")
            return

        k = math.ceil(self.quantile * (n + 1))
        k = max(1, min(k, n))  # clamp to [1, n]

        self._calibrated_alpha = k / (n + 1)
        self._cache_dirty = True

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                f"Auto-calibrated alpha: {self._calibrated_alpha:.4f} "
                f"(conformal index k={k}/{n}, target coverage: {self.quantile:.0%})"
            )

    def add_online_sample(self, prediction: float, actual: float):
        """
        Add a single sample for online calibration.

        Args:
            prediction: Model prediction (mean)
            actual: Actual observed value
        """
        residual = actual - prediction
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
        if len(self.calibration_residuals) == 0 and self._cached_quantile_value is None:
            if not self._logged_no_calibration_warning:
                logging.debug("No calibration data available. Returning mean prediction (bootstrap phase).")
                self._logged_no_calibration_warning = True
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
        if len(self.calibration_residuals) == 0 and self._cached_quantile_value is None:
            if not self._logged_no_calibration_warning:
                logging.debug("No calibration data available. Returning mean predictions (bootstrap phase).")
                self._logged_no_calibration_warning = True
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
            effective_alpha = self._calibrated_alpha if self._calibrated_alpha is not None else self.quantile
            self._cached_quantile_value = float(np.quantile(residuals_array, effective_alpha))

        self._cache_dirty = False

        if logging.getLogger().isEnabledFor(logging.DEBUG):
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
            "quantile_adjustment": self._cached_quantile_value,
            "calibrated_alpha": self._calibrated_alpha,
        }

    @property
    def calibration_sample_count(self) -> int:
        """Number of calibration samples (works for both full and compact formats)."""
        return len(self.calibration_residuals) or self._calibration_samples_count

    def get_state(self, compact: bool = False) -> dict:
        """Get current state for serialization.

        Args:
            compact: If True, save only precomputed offset (v3 format).
                     Suitable for inference-only bundles where residuals
                     aren't needed. ~200 bytes vs ~50KB.
        """
        if compact:
            # Ensure cache is fresh before saving
            if self._cache_dirty:
                self._update_quantile_cache()
            return {
                "version": 3,
                "quantile": self.quantile,
                "max_calibration_samples": self.max_calibration_samples,
                "calibrated_alpha": self._calibrated_alpha,
                "cached_quantile_value": self._cached_quantile_value,
                "calibration_samples": len(self.calibration_residuals),
            }
        return {
            "version": 2,
            "quantile": self.quantile,
            "calibration_residuals": list(self.calibration_residuals),
            "max_calibration_samples": self.max_calibration_samples,
            "calibrated_alpha": self._calibrated_alpha,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ConformalQuantilePredictor":
        """Restore from serialized state."""
        version = state.get("version", 1)

        if version == 3:
            # Compact format: no residuals, just precomputed offset
            cqp = cls(quantile=state["quantile"], max_calibration_samples=state["max_calibration_samples"])
            cqp._calibrated_alpha = state.get("calibrated_alpha")
            cqp._cached_quantile_value = state.get("cached_quantile_value", 0.0)
            cqp._cache_dirty = False  # Cache is already set
            cqp._calibration_samples_count = state.get("calibration_samples", 0)
            # calibration_residuals stays empty — conformalize() will use cached value
            return cqp

        cqp = cls(quantile=state["quantile"], max_calibration_samples=state["max_calibration_samples"])
        cqp.calibration_residuals = deque(state["calibration_residuals"], maxlen=state["max_calibration_samples"])
        cqp._cache_dirty = True
        cqp._calibration_samples_count = len(cqp.calibration_residuals)

        if version == 1:
            # v1 used absolute residuals — keep them but leave _calibrated_alpha=None
            # so we fall back to self.quantile. Conservative (over-coverage) until
            # next training cycle produces v2 with signed residuals.
            logging.warning(
                "Loading v1 conformal state (absolute residuals). "
                "Coverage may be conservative until next training cycle."
            )
        else:
            cqp._calibrated_alpha = state.get("calibrated_alpha")

        return cqp


# Example usage and testing
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Simulate training data with known quantiles
    np.random.seed(42)

    # Generate synthetic data: y = 2*x + noise
    # where noise has a specific distribution
    n_samples = 1000
    x = np.random.uniform(0, 10, n_samples)

    # True model: y = 2*x + noise (with heavy tail)
    # P85 of noise distribution is at ~1.04 std devs for normal
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
    cqp = ConformalQuantilePredictor(quantile=0.90)
    cqp.calibrate(y_pred_cal, y_true_cal)

    # Get P90 predictions on test set
    y_pred_p90 = cqp.conformalize_batch(y_pred_test)

    # Evaluate coverage
    stats = cqp.get_coverage_stats(y_pred_test, y_true_test)

    logger.info("Conformal Quantile Regression Results:")
    logger.info("  Target: P90 (90% coverage)")
    logger.info(f"  Actual coverage: {stats['coverage_percent']:.1f}%")
    logger.info(f"  Violation rate: {stats['violation_rate_percent']:.1f}%")
    logger.info(f"  Quantile adjustment: +{stats['quantile_adjustment']:.2f}")
    logger.info(f"  Calibrated alpha: {stats['calibrated_alpha']:.4f}")
    logger.info(f"  Calibration samples: {stats['calibration_samples']}")

    # Should be close to 90% coverage (allow ±5% tolerance with auto-calibration)
    assert 85 <= stats["coverage_percent"] <= 95, "Coverage should be close to 90%"
    logger.info("\n✓ Conformal prediction working correctly!")
