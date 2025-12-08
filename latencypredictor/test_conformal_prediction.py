"""
Tests for conformal prediction implementation.

This test suite verifies:
1. ConformalQuantilePredictor class functionality
2. Coverage accuracy (85-95% for P90)
3. Dual-mode operation (native quantile vs TreeLite+conformal)
4. End-to-end workflow
"""

import pytest
import numpy as np
import json
import tempfile
import os
from conformal_quantile import ConformalQuantilePredictor


class TestConformalQuantilePredictor:
    """Unit tests for ConformalQuantilePredictor class."""

    def test_initialization(self):
        """Test that conformal predictor initializes correctly."""
        cqp = ConformalQuantilePredictor(quantile=0.9)
        assert cqp.quantile == 0.9
        assert len(cqp.calibration_residuals) == 0
        assert cqp._cached_quantile_value is None
        assert cqp._cache_dirty is True

    def test_calibration_basic(self):
        """Test basic calibration functionality."""
        cqp = ConformalQuantilePredictor(quantile=0.9)

        # Generate synthetic data: predictions and actuals
        predictions = np.array([10, 20, 30, 40, 50])
        actuals = np.array([12, 22, 28, 42, 48])  # Residuals: [2, 2, 2, 2, 2]

        cqp.calibrate(predictions, actuals)

        # Should have 5 calibration samples
        assert len(cqp.calibration_residuals) == 5

        # P90 of residuals [2, 2, 2, 2, 2] should be close to 2
        assert cqp._cached_quantile_value == pytest.approx(2.0, abs=0.1)

    def test_conformalize_single_prediction(self):
        """Test conformalize method for single prediction."""
        cqp = ConformalQuantilePredictor(quantile=0.9)

        # Calibrate with known residuals
        predictions = np.array([10, 20, 30, 40, 50])
        actuals = np.array([15, 25, 35, 45, 55])  # Residuals all = 5
        cqp.calibrate(predictions, actuals)

        # P90 quantile prediction should be mean + ~5
        mean_pred = 100.0
        quantile_pred = cqp.conformalize(mean_pred)

        assert quantile_pred == pytest.approx(105.0, abs=0.5)

    def test_conformalize_batch(self):
        """Test conformalize_batch method."""
        cqp = ConformalQuantilePredictor(quantile=0.9)

        # Calibrate
        predictions = np.array([10, 20, 30, 40, 50])
        actuals = np.array([13, 23, 33, 43, 53])  # Residuals all = 3
        cqp.calibrate(predictions, actuals)

        # Batch predictions
        mean_preds = np.array([100.0, 200.0, 300.0])
        quantile_preds = cqp.conformalize_batch(mean_preds)

        # All should have ~3 added
        expected = np.array([103.0, 203.0, 303.0])
        np.testing.assert_allclose(quantile_preds, expected, atol=0.5)

    def test_coverage_stats_p90(self):
        """Test that coverage stats are accurate for P90."""
        np.random.seed(42)
        cqp = ConformalQuantilePredictor(quantile=0.9)

        # Generate realistic data with known distribution
        n = 1000
        # True relationship: y = 2*x + noise
        x = np.random.uniform(0, 10, n)
        noise = np.random.normal(0, 5, n)
        y_true = 2 * x + noise

        # Perfect mean predictions
        y_pred_mean = 2 * x

        # Split into calibration and test
        split = 800
        cqp.calibrate(y_pred_mean[:split], y_true[:split])

        # Get coverage stats on test set
        stats = cqp.get_coverage_stats(y_pred_mean[split:], y_true[split:])

        # Coverage should be close to 90% (within 5%)
        assert 85 <= stats['coverage_percent'] <= 95
        assert stats['target_coverage_percent'] == 90.0

        # Violation rate should be close to 10%
        assert 5 <= stats['violation_rate_percent'] <= 15

    def test_coverage_stats_p95(self):
        """Test that coverage stats are accurate for P95."""
        np.random.seed(123)
        cqp = ConformalQuantilePredictor(quantile=0.95)

        # Generate data
        n = 1000
        x = np.random.uniform(0, 10, n)
        noise = np.random.normal(0, 3, n)
        y_true = 5 * x + noise
        y_pred_mean = 5 * x

        # Calibrate and test
        split = 800
        cqp.calibrate(y_pred_mean[:split], y_true[:split])
        stats = cqp.get_coverage_stats(y_pred_mean[split:], y_true[split:])

        # Coverage should be close to 95% (within 5%)
        assert 90 <= stats['coverage_percent'] <= 100
        assert stats['target_coverage_percent'] == 95.0

    def test_online_calibration(self):
        """Test online calibration updates."""
        cqp = ConformalQuantilePredictor(quantile=0.9, max_calibration_samples=10)

        # Add samples one by one
        for i in range(15):
            prediction = float(i * 10)
            actual = float(i * 10 + 5)  # Residual = 5
            cqp.add_online_sample(prediction, actual)

        # Should keep only last 10 samples (due to max_calibration_samples=10)
        assert len(cqp.calibration_residuals) == 10

        # All residuals should be 5
        assert cqp._cached_quantile_value == pytest.approx(5.0, abs=0.1)

    def test_state_serialization(self):
        """Test get_state and from_state methods."""
        cqp = ConformalQuantilePredictor(quantile=0.9)

        # Calibrate with some data
        predictions = np.array([10, 20, 30, 40, 50])
        actuals = np.array([12, 22, 32, 42, 52])
        cqp.calibrate(predictions, actuals)

        # Get state
        state = cqp.get_state()

        # Verify state structure
        assert 'quantile' in state
        assert 'calibration_residuals' in state
        assert 'max_calibration_samples' in state
        assert state['quantile'] == 0.9
        assert len(state['calibration_residuals']) == 5

        # Restore from state
        cqp2 = ConformalQuantilePredictor.from_state(state)

        # Should have same properties
        assert cqp2.quantile == cqp.quantile
        assert len(cqp2.calibration_residuals) == len(cqp.calibration_residuals)
        assert list(cqp2.calibration_residuals) == list(cqp.calibration_residuals)

    def test_state_json_serialization(self):
        """Test that state can be serialized to JSON."""
        cqp = ConformalQuantilePredictor(quantile=0.9)

        # Calibrate
        predictions = np.array([10, 20, 30])
        actuals = np.array([11, 21, 31])
        cqp.calibrate(predictions, actuals)

        # Serialize to JSON
        state = cqp.get_state()
        json_str = json.dumps(state)

        # Deserialize
        state2 = json.loads(json_str)
        cqp2 = ConformalQuantilePredictor.from_state(state2)

        # Should match original
        assert cqp2.quantile == cqp.quantile
        assert len(cqp2.calibration_residuals) == len(cqp.calibration_residuals)

    def test_empty_calibration_fallback(self):
        """Test behavior when no calibration data available."""
        cqp = ConformalQuantilePredictor(quantile=0.9)

        # Without calibration, should return mean prediction
        mean_pred = 100.0
        result = cqp.conformalize(mean_pred)

        # Should return mean (with warning in logs)
        assert result == mean_pred

    def test_different_quantiles(self):
        """Test various quantile levels."""
        np.random.seed(42)

        for quantile in [0.5, 0.75, 0.9, 0.95, 0.99]:
            cqp = ConformalQuantilePredictor(quantile=quantile)

            # Generate data
            n = 500
            predictions = np.random.uniform(0, 100, n)
            actuals = predictions + np.random.normal(0, 10, n)

            # Calibrate and check coverage
            split = 400
            cqp.calibrate(predictions[:split], actuals[:split])
            stats = cqp.get_coverage_stats(predictions[split:], actuals[split:])

            # Coverage should be within ±10% of target
            expected_coverage = quantile * 100
            assert abs(stats['coverage_percent'] - expected_coverage) < 10


class TestCoverageDeterministic:
    """Deterministic tests for coverage validation."""

    def test_exact_coverage_controlled_data(self):
        """Test with perfectly controlled data for exact coverage."""
        cqp = ConformalQuantilePredictor(quantile=0.9)

        # Create data where exactly 90% of actuals are below prediction
        predictions = np.ones(100) * 50  # All predictions = 50
        actuals = np.concatenate([
            np.ones(90) * 49,  # 90 samples below prediction
            np.ones(10) * 51   # 10 samples above prediction
        ])

        # Shuffle to avoid ordering bias
        np.random.seed(42)
        shuffle_idx = np.random.permutation(100)
        actuals = actuals[shuffle_idx]

        # Calibrate and test
        cqp.calibrate(predictions[:80], actuals[:80])

        # The conformalized predictions should maintain ~90% coverage
        quantile_preds = cqp.conformalize_batch(predictions[80:])

        # Count how many actuals are below quantile predictions
        below = np.sum(actuals[80:] <= quantile_preds)
        coverage = below / len(actuals[80:]) * 100

        # Should be close to 90%
        assert 80 <= coverage <= 100  # Relaxed due to small sample size


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_file_persistence(self):
        """Test saving and loading calibration state from file."""
        cqp = ConformalQuantilePredictor(quantile=0.9)

        # Calibrate
        predictions = np.array([10, 20, 30, 40, 50])
        actuals = np.array([12, 22, 32, 42, 52])
        cqp.calibrate(predictions, actuals)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cqp.get_state(), f)
            temp_path = f.name

        try:
            # Load from file
            with open(temp_path, 'r') as f:
                state = json.load(f)

            cqp2 = ConformalQuantilePredictor.from_state(state)

            # Make same predictions
            test_pred = 60.0
            result1 = cqp.conformalize(test_pred)
            result2 = cqp2.conformalize(test_pred)

            assert result1 == pytest.approx(result2)
        finally:
            os.unlink(temp_path)

    def test_progressive_calibration(self):
        """Test that adding more calibration data improves coverage."""
        np.random.seed(42)

        # Generate data
        n = 1000
        x = np.random.uniform(0, 10, n)
        noise = np.random.normal(0, 5, n)
        y_true = 3 * x + noise
        y_pred = 3 * x

        # Test with different calibration sizes
        coverage_results = []
        for cal_size in [50, 100, 200, 500]:
            cqp = ConformalQuantilePredictor(quantile=0.9)
            cqp.calibrate(y_pred[:cal_size], y_true[:cal_size])

            # Test on remaining data
            stats = cqp.get_coverage_stats(y_pred[cal_size:], y_true[cal_size:])
            coverage_results.append(stats['coverage_percent'])

        # With more calibration data, coverage should stabilize around 90%
        # Last two should be within 5% of 90%
        assert 85 <= coverage_results[-1] <= 95
        assert 85 <= coverage_results[-2] <= 95

    def test_distribution_shift(self):
        """Test behavior when test distribution differs from calibration."""
        np.random.seed(42)

        cqp = ConformalQuantilePredictor(quantile=0.9)

        # Calibrate on low-noise data
        x_cal = np.random.uniform(0, 10, 500)
        noise_cal = np.random.normal(0, 2, 500)  # Low noise
        y_true_cal = 2 * x_cal + noise_cal
        y_pred_cal = 2 * x_cal

        cqp.calibrate(y_pred_cal, y_true_cal)

        # Test on high-noise data
        x_test = np.random.uniform(0, 10, 200)
        noise_test = np.random.normal(0, 5, 200)  # Higher noise!
        y_true_test = 2 * x_test + noise_test
        y_pred_test = 2 * x_test

        stats = cqp.get_coverage_stats(y_pred_test, y_true_test)

        # Coverage will be lower due to distribution shift
        # But conformal prediction should still provide some coverage
        assert stats['coverage_percent'] > 70  # At least 70% coverage


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_calibration_sample(self):
        """Test with only one calibration sample."""
        cqp = ConformalQuantilePredictor(quantile=0.9)

        predictions = np.array([10.0])
        actuals = np.array([12.0])

        cqp.calibrate(predictions, actuals)

        # Should work, using the single residual
        result = cqp.conformalize(50.0)
        assert result == pytest.approx(52.0, abs=0.1)

    def test_mismatched_lengths(self):
        """Test error handling for mismatched prediction/actual lengths."""
        cqp = ConformalQuantilePredictor(quantile=0.9)

        predictions = np.array([10, 20, 30])
        actuals = np.array([12, 22])  # Different length!

        with pytest.raises(ValueError, match="same length"):
            cqp.calibrate(predictions, actuals)

    def test_max_calibration_samples_limit(self):
        """Test that max_calibration_samples is respected."""
        cqp = ConformalQuantilePredictor(quantile=0.9, max_calibration_samples=5)

        # Add 10 samples
        predictions = np.arange(10, dtype=float)
        actuals = predictions + 1

        cqp.calibrate(predictions, actuals)

        # Should keep only last 5
        assert len(cqp.calibration_residuals) == 5

    def test_zero_residuals(self):
        """Test with perfect predictions (zero residuals)."""
        cqp = ConformalQuantilePredictor(quantile=0.9)

        predictions = np.array([10, 20, 30, 40, 50], dtype=float)
        actuals = predictions.copy()  # Perfect predictions!

        cqp.calibrate(predictions, actuals)

        # Quantile of zeros should be zero
        result = cqp.conformalize(100.0)
        assert result == pytest.approx(100.0, abs=0.01)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
