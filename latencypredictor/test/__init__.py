"""
Integration tests for latency prediction system.

This package contains end-to-end tests that verify:
- Training server accepts samples and trains models
- Prediction server syncs bundles and serves predictions
- TreeLite compilation works correctly
- Conformal calibration produces accurate quantiles
- High-QPS stress testing across multiple pods
"""

__all__ = ['test_dual_server_client']
