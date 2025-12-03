"""
Training server for latency prediction models.

This package contains the training server that:
- Accepts training samples from prediction servers
- Trains XGBoost/LightGBM/BayesianRidge models for TTFT and TPOT prediction
- Performs conformal calibration for quantile prediction
- Compiles models to TreeLite for fast inference
- Publishes trained models as immutable versioned bundles
"""

__all__ = ['training_server']
