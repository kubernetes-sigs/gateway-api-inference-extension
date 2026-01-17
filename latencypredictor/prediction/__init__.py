"""
Prediction server for latency estimation.

This package contains the prediction server that:
- Syncs trained model bundles from training server via HTTP
- Loads TreeLite-compiled models for fast inference
- Applies conformal calibration to convert mean → quantile predictions
- Serves predictions via FastAPI endpoint
- Reports actual latencies back to training server for continuous learning
"""

__all__ = ['prediction_server']
