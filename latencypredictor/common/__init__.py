"""
Common utilities and shared code for latency prediction system.

This package contains code shared between training and prediction servers:
- bundle_constants: Centralized bundle filename definitions
- conformal_quantile: Conformal prediction for quantile estimation

Note: Bundle management (ModelBundle, BundleRegistry) is in training/ package
      since only the training server creates and manages bundles.
"""

# Re-export commonly used classes and constants for convenience
from .bundle_constants import (
    TTFT_MODEL_FILENAME,
    TPOT_MODEL_FILENAME,
    TTFT_SCALER_FILENAME,
    TPOT_SCALER_FILENAME,
    TTFT_TREELITE_FILENAME,
    TPOT_TREELITE_FILENAME,
    TTFT_CONFORMAL_FILENAME,
    TPOT_CONFORMAL_FILENAME,
)

from .conformal_quantile import ConformalQuantilePredictor

__all__ = [
    # Bundle constants
    'TTFT_MODEL_FILENAME',
    'TPOT_MODEL_FILENAME',
    'TTFT_SCALER_FILENAME',
    'TPOT_SCALER_FILENAME',
    'TTFT_TREELITE_FILENAME',
    'TPOT_TREELITE_FILENAME',
    'TTFT_CONFORMAL_FILENAME',
    'TPOT_CONFORMAL_FILENAME',
    # Conformal prediction
    'ConformalQuantilePredictor',
]
