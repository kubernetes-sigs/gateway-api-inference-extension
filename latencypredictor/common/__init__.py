"""
Common utilities and shared code for latency prediction system.

This package contains code shared between training and prediction servers:
- bundle_constants: Centralized bundle filename definitions
- conformal_quantile: Conformal prediction for quantile estimation
- lifecycle_state: Shared lifecycle state tracking for training/build coordination

Note: Bundle management (ModelBundle, BundleRegistry) is in training/ package
      since only the training server creates and manages bundles.
"""

# Re-export commonly used classes and constants for convenience
from .bundle_constants import (
    TPOT_CONFORMAL_FILENAME,
    TPOT_TREELITE_FILENAME,
    TTFT_CONFORMAL_FILENAME,
    TTFT_TREELITE_FILENAME,
)
from .conformal_quantile import ConformalQuantilePredictor
from .feature_encoder import FeatureEncoder
from .feature_schema import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION, compute_schema_hash
from .lifecycle_state import LifecycleState, read_lifecycle_state, write_lifecycle_state

__all__ = [
    # Bundle constants
    "TTFT_TREELITE_FILENAME",
    "TPOT_TREELITE_FILENAME",
    "TTFT_CONFORMAL_FILENAME",
    "TPOT_CONFORMAL_FILENAME",
    # Conformal prediction
    "ConformalQuantilePredictor",
    # Feature encoding
    "FeatureEncoder",
    "FEATURE_SCHEMA",
    "FEATURE_SCHEMA_VERSION",
    "compute_schema_hash",
    # Lifecycle state
    "LifecycleState",
    "read_lifecycle_state",
    "write_lifecycle_state",
]
