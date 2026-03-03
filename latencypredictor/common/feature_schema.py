"""
Feature encoding schema: single source of truth for categorical and binned feature definitions.

This schema drives FeatureEncoder construction and is embedded in bundle manifests
to guarantee train-inference parity across container versions.
"""

import hashlib
import json

FEATURE_SCHEMA_VERSION = 1

FEATURE_SCHEMA = {
    "schema_version": FEATURE_SCHEMA_VERSION,
    "categorical_features": {
        "pod_type": {
            "encoded_name": "pod_type_cat",
            "mapping": {"": 0, "prefill": 1, "decode": 2},
            "default_value": 0,
            "dtype": "int32",
        }
    },
    "binned_features": {
        "prefix_cache_score": {
            "encoded_name": "prefill_score_bucket",
            "num_bins": 4,
            "min_value": 0.0,
            "max_value": 1.0,
            "dtype": "int32",
        }
    },
}


def compute_schema_hash(schema: dict | None = None) -> str:
    """Deterministic SHA-256 hash (first 16 hex chars) via canonical JSON.

    Args:
        schema: Schema dict to hash. Defaults to FEATURE_SCHEMA.

    Returns:
        First 16 hex characters of the SHA-256 hash.
    """
    if schema is None:
        schema = FEATURE_SCHEMA
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
