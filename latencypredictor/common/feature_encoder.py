"""
Centralized feature encoding for categorical and binned features.

This module provides a single source of truth for all categorical encodings
used across training and prediction servers, ensuring train-inference parity.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CategoricalFeature:
    """Definition of a categorical feature with string-to-integer mapping.

    Attributes:
        name: Source feature name (e.g., "pod_type")
        encoded_name: Output feature name (e.g., "pod_type_cat")
        mapping: Dictionary mapping string values to integer codes
        default_value: Integer code to use for unknown/missing values
        dtype: Output data type (default: "int32")
    """

    name: str
    encoded_name: str
    mapping: dict[str, int]
    default_value: int
    dtype: str = "int32"


@dataclass
class BinnedFeature:
    """Definition of a binned continuous feature.

    Bins a continuous feature into discrete buckets using uniform binning.

    Attributes:
        name: Source feature name (e.g., "prefix_cache_score")
        encoded_name: Output feature name (e.g., "prefill_score_bucket")
        num_bins: Number of bins to create
        min_value: Minimum value for clipping (default: 0.0)
        max_value: Maximum value for clipping (default: 1.0)
        dtype: Output data type (default: "int32")
    """

    name: str
    encoded_name: str
    num_bins: int
    min_value: float = 0.0
    max_value: float = 1.0
    dtype: str = "int32"


class FeatureEncoder:
    """Centralized feature encoding for train and inference.

    Provides multiple interfaces for encoding categorical and binned features:
    - DataFrame interface: encode_dataframe() for pandas DataFrames
    - Scalar interface: encode_value() for single values
    - Batch interface: encode_batch() for numpy arrays

    Encoding definitions are driven by a schema dict (see feature_schema.py).
    """

    def __init__(self, schema: dict):
        """Initialize encoder from a schema dict.

        Args:
            schema: Dict with 'categorical_features' and 'binned_features' keys.
                    See FEATURE_SCHEMA in feature_schema.py for the canonical format.
        """
        # Build categorical features from schema
        self.categorical_features: dict[str, CategoricalFeature] = {}
        for name, cfg in schema.get("categorical_features", {}).items():
            self.categorical_features[name] = CategoricalFeature(
                name=name,
                encoded_name=cfg["encoded_name"],
                mapping=cfg["mapping"],
                default_value=cfg["default_value"],
                dtype=cfg.get("dtype", "int32"),
            )

        # Build binned features from schema
        self.binned_features: dict[str, BinnedFeature] = {}
        for name, cfg in schema.get("binned_features", {}).items():
            self.binned_features[name] = BinnedFeature(
                name=name,
                encoded_name=cfg["encoded_name"],
                num_bins=cfg["num_bins"],
                min_value=cfg.get("min_value", 0.0),
                max_value=cfg.get("max_value", 1.0),
                dtype=cfg.get("dtype", "int32"),
            )

    @classmethod
    def from_default_schema(cls) -> "FeatureEncoder":
        """Create encoder from the built-in default schema."""
        from .feature_schema import FEATURE_SCHEMA

        return cls(FEATURE_SCHEMA)

    @classmethod
    def from_dict(cls, schema_dict: dict) -> "FeatureEncoder":
        """Create encoder from an arbitrary schema dict (e.g. loaded from a bundle)."""
        return cls(schema_dict)

    def encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode all categorical and binned features in a DataFrame (in-place).

        Args:
            df: DataFrame containing raw features

        Returns:
            Same DataFrame with encoded features added

        Notes:
            - Operates in-place for efficiency
            - Missing features are filled with default values
            - NaN values in categorical features are replaced with empty string
        """
        # Encode categorical features
        for feature_name, config in self.categorical_features.items():
            if feature_name in df.columns:
                # Map from a local copy so we never overwrite the raw column
                raw = df[feature_name].fillna("")
                df[config.encoded_name] = raw.map(config.mapping).fillna(config.default_value).astype(config.dtype)
            else:
                # Feature missing - use default value for all rows
                df[config.encoded_name] = np.full(len(df), config.default_value, dtype=config.dtype)

        # Encode binned features
        for feature_name, config in self.binned_features.items():
            if feature_name in df.columns:
                # Clip to range, multiply by num_bins, convert to int, clip upper bound
                df[config.encoded_name] = (
                    (df[feature_name].clip(config.min_value, config.max_value) * config.num_bins)
                    .astype(int)
                    .clip(upper=config.num_bins - 1)
                    .astype(config.dtype)
                )
            else:
                # Feature missing - use bin 0 for all rows
                df[config.encoded_name] = np.zeros(len(df), dtype=config.dtype)

        return df

    def encode_value(self, feature_name: str, value: Any) -> int:
        """Encode a single scalar value.

        Args:
            feature_name: Name of the feature to encode
            value: Raw value to encode

        Returns:
            Encoded integer value

        Raises:
            KeyError: If feature_name is not a known categorical or binned feature
        """
        # Try categorical features first
        if feature_name in self.categorical_features:
            config = self.categorical_features[feature_name]
            # Handle None/NaN as empty string
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = ""
            return config.mapping.get(value, config.default_value)

        # Try binned features
        if feature_name in self.binned_features:
            config = self.binned_features[feature_name]
            # Clip to range, multiply by num_bins, take floor, clip upper bound
            clipped = max(config.min_value, min(config.max_value, value))
            bin_index = int(clipped * config.num_bins)
            return min(bin_index, config.num_bins - 1)

        raise KeyError(f"Unknown feature: {feature_name}")

    def encode_batch(self, feature_name: str, values: np.ndarray) -> np.ndarray:
        """Encode a batch of values (numpy array).

        Args:
            feature_name: Name of the feature to encode
            values: Array of raw values to encode

        Returns:
            Array of encoded integer values

        Raises:
            KeyError: If feature_name is not a known categorical or binned feature
        """
        # Try categorical features first
        if feature_name in self.categorical_features:
            config = self.categorical_features[feature_name]
            # Vectorize the mapping lookup
            encoded = np.array(
                [config.mapping.get(v if v is not None else "", config.default_value) for v in values],
                dtype=config.dtype,
            )
            return encoded

        # Try binned features
        if feature_name in self.binned_features:
            config = self.binned_features[feature_name]
            # Replace NaN with min_value before clipping to avoid undefined int cast
            safe_values = np.nan_to_num(values.astype(np.float64), nan=config.min_value)
            clipped = np.clip(safe_values, config.min_value, config.max_value)
            bins = (clipped * config.num_bins).astype(np.int32)
            bins = np.clip(bins, 0, config.num_bins - 1)
            return bins.astype(config.dtype)

        raise KeyError(f"Unknown feature: {feature_name}")

    def to_dict(self) -> dict:
        """Export encoding schema for bundle metadata.

        Returns:
            Dictionary containing all encoding mappings and configurations,
            suitable for serialization to categorical_mapping.json
        """
        schema = {}

        # Export categorical features
        for feature_name, config in self.categorical_features.items():
            schema[f"{feature_name}_map"] = config.mapping
            schema[f"{feature_name}_map_description"] = (
                f"Maps {feature_name} string values to integer codes for {config.encoded_name}"
            )

        # Export binned features
        for feature_name, config in self.binned_features.items():
            # Use the output name for clarity (e.g., "prefix_buckets" not "prefix_cache_score_buckets")
            bucket_key = config.encoded_name.replace("_bucket", "_buckets")
            schema[bucket_key] = config.num_bins
            schema[f"{bucket_key}_description"] = f"Number of bins for {feature_name} (output: {config.encoded_name})"

        return schema
