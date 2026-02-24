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
import json
import logging
import os
import random
import threading
import time
from collections import deque
from datetime import UTC, datetime
from enum import Enum

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Please install with: pip install xgboost")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Please install with: pip install lightgbm")

# Import bundle integration
from common.bundle_constants import TPOT_CONFORMAL_FILENAME, TTFT_CONFORMAL_FILENAME, get_model_filename
from common.conformal_quantile import ConformalQuantilePredictor
from common.feature_encoder import FeatureEncoder

from .bundle_integration import BundleModelManager


class ModelType(str, Enum):
    """Supported model types (TreeLite-compatible only)."""

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class RandomDropDeque(deque):
    def __init__(self, maxlen):
        super().__init__()
        self._maxlen = maxlen

    def append(self, item):
        if len(self) >= self._maxlen:
            # pick a random index to evict
            idx = random.randrange(len(self))
            # rotate so that element at idx moves to the left end
            self.rotate(-idx)
            # remove it
            self.popleft()
            # rotate back to original ordering
            self.rotate(idx)
        super().append(item)

    def appendleft(self, item):
        if len(self) >= self._maxlen:
            idx = random.randrange(len(self))
            # rotate so that element at idx moves to the right end
            self.rotate(len(self) - idx - 1)
            self.pop()
            # rotate back
            self.rotate(-(len(self) - idx - 1))
        super().appendleft(item)


# --- Configuration ---
class Settings:
    """
    Configuration class for the latency predictor server.
    Reads settings from environment variables with sensible defaults.
    """

    # Model Configuration
    MODEL_TYPE: str = os.getenv("LATENCY_MODEL_TYPE", "xgboost")  # xgboost or lightgbm
    QUANTILE_ALPHA: float = float(os.getenv("LATENCY_QUANTILE_ALPHA", "0.85"))  # p85 quantile by default

    # Training Parameters
    RETRAINING_INTERVAL_SEC: int = int(os.getenv("LATENCY_RETRAINING_INTERVAL_SEC", 1800))  # 30 minutes
    MIN_SAMPLES_FOR_RETRAIN_FRESH: int = int(
        os.getenv("LATENCY_MIN_SAMPLES_FOR_RETRAIN_FRESH", 10)
    )  # Bootstrap threshold
    MIN_SAMPLES_FOR_RETRAIN: int = int(
        os.getenv("LATENCY_MIN_SAMPLES_FOR_RETRAIN", 1000)
    )  # Normal retraining threshold

    # Data Management
    MAX_TRAINING_DATA_SIZE_PER_BUCKET: int = int(os.getenv("LATENCY_MAX_TRAINING_DATA_SIZE_PER_BUCKET", 10000))
    TEST_TRAIN_RATIO: float = float(os.getenv("LATENCY_TEST_TRAIN_RATIO", "0.1"))  # 10% test, 90% train
    MAX_TEST_DATA_SIZE: int = int(os.getenv("LATENCY_MAX_TEST_DATA_SIZE", "1000"))  # Max test samples to retain

    # Feature Engineering
    SAMPLE_WEIGHTING_FOR_PREFIX_CACHE: bool = (
        os.getenv("LATENCY_SAMPLE_WEIGHTING_FOR_PREFIX_CACHE", "false").lower() == "true"
    )

    # Bundle System
    BUNDLE_DIR: str = os.getenv("BUNDLE_DIR", "/models/bundles")
    BUNDLE_CURRENT_SYMLINK: str = os.getenv("BUNDLE_CURRENT_SYMLINK", "/models/current")
    MAX_BUNDLES_TO_KEEP: int = int(os.getenv("MAX_BUNDLES_TO_KEEP", "5"))


settings = Settings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Add this to your Pydantic models section
class ModelInfoResponse(BaseModel):
    model_type: str
    xgboost_available: bool
    lightgbm_available: bool = Field(
        default=False, description="Whether LightGBM is available"
    )  # FIXED: Added this field

    is_ready: bool
    ttft_training_samples: int = Field(default=0, description="Number of TTFT training samples")
    tpot_training_samples: int = Field(default=0, description="Number of TPOT training samples")
    ttft_test_samples: int = Field(default=0, description="Number of TTFT test samples")
    tpot_test_samples: int = Field(default=0, description="Number of TPOT test samples")
    last_retrain_time: datetime | None = Field(default=None, description="Last retraining timestamp")
    min_samples_for_retrain: int = Field(default=0, description="Minimum samples required for retraining")
    retraining_interval_sec: int = Field(default=0, description="Retraining interval in seconds")


class FlushRequest(BaseModel):
    flush_training_data: bool = Field(default=True, description="Flush training data buckets")
    flush_test_data: bool = Field(default=True, description="Flush test data")
    flush_metrics: bool = Field(default=True, description="Flush quantile metric scores")
    reason: str | None = Field(default=None, description="Optional reason for flushing")


class FlushResponse(BaseModel):
    success: bool
    flushed_at: datetime
    reason: str | None = None
    ttft_training_samples_flushed: int
    tpot_training_samples_flushed: int
    ttft_test_samples_flushed: int
    tpot_test_samples_flushed: int
    metrics_cleared: bool
    message: str


def quantile_loss(y_true, y_pred, quantile):
    """
    Calculate quantile loss (also known as pinball loss).

    For quantile τ (tau), the loss is:
    - (τ - 1) * (y_true - y_pred) if y_true < y_pred (under-prediction)
    - τ * (y_true - y_pred) if y_true >= y_pred (over-prediction)

    Args:
        y_true: actual values
        y_pred: predicted quantile values
        quantile: the quantile being predicted (e.g., 0.85 for p85)

    Returns:
        Mean quantile loss
    """
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)


def quantile_coverage(y_true, y_pred, quantile):
    """
    Calculate quantile coverage - the proportion of actual values that fall below the predicted quantile.

    For a well-calibrated p85 model, this should be close to 0.85 (85%).

    Args:
        y_true: actual values
        y_pred: predicted quantile values
        quantile: the quantile being predicted (e.g., 0.85 for p85)

    Returns:
        Coverage percentage (0-100)
    """
    below_prediction = np.sum(y_true <= y_pred)
    coverage = below_prediction / len(y_true)
    return coverage * 100


def quantile_violation_rate(y_true, y_pred, quantile):
    """
    Calculate quantile violation rate - the proportion of times actual values exceed the predicted quantile.

    For a well-calibrated p85 model, this should be close to 0.15 (15%).

    Args:
        y_true: actual values
        y_pred: predicted quantile values
        quantile: the quantile being predicted (e.g., 0.85 for p85)

    Returns:
        Violation rate percentage (0-100)
    """
    violations = np.sum(y_true > y_pred)
    violation_rate = violations / len(y_true)
    return violation_rate * 100


class LatencyPredictor:
    """
    Manages model training, prediction, and data handling.
    """

    def __init__(self, model_type: str = None):
        # Set model type with validation
        if model_type is None:
            model_type = settings.MODEL_TYPE

        if model_type not in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(ModelType)}")

        if model_type == ModelType.XGBOOST and not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost requested but not available. Install with: pip install xgboost")

        if model_type == ModelType.LIGHTGBM and not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM requested but not available. Install with: pip install lightgbm")

        self.model_type = ModelType(model_type)
        self.quantile = settings.QUANTILE_ALPHA
        logging.info(f"Initialized LatencyPredictor with model type: {self.model_type}, quantile: {self.quantile}")

        # Initialize feature encoder for categorical encoding
        self.feature_encoder = FeatureEncoder()

        # Data buckets for sampling
        self.cache_buckets = int(1.0 / 0.05)  # 20 buckets for cache percentage (0-100% in 5% increments)
        self.queue_buckets = 5  # 0, 1-2, 3-5, 6-10, 11+ waiting requests
        self.prefix_buckets = 4  # NEW: 0-25%, 25-50%, 50-75%, 75-100% prefix cache score
        self.bucket_size = settings.MAX_TRAINING_DATA_SIZE_PER_BUCKET

        # Data buckets with tuple keys: (queue_bucket, cache_bucket, prefix_bucket)
        self.ttft_data_buckets = {
            (q, c, p): deque(maxlen=self.bucket_size)
            for q in range(self.queue_buckets)
            for c in range(self.cache_buckets)
            for p in range(self.prefix_buckets)  # NEW: Added prefix dimension
        }
        self.tpot_data_buckets = {
            (q, c, p): deque(maxlen=self.bucket_size)
            for q in range(self.queue_buckets)
            for c in range(self.cache_buckets)
            for p in range(self.prefix_buckets)  # NEW: Added prefix dimension
        }

        # Test data storage with configurable max size
        self.ttft_test_data = deque(maxlen=settings.MAX_TEST_DATA_SIZE)
        self.tpot_test_data = deque(maxlen=settings.MAX_TEST_DATA_SIZE)

        # Quantile-specific metric tracking (store last 5 scores)
        self.ttft_quantile_loss_scores = deque(maxlen=5)
        self.tpot_quantile_loss_scores = deque(maxlen=5)
        self.ttft_coverage_scores = deque(maxlen=5)
        self.tpot_coverage_scores = deque(maxlen=5)
        self.ttft_violation_rates = deque(maxlen=5)
        self.tpot_violation_rates = deque(maxlen=5)

        self.ttft_model = None
        self.tpot_model = None

        # Initialize bundle manager for model distribution
        self.bundle_manager = BundleModelManager(
            bundle_dir=settings.BUNDLE_DIR,
            current_symlink=settings.BUNDLE_CURRENT_SYMLINK,
            max_bundles=settings.MAX_BUNDLES_TO_KEEP,
        )
        logging.info(f"Initialized bundle manager: {settings.BUNDLE_DIR}")

        self.lock = threading.Lock()
        self.last_retrain_time = None
        self._shutdown_event = threading.Event()
        self._training_thread: threading.Thread = None

    def _get_prefix_bucket(self, prefix_score: float) -> int:
        """Map prefix cache score to bucket index."""
        score = max(0.0, min(1.0, prefix_score))
        return min(int(score * self.prefix_buckets), self.prefix_buckets - 1)

    def _get_queue_bucket(self, num_waiting: int) -> int:
        """Map number of waiting requests to queue bucket index."""
        if num_waiting == 0:
            return 0
        elif num_waiting <= 2:
            return 1
        elif num_waiting <= 5:
            return 2
        elif num_waiting <= 10:
            return 3
        else:
            return 4  # 11+ requests

    def _get_cache_bucket(self, cache_percentage: float) -> int:
        """Map cache percentage to cache bucket index."""
        pct = max(0.0, min(1.0, cache_percentage))
        return min(int(pct * self.cache_buckets), self.cache_buckets - 1)

    def _get_bucket_key(self, sample: dict) -> tuple:
        """Get (queue_bucket, cache_bucket) tuple key for a sample."""
        queue_bucket = self._get_queue_bucket(sample["num_request_waiting"])
        cache_bucket = self._get_cache_bucket(sample["kv_cache_percentage"])
        prefix_bucket = self._get_prefix_bucket(sample["prefix_cache_score"])  # NEW

        return (queue_bucket, cache_bucket, prefix_bucket)

    def _prepare_features_with_interaction(self, df: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """
        Prepare features with interaction terms for better model learning.
        Args:
            df: DataFrame with raw features
            model_type: 'ttft' or 'tpot'
        Returns:
            DataFrame with engineered features including interactions
        """
        # Encode all categorical features using centralized encoder
        self.feature_encoder.encode_dataframe(df)

        if model_type == "ttft":
            # Create interaction: prefix score * input length
            # This captures that prefix caching benefit scales with input size
            df["effective_input_tokens"] = (1 - df["prefix_cache_score"]) * (df["input_token_length"])

            # Return TTFT features with interaction and pod_type
            feature_cols = [
                "kv_cache_percentage",
                "input_token_length",
                "num_request_waiting",
                "num_request_running",
                "prefix_cache_score",
                "effective_input_tokens",
                "prefill_score_bucket",
                "pod_type_cat",
            ]
            return df[feature_cols]
        else:  # tpot
            # TPOT doesn't use prefix_cache_score, so no interaction needed
            feature_cols = [
                "kv_cache_percentage",
                "input_token_length",
                "num_request_waiting",
                "num_request_running",
                "num_tokens_generated",
                "pod_type_cat",
            ]
            return df[feature_cols]

    def shutdown(self):
        """Signal the training thread to exit and join it."""
        self._shutdown_event.set()
        if self._training_thread is not None:
            self._training_thread.join()

    @property
    def is_ready(self) -> bool:
        """Checks if all models are loaded/trained."""
        return all([self.ttft_model, self.tpot_model])

    @is_ready.setter
    def is_ready(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("is_ready must be a boolean value.")
        self._is_ready_override = value

    def _all_samples(self, buckets: dict) -> list:
        samples = []
        for bucket_deque in buckets.values():
            samples.extend(bucket_deque)
        return samples

    def _train_model_with_scaling(
        self, features: pd.DataFrame, target: pd.Series, model_name: str = None, sample_weight: np.ndarray = None
    ) -> xgb.XGBRegressor | lgb.LGBMRegressor:
        try:
            if len(features) == 0 or len(target) == 0:
                raise ValueError("Empty training data")
            if features.isnull().any().any() or target.isnull().any():
                raise ValueError("Training data contains NaN values")
            # Check only numeric columns for infinity (categorical columns cause isinf to fail)
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) > 0 and np.isinf(numeric_features.values).any():
                raise ValueError("Training data contains infinite values")
            if np.isinf(target.values).any():
                raise ValueError("Target data contains infinite values")

            if self.model_type == ModelType.XGBOOST:  # XGBoost with quantile regression
                if model_name == "ttft":
                    # enforce your TTFT feature order (including pod_type_cat)
                    ttft_order = [
                        "kv_cache_percentage",
                        "input_token_length",
                        "num_request_waiting",
                        "num_request_running",
                        "prefix_cache_score",
                        "effective_input_tokens",
                        "prefill_score_bucket",
                        "pod_type_cat",
                    ]
                    if list(features.columns) != ttft_order:
                        try:
                            features = features[ttft_order]
                        except Exception:
                            raise ValueError(
                                f"TTFT features must be exactly {ttft_order}; got {list(features.columns)}"
                            )

                    # ---- (A) Build a warm-start stump that must split on prefill_score_bucket ----
                    # Train on the SAME full feature set, but freeze all other features to constants,
                    # so the only useful split is the prefix bucket.
                    features_stump = features.copy()
                    for col in features_stump.columns:
                        if col != "prefill_score_bucket":
                            # keep dtype, set to a constant scalar
                            const_val = features_stump[col].iloc[0]
                            features_stump[col] = const_val

                    # Ensure prefill bucket is int32 (already done in _prepare_features_with_interaction)
                    features_stump["prefill_score_bucket"] = features_stump["prefill_score_bucket"].astype("int32")

                    # TreeLite-only mode: Use standard regression + conformal prediction for quantiles
                    # TreeLite 4.6.1 does NOT support XGBoost categorical features or quantileerror objective
                    model = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        gamma=0.2,
                        reg_alpha=0.01,
                        reg_lambda=0.1,
                        objective="reg:squarederror",  # Always mean regression for TreeLite
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=42,
                        verbosity=1,
                        enable_categorical=False,  # Always False for TreeLite compatibility
                    )
                    model.fit(features, target, sample_weight=sample_weight)
                    return model

                elif model_name == "tpot":
                    tpot_order = [
                        "kv_cache_percentage",
                        "input_token_length",
                        "num_request_waiting",
                        "num_request_running",
                        "num_tokens_generated",
                        "pod_type_cat",
                    ]
                    if list(features.columns) != tpot_order:
                        try:
                            features = features[tpot_order]
                        except Exception as _:
                            raise ValueError(
                                f"TPOT features must be exactly {tpot_order}; got {list(features.columns)}"
                            )

                # TreeLite-only mode: Use standard regression + conformal prediction for quantiles
                # TreeLite 4.6.1 does NOT support XGBoost categorical features or quantileerror objective
                model = xgb.XGBRegressor(
                    n_estimators=200,  # Number of trees to build (moderate value for balanced accuracy and speed)
                    max_depth=6,  # Depth of trees; 6 is typically a sweet spot balancing bias/variance
                    learning_rate=0.05,  # Smaller learning rate to achieve stable convergence
                    subsample=0.8,  # Use 80% of data per tree (adds regularization & reduces overfitting)
                    colsample_bytree=0.8,  # Use 80% of features per tree (improves generalization)
                    min_child_weight=5,  # Low value allows fine-grained splits near p90 boundary (prevents overprediction)
                    gamma=0.2,  # Low gamma allows splits with small loss reduction (critical for quantile accuracy)
                    reg_alpha=0.01,  # L1 regularization (Lasso) - encourages sparsity
                    reg_lambda=0.1,  # L2 regularization (Ridge) - prevents large coefficients
                    objective="reg:squarederror",  # Always mean regression for TreeLite
                    tree_method="hist",  # Efficient histogram algorithm; optimal for large datasets
                    n_jobs=-1,  # Utilize all CPU cores for parallel training
                    random_state=42,  # Ensures reproducible results
                    verbosity=1,
                    enable_categorical=False,  # Always False for TreeLite compatibility
                )
                model.fit(features, target, sample_weight=sample_weight)
                return model
            elif self.model_type == ModelType.LIGHTGBM:  # LightGBM with standard regression
                # TreeLite-only mode: Use standard regression + conformal prediction for quantiles
                # LightGBM can use categorical_feature hints with TreeLite (better than XGBoost for this)
                model = lgb.LGBMRegressor(
                    n_estimators=200,  # Number of trees
                    max_depth=6,  # Maximum tree depth
                    learning_rate=0.05,  # Learning rate
                    subsample=0.8,  # Row sampling ratio
                    colsample_bytree=0.8,  # Column sampling ratio
                    min_child_samples=20,  # Minimum samples in leaf
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=0.1,  # L2 regularization
                    objective="regression",  # Always mean regression for TreeLite
                    n_jobs=-1,  # Use all cores
                    random_state=42,  # Reproducibility
                    verbosity=-1,  # Suppress warnings
                    force_col_wise=True,  # Better for small datasets
                )
                # LightGBM can hint at categorical features even with int32 encoding
                # Include pod_type_cat for both TTFT and TPOT
                categorical_features = (
                    ["prefill_score_bucket", "pod_type_cat"] if model_name == "ttft" else ["pod_type_cat"]
                )
                model.fit(features, target, sample_weight=sample_weight, categorical_feature=categorical_features)
                return model

        except Exception as e:
            logging.error(f"Error in _train_model_with_scaling: {e}", exc_info=True)
            raise

    def _calculate_quantile_metrics_on_test(self, model, test_data, model_name, target_col):
        """Calculate quantile-specific metrics on test data"""
        try:
            df_raw = pd.DataFrame(test_data).dropna()

            # Check if target column exists
            if target_col not in df_raw.columns:
                logging.warning(
                    f"Target column '{target_col}' not found in test data for {model_name}. "
                    f"Available columns: {list(df_raw.columns)}. "
                    f"Test data size: {len(test_data)}"
                )
                return None, None, None

            df_raw = df_raw[df_raw[target_col] > 0]

            if len(df_raw) < 2:
                return None, None, None

            # Apply feature engineering to create interaction terms and categorical features
            df_features = self._prepare_features_with_interaction(df_raw.copy(), model_type=model_name)

            # Get appropriate feature columns based on model type and name
            if model_name == "ttft":
                # XGBoost or LightGBM
                feature_cols = [
                    "kv_cache_percentage",
                    "input_token_length",
                    "num_request_waiting",
                    "num_request_running",
                    "prefix_cache_score",
                    "effective_input_tokens",
                    "prefill_score_bucket",
                    "pod_type_cat",
                ]
            else:  # tpot
                feature_cols = [
                    "kv_cache_percentage",
                    "input_token_length",
                    "num_request_waiting",
                    "num_request_running",
                    "num_tokens_generated",
                    "pod_type_cat",
                ]

            X = df_features[feature_cols]
            y_true = df_raw[target_col].values
            y_pred = model.predict(X)

            # Calculate quantile-specific metrics
            ql = quantile_loss(y_true, y_pred, self.quantile)
            coverage = quantile_coverage(y_true, y_pred, self.quantile)
            violation_rate = quantile_violation_rate(y_true, y_pred, self.quantile)

            return ql, coverage, violation_rate

        except Exception as e:
            logging.error(f"Error calculating quantile metrics: {e}", exc_info=True)
            return None, None, None

    def _create_default_model(self, model_type: str) -> xgb.XGBRegressor | lgb.LGBMRegressor:
        """Creates and trains a simple default model with initial priors."""
        try:
            logging.info(f"Creating default '{model_type}' model with priors.")
            if model_type == "ttft":
                features = pd.DataFrame(
                    {
                        "kv_cache_percentage": [
                            0.0,
                        ],
                        "input_token_length": [
                            1,
                        ],
                        "num_request_waiting": [
                            0,
                        ],
                        "num_request_running": [
                            0,
                        ],
                        "prefix_cache_score": [
                            0.0,
                        ],  # Added prefix_cache_score
                    }
                )
                features = self._prepare_features_with_interaction(features, "ttft")
                target = pd.Series([10.0])

            else:
                features = pd.DataFrame(
                    {
                        "kv_cache_percentage": [0.0],
                        "input_token_length": [1],  # Added input_token_length
                        "num_request_waiting": [
                            0,
                        ],
                        "num_request_running": [
                            0,
                        ],
                        "num_tokens_generated": [
                            1,
                        ],
                    }
                )
                features = self._prepare_features_with_interaction(features, "tpot")
                target = pd.Series([10.0])
            return self._train_model_with_scaling(features, target, model_name=model_type)
        except Exception as e:
            logging.error(f"Error creating default model for {model_type}: {e}", exc_info=True)
            raise

    def train(self):
        try:
            # Create snapshots and validate sample count
            with self.lock:
                ttft_snap = list(self._all_samples(self.ttft_data_buckets))
                tpot_snap = list(self._all_samples(self.tpot_data_buckets))
                total = len(ttft_snap) + len(tpot_snap)
                if total < settings.MIN_SAMPLES_FOR_RETRAIN:
                    logging.info(f"Skipping training: only {total} samples (< {settings.MIN_SAMPLES_FOR_RETRAIN}).")
                    return
                logging.info(
                    f"Initiating training with {total} samples using {self.model_type} for quantile {self.quantile}."
                )

                # Create new bundle for this training session (inside lock to access model_type)
                bundle = self.bundle_manager.start_training(
                    model_name="combined",  # Single bundle contains both TTFT and TPOT
                    model_type=self.model_type.value,
                    quantile=self.quantile,
                )
                logging.info(f"Started training session: bundle_id={bundle.manifest.bundle_id[:8]}")

            new_ttft_model = None
            new_tpot_model = None

            # Train TTFT
            if ttft_snap:
                raw_ttft = pd.DataFrame(ttft_snap).dropna()
                raw_ttft = raw_ttft[raw_ttft["actual_ttft_ms"] > 0]
                df_ttft = self._prepare_features_with_interaction(raw_ttft.copy(), model_type="ttft")
                logging.debug(f"TTFT training data size: {len(df_ttft)} with sample data: {df_ttft.columns.tolist()}")
                if len(df_ttft) >= settings.MIN_SAMPLES_FOR_RETRAIN:
                    # Updated TTFT features to include prefix_cache_score and pod_type_cat
                    ttft_feature_cols_tree = [
                        "kv_cache_percentage",
                        "input_token_length",
                        "num_request_waiting",
                        "num_request_running",
                        "prefix_cache_score",
                        "effective_input_tokens",
                        "prefill_score_bucket",
                        "pod_type_cat",
                    ]
                    ttft_feature_cols_br = [
                        "kv_cache_percentage",
                        "input_token_length",
                        "num_request_waiting",
                        "num_request_running",
                        "prefix_cache_score",
                        "effective_input_tokens",
                    ]

                    # Build X_ttft for all model types, then trim for BR
                    X_ttft = df_ttft[ttft_feature_cols_tree]

                    y_ttft = raw_ttft["actual_ttft_ms"]

                    try:
                        # raw_ttft still has the original columns including 'prefix_cache_score'
                        raw_ttft["_prefix_bucket"] = (
                            raw_ttft["prefix_cache_score"]
                            .clip(0, 1)
                            .apply(lambda s: min(int(s * self.prefix_buckets), self.prefix_buckets - 1))
                        )

                        bucket_counts = raw_ttft["_prefix_bucket"].value_counts().to_dict()
                        total_ttft = len(raw_ttft)
                        num_buckets = max(1, len(bucket_counts))
                        epsilon = 1.0
                        bucket_weights = {
                            p: total_ttft / (num_buckets * (cnt + epsilon)) for p, cnt in bucket_counts.items()
                        }
                        sample_weight_ttft = None
                        if settings.SAMPLE_WEIGHTING_FOR_PREFIX_CACHE:
                            sample_weight_ttft = raw_ttft["_prefix_bucket"].map(bucket_weights).astype(float).to_numpy()
                            sample_weight_ttft *= len(sample_weight_ttft) / sample_weight_ttft.sum()

                        result = self._train_model_with_scaling(
                            X_ttft, y_ttft, model_name="ttft", sample_weight=sample_weight_ttft
                        )
                        new_ttft_model = result

                        # Quantile metrics on test set
                        ql = coverage = violation_rate = None
                        if self.ttft_test_data:
                            ql, coverage, violation_rate = self._calculate_quantile_metrics_on_test(
                                new_ttft_model,
                                list(self.ttft_test_data),  # Pass raw data
                                "ttft",  # Pass model name instead of feature_cols
                                "actual_ttft_ms",
                            )

                        if ql is not None:
                            self.ttft_quantile_loss_scores.append(ql)
                            self.ttft_coverage_scores.append(coverage)
                            self.ttft_violation_rates.append(violation_rate)
                            logging.info(
                                f"TTFT model trained on {len(df_ttft)} samples. "
                                f"Quantile Loss = {ql:.4f}, "
                                f"Coverage = {coverage:.2f}% (target: {self.quantile*100:.0f}%), "
                                f"Violation Rate = {violation_rate:.2f}% (target: {(1-self.quantile)*100:.0f}%)"
                            )
                        else:
                            logging.info(
                                f"TTFT model trained on {len(df_ttft)} samples. Quantile metrics = N/A (insufficient test data)"
                            )

                    except Exception:
                        logging.error("Error training TTFT model", exc_info=True)

            # Train TPOT
            if tpot_snap:
                df_tpot = pd.DataFrame(tpot_snap).dropna()
                df_tpot = df_tpot[df_tpot["actual_tpot_ms"] > 0]
                if len(df_tpot) >= settings.MIN_SAMPLES_FOR_RETRAIN:
                    # TPOT features - use feature preparation to add pod_type_cat
                    X_tpot = self._prepare_features_with_interaction(df_tpot.copy(), model_type="tpot")
                    y_tpot = df_tpot["actual_tpot_ms"]
                    try:
                        result = self._train_model_with_scaling(X_tpot, y_tpot, model_name="tpot")
                        new_tpot_model = result

                        # Calculate quantile metrics on test data
                        ql, coverage, violation_rate = self._calculate_quantile_metrics_on_test(
                            new_tpot_model,
                            list(self.tpot_test_data),  # Pass raw data
                            "tpot",  # Pass model name instead of feature_cols
                            "actual_tpot_ms",
                        )

                        if ql is not None:
                            self.tpot_quantile_loss_scores.append(ql)
                            self.tpot_coverage_scores.append(coverage)
                            self.tpot_violation_rates.append(violation_rate)
                            logging.info(
                                f"TPOT model trained on {len(df_tpot)} samples. "
                                f"Quantile Loss = {ql:.4f}, "
                                f"Coverage = {coverage:.2f}% (target: {self.quantile*100:.0f}%), "
                                f"Violation Rate = {violation_rate:.2f}% (target: {(1-self.quantile)*100:.0f}%)"
                            )
                        else:
                            logging.info(
                                f"TPOT model trained on {len(df_tpot)} samples. Quantile metrics = N/A (insufficient test data)"
                            )

                    except Exception:
                        logging.error("Error training TPOT model", exc_info=True)
                else:
                    logging.warning("Not enough TPOT samples, skipping TPOT training.")

            with self.lock:
                if new_ttft_model:
                    self.ttft_model = new_ttft_model

                if new_tpot_model:
                    self.tpot_model = new_tpot_model

                if self.is_ready:
                    self.last_retrain_time = datetime.now(UTC)
                    try:
                        # FIT PHASE: Save models to staging for Build container
                        self._save_models_to_staging(bundle, new_ttft_model, new_tpot_model, ttft_snap, tpot_snap)
                    except Exception:
                        logging.error("Error saving models to staging after training.", exc_info=True)
        except Exception as e:
            logging.error(f"Critical error in train(): {e}", exc_info=True)

    def predict(self, features: dict) -> tuple[float, float, float, float]:
        try:
            with self.lock:
                if not self.is_ready:
                    raise HTTPException(status_code=503, detail="Models not ready")
                required = [
                    "kv_cache_percentage",
                    "input_token_length",
                    "num_request_waiting",
                    "num_request_running",
                    "num_tokens_generated",
                    "prefix_cache_score",
                ]
                for f in required:
                    if f not in features:
                        raise ValueError(f"Missing required feature: {f}")
                    if not isinstance(features[f], (int, float)):
                        raise ValueError(f"Invalid type for feature {f}: expected number")

                # Updated TTFT features to include prefix_cache_score and pod_type
                ttft_cols = [
                    "kv_cache_percentage",
                    "input_token_length",
                    "num_request_waiting",
                    "num_request_running",
                    "prefix_cache_score",
                ]
                tpot_cols = [
                    "kv_cache_percentage",
                    "input_token_length",
                    "num_request_waiting",
                    "num_request_running",
                    "num_tokens_generated",
                ]
                # Create DataFrames for predictions
                df_ttft = pd.DataFrame([{col: features[col] for col in ttft_cols}])
                # Add pod_type if present (otherwise _prepare_features_with_interaction will default to '')
                if "pod_type" in features:
                    df_ttft["pod_type"] = features["pod_type"]
                # Add interaction term for TTFT (includes pod_type encoding)
                df_ttft = self._prepare_features_with_interaction(df_ttft, model_type="ttft")

                df_tpot = pd.DataFrame([{col: features[col] for col in tpot_cols}])
                # Add pod_type if present
                if "pod_type" in features:
                    df_tpot["pod_type"] = features["pod_type"]
                # Add pod_type encoding for TPOT
                df_tpot = self._prepare_features_with_interaction(df_tpot, model_type="tpot")

                if self.model_type == ModelType.XGBOOST:
                    # XGBoost quantile regression directly predicts the quantile
                    ttft_pred = self.ttft_model.predict(df_ttft)
                    tpot_pred = self.tpot_model.predict(df_tpot)

                    # For XGBoost quantile regression, uncertainty estimation is more complex
                    ttft_std = ttft_pred[0] * 0.1  # 10% of prediction as uncertainty estimate
                    tpot_std = tpot_pred[0] * 0.1

                    return ttft_pred[0], tpot_pred[0], ttft_std, tpot_std

                else:  # LightGBM with quantile regression
                    # LightGBM quantile regression directly predicts the quantile
                    ttft_pred = self.ttft_model.predict(df_ttft)
                    tpot_pred = self.tpot_model.predict(df_tpot)

                    # For LightGBM quantile regression, use a similar uncertainty estimate as XGBoost
                    ttft_std = ttft_pred[0] * 0.1  # 10% of prediction as uncertainty estimate
                    tpot_std = tpot_pred[0] * 0.1

                    return ttft_pred[0], tpot_pred[0], ttft_std, tpot_std

        except ValueError as ve:
            logging.warning(f"Client error in predict(): {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except HTTPException:
            raise
        except Exception:
            logging.error("Error in predict():", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during prediction")

    def add_training_sample(self, sample: dict):
        try:
            required = [
                "kv_cache_percentage",
                "actual_ttft_ms",
                "actual_tpot_ms",
                "num_tokens_generated",
                "input_token_length",
                "num_request_waiting",
                "num_request_running",
                "prefix_cache_score",
            ]
            for field in required:
                if field not in sample or not isinstance(sample[field], (int, float)):
                    logging.warning(f"Invalid sample field: {field}")
                    return

            # Use hash-based deterministic split to ensure consistent train/test assignment
            # This ensures the same sample always goes to the same split
            sample_hash = hash(str(sorted(sample.items())))
            is_test = (sample_hash % 100) < (settings.TEST_TRAIN_RATIO * 100)

            # Create subsets based on conditions
            ttft_valid = sample["actual_ttft_ms"] > 0
            tpot_valid = sample["actual_tpot_ms"] > 0

            if is_test:
                # Add to test data only if the respective metric is valid
                if ttft_valid:
                    self.ttft_test_data.append(sample.copy())
                if tpot_valid:
                    self.tpot_test_data.append(sample.copy())
            else:
                # Add to training buckets only if the respective metric is valid
                bucket_key = self._get_bucket_key(sample)

                if ttft_valid:
                    self.ttft_data_buckets[bucket_key].append(sample)
                if tpot_valid:
                    self.tpot_data_buckets[bucket_key].append(sample)

        except Exception as e:
            logging.error(f"Error adding training sample: {e}", exc_info=True)

    def add_training_samples(self, samples: list):
        """Bulk-add multiple training samples in one go."""
        with self.lock:
            for sample in samples:
                try:
                    # reuse the single-sample logic
                    self.add_training_sample(sample)
                except Exception:
                    # log & continue on individual failures
                    logging.exception("Failed to add one sample in bulk ingestion")

    def _compute_file_hash(self, file_path) -> str:
        """Compute SHA256 hash of a file."""
        import hashlib

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _save_models_to_staging(self, bundle, ttft_model, tpot_model, ttft_snap, tpot_snap):
        """
        NEW FIT/BUILD SPLIT ARCHITECTURE:
        Save trained models to staging directory for Build container to process.

        Fit phase responsibilities:
        1. Train models
        2. Save as native format (XGB JSON, LGB txt)
        3. Compute conformal weights
        4. Write artifacts to /work/staging/<run-id>/
        5. Write READY marker

        Build container will:
        - Watch for READY
        - Load native models → TreeLite → compile .so
        - Create bundle and publish
        """
        from pathlib import Path

        # Create staging directory
        run_id = bundle.manifest.bundle_id
        staging_dir = Path("/work/staging") / run_id
        staging_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Creating staging artifacts for run_id={run_id[:8]} at {staging_dir}")

        artifacts = {}  # Track all artifacts for manifest

        # Save TTFT model in native format
        if ttft_model:
            ttft_filename = get_model_filename(self.model_type.value, "ttft")
            ttft_model_path = staging_dir / ttft_filename

            if self.model_type == ModelType.XGBOOST:
                # XGBoost: save booster as JSON
                booster = ttft_model.get_booster()
                booster.save_model(str(ttft_model_path))
                artifacts["ttft_model"] = {
                    "path": ttft_filename,
                    "format": "xgboost_json",
                    "hash": self._compute_file_hash(ttft_model_path),
                }
                logging.info(f"TTFT XGBoost model saved as JSON: {ttft_model_path}")

            elif self.model_type == ModelType.LIGHTGBM:
                # LightGBM: save booster as text
                ttft_model.booster_.save_model(str(ttft_model_path))
                artifacts["ttft_model"] = {
                    "path": ttft_filename,
                    "format": "lightgbm_txt",
                    "hash": self._compute_file_hash(ttft_model_path),
                }
                logging.info(f"TTFT LightGBM model saved as txt: {ttft_model_path}")

        # Save TPOT model in native format
        if tpot_model:
            tpot_filename = get_model_filename(self.model_type.value, "tpot")
            tpot_model_path = staging_dir / tpot_filename

            if self.model_type == ModelType.XGBOOST:
                booster = tpot_model.get_booster()
                booster.save_model(str(tpot_model_path))
                artifacts["tpot_model"] = {
                    "path": tpot_filename,
                    "format": "xgboost_json",
                    "hash": self._compute_file_hash(tpot_model_path),
                }
                logging.info(f"TPOT XGBoost model saved as JSON: {tpot_model_path}")

            elif self.model_type == ModelType.LIGHTGBM:
                tpot_model.booster_.save_model(str(tpot_model_path))
                artifacts["tpot_model"] = {
                    "path": tpot_filename,
                    "format": "lightgbm_txt",
                    "hash": self._compute_file_hash(tpot_model_path),
                }
                logging.info(f"TPOT LightGBM model saved as txt: {tpot_model_path}")

        # Create conformal predictors using test data (Fit phase responsibility)
        # Build phase will use these conformal weights with compiled .so files
        if ttft_model and self.ttft_test_data:
            test_df_ttft = pd.DataFrame(list(self.ttft_test_data))
            X_test_ttft = self._prepare_features_with_interaction(test_df_ttft, "ttft")
            y_test_ttft = test_df_ttft["actual_ttft_ms"]
            y_pred_mean_ttft = ttft_model.predict(X_test_ttft)

            ttft_conformal = ConformalQuantilePredictor(quantile=self.quantile)
            ttft_conformal.calibrate(y_pred_mean_ttft, y_test_ttft.values)

            # Save to staging
            ttft_conformal_path = staging_dir / TTFT_CONFORMAL_FILENAME
            with open(ttft_conformal_path, "w") as f:
                json.dump(ttft_conformal.get_state(), f, indent=2)

            artifacts["ttft_conformal"] = {
                "path": TTFT_CONFORMAL_FILENAME,
                "format": "conformal_json",
                "hash": self._compute_file_hash(ttft_conformal_path),
            }
            logging.info(f"Created TTFT conformal predictor (quantile={self.quantile})")

        # TPOT conformal
        if tpot_model and self.tpot_test_data:
            test_df_tpot = pd.DataFrame(list(self.tpot_test_data))
            X_test_tpot = self._prepare_features_with_interaction(test_df_tpot, "tpot")
            y_test_tpot = test_df_tpot["actual_tpot_ms"]
            y_pred_mean_tpot = tpot_model.predict(X_test_tpot)

            tpot_conformal = ConformalQuantilePredictor(quantile=self.quantile)
            tpot_conformal.calibrate(y_pred_mean_tpot, y_test_tpot.values)

            tpot_conformal_path = staging_dir / TPOT_CONFORMAL_FILENAME
            with open(tpot_conformal_path, "w") as f:
                json.dump(tpot_conformal.get_state(), f, indent=2)

            artifacts["tpot_conformal"] = {
                "path": TPOT_CONFORMAL_FILENAME,
                "format": "conformal_json",
                "hash": self._compute_file_hash(tpot_conformal_path),
            }
            logging.info(f"Created TPOT conformal predictor (quantile={self.quantile})")

        # Create feature schema (ensures training/inference consistency)
        feature_schema = {
            "ttft_features": [
                "kv_cache_percentage",
                "input_token_length",
                "num_request_waiting",
                "num_request_running",
                "prefix_cache_score",
                "effective_input_tokens",
                "prefill_score_bucket",
                "pod_type_cat",
            ],
            "tpot_features": [
                "kv_cache_percentage",
                "input_token_length",
                "num_request_waiting",
                "num_request_running",
                "num_tokens_generated",
                "pod_type_cat",
            ],
            "feature_dtypes": {
                "kv_cache_percentage": "float64",
                "input_token_length": "int64",
                "num_request_waiting": "int64",
                "num_request_running": "int64",
                "prefix_cache_score": "float64",
                "effective_input_tokens": "float64",
                "prefill_score_bucket": "int32",
                "pod_type_cat": "int32",
                "num_tokens_generated": "int64",
            },
        }

        feature_schema_path = staging_dir / "feature_schema.json"
        with open(feature_schema_path, "w") as f:
            json.dump(feature_schema, f, indent=2)

        artifacts["feature_schema"] = {
            "path": "feature_schema.json",
            "format": "json",
            "hash": self._compute_file_hash(feature_schema_path),
        }
        logging.info("Created feature schema")

        # Create categorical mapping (critical for TreeLite categorical correctness)
        # Use centralized encoder schema
        categorical_mapping = self.feature_encoder.to_dict()

        categorical_mapping_path = staging_dir / "categorical_mapping.json"
        with open(categorical_mapping_path, "w") as f:
            json.dump(categorical_mapping, f, indent=2)

        artifacts["categorical_mapping"] = {
            "path": "categorical_mapping.json",
            "format": "json",
            "hash": self._compute_file_hash(categorical_mapping_path),
        }
        logging.info("Created categorical mapping")

        # Collect training/test sample counts
        training_samples = {"ttft": len(ttft_snap) if ttft_snap else 0, "tpot": len(tpot_snap) if tpot_snap else 0}

        test_samples = {"ttft": len(self.ttft_test_data), "tpot": len(self.tpot_test_data)}

        # Create fit manifest
        fit_manifest = {
            "run_id": run_id,
            "bundle_id": bundle.manifest.bundle_id,
            "created_at": datetime.now(UTC).isoformat(),
            "model_type": self.model_type.value,
            "quantile": self.quantile,
            "training_samples": training_samples,
            "test_samples": test_samples,
            "artifacts": artifacts,
        }

        fit_manifest_path = staging_dir / "fit_manifest.json"
        with open(fit_manifest_path, "w") as f:
            json.dump(fit_manifest, f, indent=2)

        logging.info(f"Created fit manifest: {len(artifacts)} artifacts")

        # Write READY marker (signals Build container to start)
        ready_marker_path = staging_dir / "READY"
        with open(ready_marker_path, "w") as f:
            f.write(f"{datetime.now(UTC).isoformat()}\n")

        logging.info(
            f"✅ Fit phase complete for run_id={run_id[:8]} "
            f"({sum(training_samples.values())} training samples, "
            f"{sum(test_samples.values())} test samples). "
            f"Build container will now compile and publish bundle."
        )

    def flush_training_data(
        self, flush_training: bool = True, flush_test: bool = True, flush_metrics: bool = True, reason: str = None
    ) -> dict:
        """
        Manually flush training data, test data, and/or metrics.
        Returns statistics about what was flushed.

        Args:
            flush_training: Whether to flush training data buckets
            flush_test: Whether to flush test data
            flush_metrics: Whether to flush quantile metric scores
            reason: Optional reason for flushing (for logging)

        Returns:
            Dictionary with flush statistics
        """
        try:
            with self.lock:
                # Count samples before flushing (handles 3D buckets)
                ttft_training_count = sum(len(bucket) for bucket in self.ttft_data_buckets.values())
                tpot_training_count = sum(len(bucket) for bucket in self.tpot_data_buckets.values())
                ttft_test_count = len(self.ttft_test_data)
                tpot_test_count = len(self.tpot_test_data)

                reason_str = f" Reason: {reason}" if reason else ""
                logging.info(
                    f"Manual flush requested.{reason_str} "
                    f"Training: {flush_training}, Test: {flush_test}, Metrics: {flush_metrics}"
                )

                # Flush training data (now handles 3D buckets automatically)
                if flush_training:
                    for bucket_key in self.ttft_data_buckets:
                        self.ttft_data_buckets[bucket_key].clear()
                    for bucket_key in self.tpot_data_buckets:
                        self.tpot_data_buckets[bucket_key].clear()
                    logging.info(f"Flushed {ttft_training_count} TTFT and {tpot_training_count} TPOT training samples")

                # Flush test data
                if flush_test:
                    self.ttft_test_data.clear()
                    self.tpot_test_data.clear()
                    logging.info(f"Flushed {ttft_test_count} TTFT and {tpot_test_count} TPOT test samples")

                # Clear metrics
                metrics_cleared = False
                if flush_metrics:
                    self.ttft_quantile_loss_scores.clear()
                    self.tpot_quantile_loss_scores.clear()
                    self.ttft_coverage_scores.clear()
                    self.tpot_coverage_scores.clear()
                    self.ttft_violation_rates.clear()
                    self.tpot_violation_rates.clear()
                    metrics_cleared = True
                    logging.info("Cleared all quantile metric scores")

                return {
                    "success": True,
                    "ttft_training_samples_flushed": ttft_training_count if flush_training else 0,
                    "tpot_training_samples_flushed": tpot_training_count if flush_training else 0,
                    "ttft_test_samples_flushed": ttft_test_count if flush_test else 0,
                    "tpot_test_samples_flushed": tpot_test_count if flush_test else 0,
                    "metrics_cleared": metrics_cleared,
                }

        except Exception as e:
            logging.error(f"Error flushing data: {e}", exc_info=True)
            raise

    def load_models(self):
        """
        Initialize fresh models on startup.

        Training server uses bundle system - models are created fresh on startup
        and will be trained/published via bundles as training data accumulates.
        """
        try:
            with self.lock:
                # Always create fresh default models (bundle system handles persistence)
                logging.info("Initializing fresh models (bundle system)")
                self.ttft_model = self._create_default_model("ttft")
                self.tpot_model = self._create_default_model("tpot")

                # Use fresh sample threshold since we're starting with default models
                settings.MIN_SAMPLES_FOR_RETRAIN = settings.MIN_SAMPLES_FOR_RETRAIN_FRESH

                if not self.is_ready:
                    raise RuntimeError("Failed to initialize models")

                logging.info("✓ Models initialized successfully")
        except Exception as e:
            logging.error(f"Critical error in load_models: {e}", exc_info=True)
            raise

    def get_metrics(self) -> str:
        """Render Prometheus-style metrics: model, coefficients/importances, bucket counts, and quantile-specific scores."""
        try:
            # Snapshot models
            ttft_model, tpot_model = self.ttft_model, self.tpot_model

            lines: list[str] = []
            # 1) Model type and quantile info
            lines.append(f'model_type{{type="{self.model_type.value}"}} 1')
            lines.append(f"model_quantile{{}} {self.quantile}")

            # Helper: emit linear‐model coefs or tree importances
            def emit_metrics(model, coefficients, feats, prefix):
                if model is None:
                    # placeholders
                    lines.append(f"{prefix}_intercept{{}} 0.0")
                    kind = "importance"
                    for f in feats:
                        lines.append(f'{prefix}_{kind}{{feature="{f}"}} 0.0')
                    return

                # XGBoost/LightGBM importances
                try:
                    imps = model.feature_importances_
                except Exception:
                    imps = [0.0] * len(feats)
                lines.append(f"{prefix}_intercept{{}} 0.0")
                for f, imp in zip(feats, imps, strict=False):
                    lines.append(f'{prefix}_importance{{feature="{f}"}} {imp:.6f}')

            ttft_feats = [
                "kv_cache_percentage",
                "input_token_length",
                "num_request_waiting",
                "num_request_running",
                "prefix_cache_score",
                "effective_input_tokens",
                "prefill_score_bucket",
                "pod_type_cat",
            ]
            tpot_feats = [
                "kv_cache_percentage",
                "input_token_length",
                "num_request_waiting",
                "num_request_running",
                "num_tokens_generated",
                "pod_type_cat",
            ]
            emit_metrics(ttft_model, self.ttft_coefficients, ttft_feats, "ttft")
            emit_metrics(tpot_model, self.tpot_coefficients, tpot_feats, "tpot")

            # 3) Multi-dimensional bucket counts with 3D keys
            for (queue_bucket, cache_bucket, prefix_bucket), bucket_deque in self.ttft_data_buckets.items():
                count = len(bucket_deque)
                lines.append(
                    f'training_samples_count{{model="ttft",queue_bucket="{queue_bucket}",cache_bucket="{cache_bucket}",prefix_bucket="{prefix_bucket}"}} {count}'
                )

            for (queue_bucket, cache_bucket, prefix_bucket), bucket_deque in self.tpot_data_buckets.items():
                count = len(bucket_deque)
                lines.append(
                    f'training_samples_count{{model="tpot",queue_bucket="{queue_bucket}",cache_bucket="{cache_bucket}",prefix_bucket="{prefix_bucket}"}} {count}'
                )

            # Summary metrics by queue state
            for q in range(self.queue_buckets):
                ttft_total = sum(
                    len(self.ttft_data_buckets[(q, c, p)])
                    for c in range(self.cache_buckets)
                    for p in range(self.prefix_buckets)
                )
                tpot_total = sum(
                    len(self.tpot_data_buckets[(q, c, p)])
                    for c in range(self.cache_buckets)
                    for p in range(self.prefix_buckets)
                )
                lines.append(f'training_samples_queue_total{{model="ttft",queue_bucket="{q}"}} {ttft_total}')
                lines.append(f'training_samples_queue_total{{model="tpot",queue_bucket="{q}"}} {tpot_total}')

            # Summary metrics by cache state
            for c in range(self.cache_buckets):
                ttft_total = sum(
                    len(self.ttft_data_buckets[(q, c, p)])
                    for q in range(self.queue_buckets)
                    for p in range(self.prefix_buckets)
                )
                tpot_total = sum(
                    len(self.tpot_data_buckets[(q, c, p)])
                    for q in range(self.queue_buckets)
                    for p in range(self.prefix_buckets)
                )
                lines.append(f'training_samples_cache_total{{model="ttft",cache_bucket="{c}"}} {ttft_total}')
                lines.append(f'training_samples_cache_total{{model="tpot",cache_bucket="{c}"}} {tpot_total}')

            # Summary metrics by prefix state
            for p in range(self.prefix_buckets):
                ttft_total = sum(
                    len(self.ttft_data_buckets[(q, c, p)])
                    for q in range(self.queue_buckets)
                    for c in range(self.cache_buckets)
                )
                tpot_total = sum(
                    len(self.tpot_data_buckets[(q, c, p)])
                    for q in range(self.queue_buckets)
                    for c in range(self.cache_buckets)
                )

                # Calculate prefix range for this bucket
                prefix_low = p / self.prefix_buckets
                prefix_high = (p + 1) / self.prefix_buckets

                lines.append(
                    f'training_samples_prefix_total{{model="ttft",prefix_bucket="{p}",range="{prefix_low:.2f}-{prefix_high:.2f}"}} {ttft_total}'
                )
                lines.append(
                    f'training_samples_prefix_total{{model="tpot",prefix_bucket="{p}",range="{prefix_low:.2f}-{prefix_high:.2f}"}} {tpot_total}'
                )

            # Add prefix score distribution statistics
            all_ttft_samples = self._all_samples(self.ttft_data_buckets)
            if all_ttft_samples:
                prefix_scores = [s["prefix_cache_score"] for s in all_ttft_samples]
                ttfts = [s["actual_ttft_ms"] for s in all_ttft_samples]

                lines.append(f"prefix_score_mean{{}} {np.mean(prefix_scores):.4f}")
                lines.append(f"prefix_score_std{{}} {np.std(prefix_scores):.4f}")
                lines.append(f"prefix_score_min{{}} {np.min(prefix_scores):.4f}")
                lines.append(f"prefix_score_max{{}} {np.max(prefix_scores):.4f}")

                # Average TTFT by prefix bucket
                for p in range(self.prefix_buckets):
                    prefix_low = p / self.prefix_buckets
                    prefix_high = (p + 1) / self.prefix_buckets

                    if p == self.prefix_buckets - 1:
                        mask = [(prefix_low <= score <= prefix_high) for score in prefix_scores]  # include 1.0
                    else:
                        mask = [(prefix_low <= score < prefix_high) for score in prefix_scores]
                    bucket_ttfts = [t for t, m in zip(ttfts, mask, strict=False) if m]

                    if bucket_ttfts:
                        avg_ttft = np.mean(bucket_ttfts)
                        median_ttft = np.median(bucket_ttfts)
                        lines.append(
                            f'avg_ttft_by_prefix{{prefix_bucket="{p}",range="{prefix_low:.2f}-{prefix_high:.2f}"}} {avg_ttft:.2f}'
                        )
                        lines.append(
                            f'median_ttft_by_prefix{{prefix_bucket="{p}",range="{prefix_low:.2f}-{prefix_high:.2f}"}} {median_ttft:.2f}'
                        )

            # 4) Quantile Loss scores (last up to 5)
            for idx, score in enumerate(self.ttft_quantile_loss_scores):
                lines.append(f'ttft_quantile_loss{{idx="{idx}"}} {score:.6f}')
            for idx, score in enumerate(self.tpot_quantile_loss_scores):
                lines.append(f'tpot_quantile_loss{{idx="{idx}"}} {score:.6f}')

            # 5) Coverage scores (should be close to quantile * 100)
            for idx, coverage in enumerate(self.ttft_coverage_scores):
                lines.append(f'ttft_coverage_percent{{idx="{idx}"}} {coverage:.6f}')
            for idx, coverage in enumerate(self.tpot_coverage_scores):
                lines.append(f'tpot_coverage_percent{{idx="{idx}"}} {coverage:.6f}')

            # 6) Violation rates (should be close to (1-quantile) * 100)
            for idx, violation_rate in enumerate(self.ttft_violation_rates):
                lines.append(f'ttft_violation_rate_percent{{idx="{idx}"}} {violation_rate:.6f}')
            for idx, violation_rate in enumerate(self.tpot_violation_rates):
                lines.append(f'tpot_violation_rate_percent{{idx="{idx}"}} {violation_rate:.6f}')

            # 7) Target metrics for reference
            target_coverage = self.quantile * 100
            target_violation_rate = (1 - self.quantile) * 100
            lines.append(f"target_coverage_percent{{}} {target_coverage:.1f}")
            lines.append(f"target_violation_rate_percent{{}} {target_violation_rate:.1f}")

            return "\n".join(lines) + "\n"

        except Exception as e:
            logging.error(f"Error generating metrics: {e}", exc_info=True)
            return "# error_generating_metrics 1\n"


# --- FastAPI Application ---
app = FastAPI(
    title="Latency Predictor Service",
    description="A service to predict TTFT and TPOT using quantile regression with continuous training and feature scaling.",
)

predictor = LatencyPredictor()


# --- Pydantic Models for API ---
class TrainingEntry(BaseModel):
    kv_cache_percentage: float = Field(..., ge=0.0, le=1.0)
    input_token_length: int = Field(..., ge=0)
    num_request_waiting: int = Field(..., ge=0)
    num_request_running: int = Field(..., ge=0)
    actual_ttft_ms: float = Field(..., ge=0.0)
    actual_tpot_ms: float = Field(..., ge=0.0)
    num_tokens_generated: int = Field(..., ge=0)
    prefix_cache_score: float = Field(..., ge=0.0, le=1.0, description="Prefix cache hit ratio score (0.0 to 1.0)")
    pod_type: str | None = Field(default="", description="Pod type: 'prefill', 'decode', or '' for monolithic")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PredictionRequest(BaseModel):
    kv_cache_percentage: float = Field(..., ge=0.0, le=1.0)
    input_token_length: int = Field(..., ge=0)
    num_request_waiting: int = Field(..., ge=0)
    num_request_running: int = Field(..., ge=0)
    num_tokens_generated: int = Field(..., ge=0)
    prefix_cache_score: float = Field(..., ge=0.0, le=1.0, description="Prefix cache hit ratio score (0.0 to 1.0)")
    pod_type: str | None = Field(default="", description="Pod type: 'prefill', 'decode', or '' for monolithic")


class PredictionResponse(BaseModel):
    ttft_ms: float = Field(..., description=f"Predicted {settings.QUANTILE_ALPHA:.0%} quantile TTFT in milliseconds")
    tpot_ms: float = Field(..., description=f"Predicted {settings.QUANTILE_ALPHA:.0%} quantile TPOT in milliseconds")
    ttft_uncertainty: float = Field(..., description="Uncertainty estimate for TTFT prediction")
    tpot_uncertainty: float = Field(..., description="Uncertainty estimate for TPOT prediction")
    ttft_prediction_bounds: tuple[float, float] = Field(..., description="Approximate prediction bounds for TTFT")
    tpot_prediction_bounds: tuple[float, float] = Field(..., description="Approximate prediction bounds for TPOT")
    predicted_at: datetime
    model_type: ModelType = Field(default=predictor.model_type.value, description="Type of model used for prediction")
    quantile: float = Field(default=settings.QUANTILE_ALPHA, description="Quantile being predicted")


class BulkTrainingRequest(BaseModel):
    entries: list[TrainingEntry]


# --- Background Training Loop ---
def continuous_training_loop():
    time.sleep(10)
    while not predictor._shutdown_event.is_set():
        try:
            logging.debug("Checking if training should run...")
            predictor.train()
        except Exception:
            logging.error("Error in periodic retraining", exc_info=True)
        if predictor._shutdown_event.wait(timeout=settings.RETRAINING_INTERVAL_SEC):
            break
    logging.info("Training loop exiting.")


# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    logging.info("Server starting up...")
    predictor.load_models()
    t = threading.Thread(target=continuous_training_loop, daemon=True)
    predictor._training_thread = t
    t.start()
    logging.info("Background training started.")


@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Server shutting down...")
    predictor.shutdown()


@app.post("/add_training_data_bulk", status_code=status.HTTP_202_ACCEPTED)
async def add_training_data_bulk(batch: BulkTrainingRequest):
    """
    Accepts a JSON body like:
      { "entries": [ { …TrainingEntry… }, { … }, … ] }
    """
    try:
        predictor.add_training_samples([e.dict() for e in batch.entries])
        return {"message": f"Accepted {len(batch.entries)} training samples."}
    except Exception:
        logging.error("Failed to add bulk training data", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to add training data in bulk")


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    try:
        ttft_pred, tpot_pred, ttft_std, tpot_std = predictor.predict(request.dict())
        ttft_pred = max(0, ttft_pred)
        tpot_pred = max(0, tpot_pred)
        ttft_bounds = (max(0, ttft_pred - 2 * ttft_std), ttft_pred + 2 * ttft_std)
        tpot_bounds = (max(0, tpot_pred - 2 * tpot_std), tpot_pred + 2 * tpot_std)
        return PredictionResponse(
            ttft_ms=ttft_pred,
            tpot_ms=tpot_pred,
            ttft_uncertainty=ttft_std,
            tpot_uncertainty=tpot_std,
            ttft_prediction_bounds=ttft_bounds,
            tpot_prediction_bounds=tpot_bounds,
            predicted_at=datetime.now(UTC),
            model_type=predictor.model_type.value,
            quantile=predictor.quantile,
        )
    except HTTPException:
        raise
    except Exception:
        logging.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}


@app.get("/readyz", status_code=status.HTTP_200_OK)
async def readiness_check():
    if not predictor.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models are not ready.")
    return {"status": "ready"}


@app.get("/metrics", status_code=status.HTTP_200_OK)
async def metrics():
    """Prometheus metrics including coefficients/importances, bucket counts, and quantile-specific metrics."""
    try:
        content = predictor.get_metrics()
        return Response(content, media_type="text/plain; version=0.0.4")
    except Exception as e:
        logging.error(f"Error in metrics endpoint: {e}", exc_info=True)
        return Response("# Error generating metrics\n", media_type="text/plain; version=0.0.4")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Latency Predictor is running.",
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile,
        "description": f"Predicting {predictor.quantile:.0%} quantile for TTFT and TPOT latencies",
    }


@app.post("/flush", response_model=FlushResponse, status_code=status.HTTP_200_OK)
async def flush_data(request: FlushRequest = FlushRequest()):
    """
    Manually flush training data, test data, and/or metrics.

    Useful when:
    - Server workload has changed significantly
    - You want to start fresh with new data
    - Testing or debugging model behavior
    - Forcing a clean state after deployment

    Example requests:
    - Flush everything: POST /flush with empty body
    - Flush only training: POST /flush with {"flush_test_data": false, "flush_metrics": false}
    - Flush with reason: POST /flush with {"reason": "New deployment"}
    """
    try:
        result = predictor.flush_training_data(
            flush_training=request.flush_training_data,
            flush_test=request.flush_test_data,
            flush_metrics=request.flush_metrics,
            reason=request.reason,
        )

        total_flushed = (
            result["ttft_training_samples_flushed"]
            + result["tpot_training_samples_flushed"]
            + result["ttft_test_samples_flushed"]
            + result["tpot_test_samples_flushed"]
        )

        message_parts = []
        if request.flush_training_data:
            message_parts.append(
                f"{result['ttft_training_samples_flushed']} TTFT and "
                f"{result['tpot_training_samples_flushed']} TPOT training samples"
            )
        if request.flush_test_data:
            message_parts.append(
                f"{result['ttft_test_samples_flushed']} TTFT and "
                f"{result['tpot_test_samples_flushed']} TPOT test samples"
            )
        if request.flush_metrics:
            message_parts.append("all metric scores")

        message = f"Successfully flushed: {', '.join(message_parts)}" if message_parts else "No data flushed"

        return FlushResponse(
            success=True,
            flushed_at=datetime.now(UTC),
            reason=request.reason,
            ttft_training_samples_flushed=result["ttft_training_samples_flushed"],
            tpot_training_samples_flushed=result["tpot_training_samples_flushed"],
            ttft_test_samples_flushed=result["ttft_test_samples_flushed"],
            tpot_test_samples_flushed=result["tpot_test_samples_flushed"],
            metrics_cleared=result["metrics_cleared"],
            message=message,
        )

    except Exception as e:
        logging.error(f"Error in flush endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to flush data: {e!s}")


@app.get("/data/status", status_code=status.HTTP_200_OK)
async def get_data_status():
    """
    Get current status of training data.
    Useful for monitoring and deciding whether to flush.
    """
    ttft_training_count = sum(len(bucket) for bucket in predictor.ttft_data_buckets.values())
    tpot_training_count = sum(len(bucket) for bucket in predictor.tpot_data_buckets.values())

    bucket_distribution = {}
    for (q, c, p), bucket in predictor.ttft_data_buckets.items():
        if len(bucket) > 0:
            key = f"queue_{q}_cache_{c}_prefix_{p}"
            bucket_distribution[key] = len(bucket)

    return {
        "training_data": {
            "ttft_samples": ttft_training_count,
            "tpot_samples": tpot_training_count,
            "total_samples": ttft_training_count + tpot_training_count,
        },
        "test_data": {
            "ttft_samples": len(predictor.ttft_test_data),
            "tpot_samples": len(predictor.tpot_test_data),
            "total_samples": len(predictor.ttft_test_data) + len(predictor.tpot_test_data),
        },
        "metrics": {
            "ttft_scores_count": len(predictor.ttft_quantile_loss_scores),
            "tpot_scores_count": len(predictor.tpot_quantile_loss_scores),
        },
        "bucket_distribution": bucket_distribution,
        "model_ready": predictor.is_ready,
        "last_retrain": predictor.last_retrain_time.isoformat() if predictor.last_retrain_time else None,
    }


@app.get("/model/download/info")
async def model_download_info():
    """
    Get information about available model downloads and coefficients.
    """
    info = {"model_type": predictor.model_type.value, "quantile": predictor.quantile, "available_endpoints": {}}

    if predictor.model_type == ModelType.XGBOOST:
        info["available_endpoints"]["trees"] = {
            "ttft_trees": "/model/ttft/xgb/json",
            "tpot_trees": "/model/tpot/xgb/json",
        }
    else:
        info["available_endpoints"]["lightgbm"] = {
            "ttft_model_txt": "/model/ttft/lgb/txt",
            "tpot_model_txt": "/model/tpot/lgb/txt",
            "ttft_importances": "/model/ttft/lgb/importances",
            "tpot_importances": "/model/tpot/lgb/importances",
        }

    info["model_status"] = {
        "ttft_model_ready": predictor.ttft_model is not None,
        "tpot_model_ready": predictor.tpot_model is not None,
    }

    # Add quantile-specific evaluation info
    info["evaluation_info"] = {
        "quantile_loss": "Pinball loss for quantile regression evaluation",
        "coverage_percent": f"Percentage of actual values below predicted {predictor.quantile:.0%} quantile (target: {predictor.quantile*100:.1f}%)",
        "violation_rate_percent": f"Percentage of actual values above predicted {predictor.quantile:.0%} quantile (target: {(1-predictor.quantile)*100:.1f}%)",
    }

    return info


@app.get("/model/ttft/xgb/json")
async def ttft_xgb_json():
    """
    Dump the TTFT XGBoost model as JSON trees.
    """
    if predictor.model_type != ModelType.XGBOOST:
        raise HTTPException(status_code=404, detail="TTFT model is not XGBoost")

    if not predictor.ttft_model:
        raise HTTPException(status_code=404, detail="TTFT model not available")

    try:
        booster = predictor.ttft_model.get_booster()
        # get_dump with dump_format="json" gives one JSON string per tree
        raw_trees = booster.get_dump(dump_format="json")
        # parse each string into a dict so the response is a JSON array of objects
        trees = [json.loads(t) for t in raw_trees]
        return JSONResponse(content=trees)
    except Exception as e:
        logging.error(f"Error dumping TTFT XGBoost trees: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error dumping TTFT XGBoost trees")


@app.get("/model/tpot/xgb/json")
async def tpot_xgb_json():
    """
    Dump the TPOT XGBoost model as JSON trees.
    """
    if predictor.model_type != ModelType.XGBOOST:
        raise HTTPException(status_code=404, detail="TPOT model is not XGBoost")

    if not predictor.tpot_model:
        raise HTTPException(status_code=404, detail="TPOT model not available")

    try:
        booster = predictor.tpot_model.get_booster()
        raw_trees = booster.get_dump(dump_format="json")
        trees = [json.loads(t) for t in raw_trees]
        return JSONResponse(content=trees)
    except Exception as e:
        logging.error(f"Error dumping TPOT XGBoost trees: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error dumping TPOT XGBoost trees")


# REMOVED: Legacy file-based model endpoints (replaced by bundle system)
# - /model/{model_name}/info → Use /bundle/current/info
# - /model/{model_name}/download → Use /bundle/{bundle_id}/file/{filename}
# - /model/{ttft|tpot}/lgb/txt → Models available in bundle as native format
# - /model/{ttft|tpot}/lgb/importances → Feature importances can be computed from loaded models


@app.get("/models/list")
async def models_list():
    """
    List available models in the current bundle.

    This endpoint provides backward compatibility with tests that expect
    the old /models/list format. It maps bundle files to model entries.

    Returns:
        Dictionary with models, model_type, and server_time
    """
    bundle = predictor.bundle_manager.get_active_bundle()

    if not bundle:
        # No bundle available yet
        return {
            "models": {},
            "model_type": predictor.model_type.value,
            "server_time": datetime.now(UTC).isoformat(),
        }

    # Map bundle files to model entries
    models = {}

    # TreeLite models (always present in TreeLite mode)
    if "ttft_treelite.so" in bundle.manifest.files:
        file_info = bundle.manifest.files["ttft_treelite.so"]
        models["ttft_treelite"] = {
            "exists": True,
            "size_bytes": file_info.get("size_bytes", 0),
        }
        # For backward compatibility, also report as "ttft" model
        models["ttft"] = {
            "exists": True,
            "size_bytes": file_info.get("size_bytes", 0),
        }
    else:
        models["ttft_treelite"] = {"exists": False, "size_bytes": 0}
        models["ttft"] = {"exists": False, "size_bytes": 0}

    if "tpot_treelite.so" in bundle.manifest.files:
        file_info = bundle.manifest.files["tpot_treelite.so"]
        models["tpot_treelite"] = {
            "exists": True,
            "size_bytes": file_info.get("size_bytes", 0),
        }
        # For backward compatibility, also report as "tpot" model
        models["tpot"] = {
            "exists": True,
            "size_bytes": file_info.get("size_bytes", 0),
        }
    else:
        models["tpot_treelite"] = {"exists": False, "size_bytes": 0}
        models["tpot"] = {"exists": False, "size_bytes": 0}

    # Conformal calibration files (.json, not .pkl)
    if "ttft_conformal.json" in bundle.manifest.files:
        file_info = bundle.manifest.files["ttft_conformal.json"]
        models["ttft_conformal"] = {
            "exists": True,
            "size_bytes": file_info.get("size_bytes", 0),
        }
    else:
        models["ttft_conformal"] = {"exists": False, "size_bytes": 0}

    if "tpot_conformal.json" in bundle.manifest.files:
        file_info = bundle.manifest.files["tpot_conformal.json"]
        models["tpot_conformal"] = {
            "exists": True,
            "size_bytes": file_info.get("size_bytes", 0),
        }
    else:
        models["tpot_conformal"] = {"exists": False, "size_bytes": 0}

    return {
        "models": models,
        "model_type": bundle.manifest.model_type,
        "server_time": datetime.now(UTC).isoformat(),
    }


@app.get("/debug/prefix_distribution")
async def prefix_distribution():
    """
    Debug endpoint to analyze the relationship between prefix_cache_score and TTFT.
    This helps verify that the model is seeing the data it needs to learn.
    """
    all_samples = predictor._all_samples(predictor.ttft_data_buckets)
    if not all_samples:
        return {"error": "No training samples available"}

    prefix_scores = [s["prefix_cache_score"] for s in all_samples]
    ttfts = [s["actual_ttft_ms"] for s in all_samples]

    # Group by prefix score ranges
    ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    distribution = {}

    for low, high in ranges:
        # include the right edge only for the final bin so 1.0 is counted
        if high == 1.0:
            mask = [(low <= p <= high) for p in prefix_scores]
        else:
            mask = [(low <= p < high) for p in prefix_scores]

        range_ttfts = [t for t, m in zip(ttfts, mask, strict=False) if m]
        range_prefix = [p for p, m in zip(prefix_scores, mask, strict=False) if m]
        distribution[f"{low}-{high}"] = {
            "count": len(range_ttfts),
            "mean_ttft_ms": float(np.mean(range_ttfts)) if range_ttfts else 0,
            "median_ttft_ms": float(np.median(range_ttfts)) if range_ttfts else 0,
            "std_ttft_ms": float(np.std(range_ttfts)) if range_ttfts else 0,
            "mean_prefix_score": float(np.mean(range_prefix)) if range_prefix else 0,
            "min_ttft_ms": float(np.min(range_ttfts)) if range_ttfts else 0,
            "max_ttft_ms": float(np.max(range_ttfts)) if range_ttfts else 0,
        }

    # Overall statistics
    overall = {
        "total_samples": len(all_samples),
        "prefix_score_mean": float(np.mean(prefix_scores)),
        "prefix_score_std": float(np.std(prefix_scores)),
        "prefix_score_min": float(np.min(prefix_scores)),
        "prefix_score_max": float(np.max(prefix_scores)),
        "ttft_mean": float(np.mean(ttfts)),
        "ttft_std": float(np.std(ttfts)),
        "correlation": float(np.corrcoef(prefix_scores, ttfts)[0, 1]),
    }

    return {
        "overall_stats": overall,
        "distribution_by_prefix_range": distribution,
        "interpretation": {
            "correlation": "Negative correlation means higher prefix score → lower TTFT (good!)",
            "check_distribution": "All ranges should have samples. Empty ranges mean missing data.",
            "expected_pattern": "TTFT should decrease significantly as prefix score increases",
        },
    }


# === Bundle System HTTP Endpoints ===


@app.get("/bundle/current/info")
async def get_current_bundle_info():
    """
    Get information about the current active bundle.

    Returns bundle metadata including bundle_id, files with checksums/sizes,
    training/test sample counts, and state.

    This endpoint is used by prediction servers to discover new models.

    Response Statuses:
    - status: "no_bundles_available" - Bootstrap case (no training completed yet)
    - status: "bundle_available" - Bundle exists and is ready
    """
    bundle = predictor.bundle_manager.get_active_bundle()

    if not bundle:
        # Bootstrap case: No bundles published yet (expected during startup)
        return {
            "status": "no_bundles_available",
            "message": "No bundles have been published yet. Waiting for first training cycle to complete.",
            "min_samples_required": settings.MIN_SAMPLES_FOR_RETRAIN,
            "bundle_id": None,
            "files": {},
            "training_samples": {},
            "test_samples": {},
        }

    return {
        "status": "bundle_available",
        "bundle_id": bundle.manifest.bundle_id,
        "files": bundle.manifest.files,
        "training_samples": bundle.manifest.training_samples,
        "test_samples": bundle.manifest.test_samples,
        "state": bundle.manifest.state.value,
        "created_at": bundle.manifest.created_at,  # Already ISO format string
        "model_type": bundle.manifest.model_type,
        "quantile": bundle.manifest.quantile,
    }


@app.get("/bundle/{bundle_id}/file/{file_name}")
async def download_bundle_file(bundle_id: str, file_name: str):
    """
    Download a specific file from a bundle.

    Args:
        bundle_id: Bundle ID (can be short form like first 8 chars)
        file_name: Name of file to download

    Returns:
        File content as streaming response
    """
    # Find bundle by ID (support short form)
    # list_bundles() returns bundle ID strings, not ModelBundle objects
    bundle_ids = predictor.bundle_manager.list_bundles()
    matched_bundle_id = None

    for bid in bundle_ids:
        if bid.startswith(bundle_id):
            matched_bundle_id = bid
            break

    if not matched_bundle_id:
        raise HTTPException(status_code=404, detail=f"Bundle {bundle_id} not found")

    # Load the bundle using ModelBundle.load() static method
    from .model_bundle import ModelBundle

    try:
        bundle = ModelBundle.load(str(predictor.bundle_manager.registry.bundle_dir), matched_bundle_id)
    except Exception as e:
        logging.error(f"Error loading bundle {matched_bundle_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Bundle {bundle_id} not found or failed to load")

    # Check if file exists in bundle
    if file_name not in bundle.manifest.files:
        raise HTTPException(status_code=404, detail=f"File {file_name} not found in bundle {bundle_id}")

    # Build file path
    file_path = bundle.path / file_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {file_name} exists in manifest but not on disk")

    # Return file as streaming response
    from fastapi.responses import FileResponse

    return FileResponse(path=str(file_path), media_type="application/octet-stream", filename=file_name)


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)
