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
import os
import random
import time
import logging
import threading
import hashlib
from io import BytesIO
from datetime import datetime, timezone
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

from fastapi.responses import Response 
from fastapi.responses import JSONResponse, FileResponse

import joblib
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

import tempfile
import shutil
import subprocess
import sys

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

try:
    import treelite
    import treelite.sklearn
    TREELITE_AVAILABLE = True
except ImportError:
    TREELITE_AVAILABLE = False
    logging.warning("TreeLite not available. Please install with: pip install treelite treelite_runtime")

try:
    import tl2cgen
    TL2CGEN_AVAILABLE = True
except ImportError:
    TL2CGEN_AVAILABLE = False
    logging.warning("TL2cgen not available. Please install with: pip install tl2cgen")

try:
    from common.conformal_quantile import ConformalQuantilePredictor
    CONFORMAL_AVAILABLE = True
except ImportError:
    CONFORMAL_AVAILABLE = False
    logging.warning("ConformalQuantilePredictor not available. Check common/conformal_quantile.py exists.")

# Import bundle constants for consistent file naming
from common.bundle_constants import (
    TTFT_MODEL_FILENAME,
    TPOT_MODEL_FILENAME,
    TTFT_SCALER_FILENAME,
    TPOT_SCALER_FILENAME,
    TTFT_TREELITE_FILENAME,
    TPOT_TREELITE_FILENAME,
    TTFT_CONFORMAL_FILENAME,
    TPOT_CONFORMAL_FILENAME
)

try:
    from .model_bundle import ModelBundle, BundleRegistry, BundleState
    from .bundle_integration import BundleModelManager
    BUNDLE_REGISTRY_AVAILABLE = True
except ImportError as e:
    BUNDLE_REGISTRY_AVAILABLE = False
    logging.warning(f"ModelBundle not available: {e}")

# --- Configuration ---
class Settings:
    """
    Configuration class for the latency predictor server.
    Reads settings from environment variables with sensible defaults.
    """
    # Training configuration
    RETRAINING_INTERVAL_SEC: int = int(os.getenv("LATENCY_RETRAINING_INTERVAL_SEC", 1800))
    MIN_SAMPLES_FOR_RETRAIN_FRESH: int = int(os.getenv("LATENCY_MIN_SAMPLES_FOR_RETRAIN_FRESH", 10))
    MIN_SAMPLES_FOR_RETRAIN: int = int(os.getenv("LATENCY_MIN_SAMPLES_FOR_RETRAIN", 1000))
    MAX_TRAINING_DATA_SIZE_PER_BUCKET: int = int(os.getenv("LATENCY_MAX_TRAINING_DATA_SIZE_PER_BUCKET", 10000))
    TEST_TRAIN_RATIO: float = float(os.getenv("LATENCY_TEST_TRAIN_RATIO", "0.1"))  # Default 1:10 (10% test, 90% train)
    MAX_TEST_DATA_SIZE: int = int(os.getenv("LATENCY_MAX_TEST_DATA_SIZE", "1000"))  # Max test samples to keep
    MODEL_TYPE: str = os.getenv("LATENCY_MODEL_TYPE", "xgboost")  # Default to XGBoost
    QUANTILE_ALPHA: float = float(os.getenv("LATENCY_QUANTILE_ALPHA", "0.9"))  # p90 quantile
    SAMPLE_WEIGHTING_FOR_PREFIX_CACHE: bool = os.getenv("LATENCY_SAMPLE_WEIGHTING_FOR_PREFIX_CACHE", "false").lower() == "true"

    # TreeLite mode: if true, use standard regression + conformal prediction for quantiles
    # if false, use native quantile regression (more accurate but slower)
    USE_TREELITE: bool = os.getenv("USE_TREELITE", "true").lower() == "true"

    # Bundle registry configuration (persistent storage for trained models)
    # Training server creates bundles here - must be persistent volume to survive restarts!
    BUNDLE_DIR: str = os.getenv("BUNDLE_DIR", "/models/bundles")
    BUNDLE_CURRENT_SYMLINK: str = os.getenv("BUNDLE_CURRENT_SYMLINK", "/models/current")
    MAX_BUNDLES_TO_KEEP: int = int(os.getenv("MAX_BUNDLES_TO_KEEP", "5"))

settings = Settings()

# Validate dependencies at startup (fail-fast)
if settings.USE_TREELITE and not CONFORMAL_AVAILABLE:
    raise RuntimeError(
        "USE_TREELITE=true but conformal_quantile library not available. "
        "Install with: pip install -e . (or ensure conformal_quantile.py exists)"
    )


class ModelType(str, Enum):
    BAYESIAN_RIDGE = "bayesian_ridge"
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


# Configure logging level from environment variable (default: INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

# Add this to your Pydantic models section
class ModelInfoResponse(BaseModel):
    model_type: str
    xgboost_available: bool
    lightgbm_available: bool = Field(default=False, description="Whether LightGBM is available")  # FIXED: Added this field

    is_ready: bool
    ttft_training_samples: int = Field(default=0, description="Number of TTFT training samples")
    tpot_training_samples: int = Field(default=0, description="Number of TPOT training samples") 
    ttft_test_samples: int = Field(default=0, description="Number of TTFT test samples")
    tpot_test_samples: int = Field(default=0, description="Number of TPOT test samples")
    last_retrain_time: Optional[datetime] = Field(default=None, description="Last retraining timestamp")
    min_samples_for_retrain: int = Field(default=0, description="Minimum samples required for retraining")
    retraining_interval_sec: int = Field(default=0, description="Retraining interval in seconds")
    

class FlushRequest(BaseModel):
    flush_training_data: bool = Field(default=True, description="Flush training data buckets")
    flush_test_data: bool = Field(default=True, description="Flush test data")
    flush_metrics: bool = Field(default=True, description="Flush quantile metric scores")
    reason: Optional[str] = Field(default=None, description="Optional reason for flushing")

class FlushResponse(BaseModel):
    success: bool
    flushed_at: datetime
    reason: Optional[str] = None
    ttft_training_samples_flushed: int
    tpot_training_samples_flushed: int
    ttft_test_samples_flushed: int
    tpot_test_samples_flushed: int
    metrics_cleared: bool
    conformal_cleared: bool = Field(default=False, description="Whether conformal calibration was cleared")
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
        quantile: the quantile being predicted (e.g., 0.9 for p90)
    
    Returns:
        Mean quantile loss
    """
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)


def quantile_coverage(y_true, y_pred, quantile):
    """
    Calculate quantile coverage - the proportion of actual values that fall below the predicted quantile.
    
    For a well-calibrated p90 model, this should be close to 0.9 (90%).
    
    Args:
        y_true: actual values
        y_pred: predicted quantile values
        quantile: the quantile being predicted (e.g., 0.9 for p90)
    
    Returns:
        Coverage percentage (0-100)
    """
    below_prediction = np.sum(y_true <= y_pred)
    coverage = below_prediction / len(y_true)
    return coverage * 100


def quantile_violation_rate(y_true, y_pred, quantile):
    """
    Calculate quantile violation rate - the proportion of times actual values exceed the predicted quantile.
    
    For a well-calibrated p90 model, this should be close to 0.1 (10%).
    
    Args:
        y_true: actual values
        y_pred: predicted quantile values
        quantile: the quantile being predicted (e.g., 0.9 for p90)
    
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
    
        if model_type not in [e.value for e in ModelType]:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {[e.value for e in ModelType]}")

        if model_type == ModelType.XGBOOST.value and not XGBOOST_AVAILABLE:
            logging.warning("XGBoost requested but not available. Falling back to Bayesian Ridge.")
            model_type = ModelType.BAYESIAN_RIDGE.value

        if model_type == ModelType.LIGHTGBM.value and not LIGHTGBM_AVAILABLE:
            logging.warning("LightGBM requested but not available. Falling back to Bayesian Ridge.")
            model_type = ModelType.BAYESIAN_RIDGE.value

        self.model_type = ModelType(model_type)
        self.quantile = settings.QUANTILE_ALPHA
        logging.info(f"Initialized LatencyPredictor with model type: {self.model_type}, quantile: {self.quantile}, use_treelite: {settings.USE_TREELITE}")

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
        self.ttft_scaler = None
        self.tpot_scaler = None

        # Conformal predictors (only used in TreeLite mode)
        self.ttft_conformal = None
        self.tpot_conformal = None

        self.ttft_coefficients = None  # Will store descaled coefficients as dict
        self.tpot_coefficients = None  # Will store descaled coefficients as dict

        # Model hashes for TreeLite compilation caching
        self.ttft_model_hash = None
        self.tpot_model_hash = None

        # Conformal calibration hashes (to avoid rewriting unchanged files)
        self.ttft_conformal_hash = None
        self.tpot_conformal_hash = None

        # Cache hit tracking for observability
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_cache_report_time = time.time()

        # Track ongoing TreeLite compilations to prevent races
        self.ttft_compilation_in_progress = False
        self.tpot_compilation_in_progress = False
        self.compilation_lock = threading.Lock()

        self.lock = threading.Lock()
        self.last_retrain_time = None
        self._shutdown_event = threading.Event()
        self._training_thread: threading.Thread = None

        # Bundle registry (always enabled - modern model distribution)
        self.bundle_registry = None
        self.current_bundle = None
        self.bundle_manager = None
        if BUNDLE_REGISTRY_AVAILABLE:
            self.bundle_manager = BundleModelManager(
                bundle_dir=settings.BUNDLE_DIR,
                current_symlink=settings.BUNDLE_CURRENT_SYMLINK,
                max_bundles=settings.MAX_BUNDLES_TO_KEEP
            )
            # Keep bundle_registry for backward compatibility
            self.bundle_registry = self.bundle_manager.registry
            logging.info(f"Bundle manager initialized at {settings.BUNDLE_DIR}")
        else:
            raise RuntimeError("model_bundle.py not available - bundle system is required")
        
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
        queue_bucket = self._get_queue_bucket(sample['num_request_waiting'])
        cache_bucket = self._get_cache_bucket(sample['kv_cache_percentage'])
        prefix_bucket = self._get_prefix_bucket(sample['prefix_cache_score'])  # NEW

        return (queue_bucket, cache_bucket, prefix_bucket)

    def _store_descaled_coefficients(self, model, scaler, feature_names, model_name):
        """
        Store descaled coefficients for Bayesian Ridge models.
        Returns a dict with feature names as keys and coefficients as values.
        """
        if self.model_type != ModelType.BAYESIAN_RIDGE or model is None or scaler is None:
            return None
            
        try:
            # Get scaled coefficients and scaler parameters
            coef_scaled = model.coef_
            scale, mean = scaler.scale_, scaler.mean_
            
            # Descale coefficients: w_original = w_scaled / scale
            w_orig = coef_scaled / scale
            
            # Calculate descaled intercept: b_orig = b_scaled - sum(w_scaled * mean / scale)
            intercept = float(model.intercept_) - float(np.dot(coef_scaled, mean / scale))
            
            # Create coefficient dictionary
            coefficients = {"intercept": intercept}
            for feature, coef in zip(feature_names, w_orig):
                coefficients[feature] = float(coef)
                
            logging.info(f"Stored descaled coefficients for {model_name}: {coefficients}")
            return coefficients
            
        except Exception as e:
            logging.error(f"Error storing descaled coefficients for {model_name}: {e}")
            return None
        
    def _prepare_features_with_interaction(self, df: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """
        Prepare features with interaction terms for better model learning.
    
        Args:
            df: DataFrame with raw features
            model_type: 'ttft' or 'tpot'
    
        Returns:
            DataFrame with engineered features including interactions
        """
        if model_type == "ttft":
            # Create interaction: prefix score * input length
            # This captures that prefix caching benefit scales with input size
            df['effective_input_tokens'] = (1-df['prefix_cache_score']) * (df['input_token_length'])
            df['prefill_score_bucket'] = (
            (df['prefix_cache_score'].clip(0, 1) * self.prefix_buckets)
            .astype(int)
            .clip(upper=self.prefix_buckets - 1)
        )

            # TreeLite compatibility: Only use categorical dtype when NOT in TreeLite mode
            # TreeLite cannot parse XGBoost models with categorical features
            if not settings.USE_TREELITE:
                # Make it categorical for tree models (safe for LGB, XGB with enable_categorical)
                df['prefill_score_bucket'] = pd.Categorical(df['prefill_score_bucket'], categories=[0,1,2,3], ordered=True)
            else:
                # TreeLite mode: keep as int32 to avoid categorical encoding in XGBoost model JSON
                df['prefill_score_bucket'] = df['prefill_score_bucket'].astype('int32')


            # Return TTFT features with interaction
            feature_cols = [
            'kv_cache_percentage',
            'input_token_length',
            'num_request_waiting',
            'num_request_running',
            'prefix_cache_score',
            'effective_input_tokens',
            'prefill_score_bucket'
            ]
        
            return df[feature_cols]
        
        else:  # tpot
            # TPOT doesn't use prefix_cache_score, so no interaction needed
            feature_cols = [
                'kv_cache_percentage',
                'input_token_length',
                'num_request_waiting',
                'num_request_running',
                'num_tokens_generated'
            ]
        
            return df[feature_cols]


    def shutdown(self):
        """Signal the training thread to exit and join it."""
        self._shutdown_event.set()
        if self._training_thread is not None:
            self._training_thread.join()

    @property
    def is_ready(self) -> bool:
        """Checks if all models and scalers are loaded/trained."""
        if self.model_type == ModelType.BAYESIAN_RIDGE:
            return all([self.ttft_model, self.tpot_model, self.ttft_scaler, self.tpot_scaler])
        elif self.model_type in (ModelType.XGBOOST, ModelType.LIGHTGBM):
            return all([self.ttft_model, self.tpot_model])
        else:
            # TREELITE is not a valid training model type
            raise ValueError(f"Invalid model_type: {self.model_type}. Use XGBOOST, LIGHTGBM, or BAYESIAN_RIDGE.")

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
    


    def _train_model_with_scaling(self, features: pd.DataFrame, target: pd.Series, model_name: str = None, sample_weight: np.ndarray = None) -> Union[Tuple[BayesianRidge, StandardScaler], xgb.XGBRegressor, lgb.LGBMRegressor]:
        """
        Train a model with the appropriate scaling/preprocessing.

        TreeLite Compatibility Notes:
        - TreeLite does NOT support XGBoost's categorical encoder (enable_categorical=True)
        - When USE_TREELITE=true, we disable enable_categorical and convert categorical
          features to integers manually before training
        - This allows the trained XGBoost model to be compiled to TreeLite .so format
        - The prediction server then uses the compiled TreeLite model for faster inference
        """
        try:
            if len(features) == 0 or len(target) == 0:
                raise ValueError("Empty training data")
            if features.isnull().any().any() or target.isnull().any():
                raise ValueError("Training data contains NaN values")
            if np.isinf(features.values).any() or np.isinf(target.values).any():
                raise ValueError("Training data contains infinite values")

            if self.model_type == ModelType.BAYESIAN_RIDGE:
                features = features.drop(columns=['prefill_score_bucket'], errors='ignore')
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
                    raise ValueError("Scaling produced invalid values")

                # For Bayesian Ridge, we'll approximate quantile regression by training on the mean
                # but adjusting predictions later. This is not ideal but Bayesian Ridge doesn't
                # natively support quantile regression.
                model = BayesianRidge(compute_score=True)
                model.fit(features_scaled, target, sample_weight=sample_weight)
                return model, scaler

            elif self.model_type == ModelType.XGBOOST:
                if model_name == "ttft":
                     # enforce your TTFT feature order
                        ttft_order = [
                            "kv_cache_percentage", "input_token_length", "num_request_waiting",
                            "num_request_running", "prefix_cache_score", "effective_input_tokens", "prefill_score_bucket"
                        ]
                        if list(features.columns) != ttft_order:
                            try:
                                features = features[ttft_order]
                            except Exception:
                                raise ValueError(f"TTFT features must be exactly {ttft_order}; got {list(features.columns)}")

                        # ---- (A) Build a warm-start stump that must split on prefill_score_bucket ----
                        # Train on the SAME full feature set, but freeze all other features to constants,
                        # so the only useful split is the prefix bucket.
                        features_stump = features.copy()
                        for col in features_stump.columns:
                            if col != "prefill_score_bucket":
                                # keep dtype, set to a constant scalar
                                const_val = features_stump[col].iloc[0]
                                features_stump[col] = const_val

                        # ensure prefill bucket is int codes if it's categorical
                        if str(features_stump["prefill_score_bucket"].dtype) == "category":
                            features_stump["prefill_score_bucket"] = (
                                features_stump["prefill_score_bucket"].cat.codes.astype("int32")
                            )
                        else:
                            features_stump["prefill_score_bucket"] = features_stump["prefill_score_bucket"].astype("int32")

                        # Choose objective based on TreeLite mode
                        if settings.USE_TREELITE:
                            # Standard regression for TreeLite compatibility
                            # Quantile predictions via conformal prediction
                            objective = "reg:squarederror"
                            model_params = {}
                        else:
                            # Native quantile regression (more accurate)
                            objective = "reg:quantileerror"
                            model_params = {"quantile_alpha": self.quantile}

                        # TreeLite doesn't support XGBoost categorical encoder
                        # So we disable it in TreeLite mode (dtype already converted to int32 in _prepare_features_with_interaction)
                        enable_categorical = not settings.USE_TREELITE

                        # CRITICAL FIX: XGBoost 2.0+ ignores enable_categorical=False for hist tree_method
                        # It still infers categorical features based on cardinality
                        # We must use tree_method='auto' or 'approx' in TreeLite mode to prevent categorical inference
                        tree_method = 'approx' if settings.USE_TREELITE else 'hist'

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
                            objective=objective,
                            tree_method=tree_method,  # Use 'approx' in TreeLite mode to prevent categorical inference
                            n_jobs=-1,
                            random_state=42,
                            verbosity=1,
                            enable_categorical=enable_categorical,  # Disabled for TreeLite compatibility
                            **model_params
                            )
                        model.fit(features, target, sample_weight=sample_weight)
                        return model


                elif model_name == "tpot":
                    tpot_order = ["kv_cache_percentage","input_token_length","num_request_waiting","num_request_running","num_tokens_generated"]
                    if list(features.columns) != tpot_order:
                        try:
                            features = features[tpot_order]
                        except Exception as _:
                            raise ValueError(f"TPOT features must be exactly {tpot_order}; got {list(features.columns)}")
                    mono_str = "(1,1,1,1,1)"
                else:
                    mono_str = "(0,0,0,0,0)"  # default

                # Choose objective based on TreeLite mode
                if settings.USE_TREELITE:
                    # Standard regression for TreeLite compatibility
                    objective = "reg:squarederror"
                    model_params = {}
                else:
                    # Native quantile regression (more accurate)
                    objective = "reg:quantileerror"
                    model_params = {"quantile_alpha": self.quantile}

                # TreeLite doesn't support XGBoost categorical encoder
                # (TPOT doesn't use prefill_score_bucket, so no categorical conversion needed)
                enable_categorical = not settings.USE_TREELITE

                # Use 'approx' tree_method in TreeLite mode to prevent categorical inference
                tree_method = 'approx' if settings.USE_TREELITE else 'hist'

                model = xgb.XGBRegressor(
                n_estimators=200,            # Number of trees to build (moderate value for balanced accuracy and speed)
                max_depth=6,                 # Depth of trees; 6 is typically a sweet spot balancing bias/variance
                learning_rate=0.05,          # Smaller learning rate to achieve stable convergence
                subsample=0.8,               # Use 80% of data per tree (adds regularization & reduces overfitting)
                colsample_bytree=0.8,        # Use 80% of features per tree (improves generalization)

                # Key parameters for regression:
                min_child_weight=5,          # Low value allows fine-grained splits
                gamma=0.2,                  # Low gamma allows splits with small loss reduction
                #monotone_constraints=mono_str,  # Enforce monotonicity based on feature impact on latency
                # Regularization to prevent overfitting:
                reg_alpha=0.01,              # L1 regularization (Lasso) - encourages sparsity
                reg_lambda=0.1,              # L2 regularization (Ridge) - prevents large coefficients

                objective=objective,         # Conditional: quantile or standard regression

                # Performance optimization:
                tree_method=tree_method,     # Use 'approx' in TreeLite mode to prevent categorical inference
                n_jobs=-1,                   # Utilize all CPU cores for parallel training
                random_state=42,             # Ensures reproducible results
                verbosity=1,
                enable_categorical=enable_categorical,  # Disabled for TreeLite compatibility
                **model_params
    )
                model.fit(features, target, sample_weight=sample_weight)
                return model
            elif self.model_type == ModelType.LIGHTGBM:
                # Choose objective based on TreeLite mode
                if settings.USE_TREELITE:
                    # Standard regression for TreeLite compatibility
                    objective = "regression"
                    model_params = {}

                    # Convert categorical to int for TreeLite compatibility
                    if 'prefill_score_bucket' in features.columns:
                        features = features.copy()
                        if str(features['prefill_score_bucket'].dtype) == 'category':
                            features['prefill_score_bucket'] = features['prefill_score_bucket'].cat.codes.astype('int32')
                        else:
                            features['prefill_score_bucket'] = features['prefill_score_bucket'].astype('int32')
                else:
                    # Native quantile regression (more accurate)
                    objective = "quantile"
                    model_params = {"alpha": self.quantile}
                    logging.info(f"Training LightGBM with quantile objective (alpha={self.quantile:.2f})")

                model = lgb.LGBMRegressor(
                    n_estimators=200,           # Number of trees
                    max_depth=6,                # Maximum tree depth
                    learning_rate=0.05,         # Learning rate
                    subsample=0.8,              # Row sampling ratio
                    colsample_bytree=0.8,       # Column sampling ratio
                    min_child_samples=20,       # Minimum samples in leaf
                    reg_alpha=0.1,              # L1 regularization
                    reg_lambda=0.1,             # L2 regularization
                    objective=objective,        # Conditional: quantile or standard regression
                    n_jobs=-1,                  # Use all cores
                    random_state=42,            # Reproducibility
                    verbosity=-1,               # Suppress warnings
                    force_col_wise=True,        # Better for small datasets
                    **model_params
                )

                # TreeLite doesn't support categorical features (same as XGBoost with enable_categorical=False)
                # So we disable categorical_feature in TreeLite mode and treat prefill_score_bucket as int
                categorical_feature = None if settings.USE_TREELITE else (['prefill_score_bucket'] if model_name == "ttft" else None)
                model.fit(features, target, sample_weight=sample_weight, categorical_feature=categorical_feature)
                return model
                
        except Exception as e:
            logging.error(f"Error in _train_model_with_scaling: {e}", exc_info=True)
            raise
        
    def _calculate_quantile_metrics_on_test(self, model, scaler, test_data, model_name, target_col):
        """Calculate quantile-specific metrics on test data"""
        try:
            df_raw = pd.DataFrame(test_data).dropna()
            df_raw = df_raw[df_raw[target_col] > 0]

            if len(df_raw) < 2:
                return None, None, None

        # Apply feature engineering to create interaction terms and categorical features
            df_features = self._prepare_features_with_interaction(df_raw.copy(), model_type=model_name)

        # Get appropriate feature columns based on model type and name
            if model_name == "ttft":
                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    feature_cols = [
                        'kv_cache_percentage','input_token_length','num_request_waiting',
                        'num_request_running','prefix_cache_score','effective_input_tokens'
                    ]
                else:  # XGBoost or LightGBM
                    feature_cols = [
                        'kv_cache_percentage','input_token_length','num_request_waiting',
                        'num_request_running','prefix_cache_score','effective_input_tokens','prefill_score_bucket'
                    ]
            else:  # tpot
                feature_cols = ['kv_cache_percentage', 'input_token_length', 
                   'num_request_waiting', 'num_request_running', 'num_tokens_generated']

            X = df_features[feature_cols]

            # Defensive check: Ensure prefill_score_bucket is int32 in TreeLite mode
            # (should already be int32 from _prepare_features_with_interaction, but verify for safety)
            if settings.USE_TREELITE and 'prefill_score_bucket' in X.columns:
                if str(X['prefill_score_bucket'].dtype) == 'category':
                    # This shouldn't happen - log warning if it does
                    logging.warning("prefill_score_bucket is categorical in TreeLite mode - converting to int32")
                    X = X.copy()
                    X['prefill_score_bucket'] = X['prefill_score_bucket'].cat.codes.astype('int32')
                elif str(X['prefill_score_bucket'].dtype) != 'int32':
                    X = X.copy()
                    X['prefill_score_bucket'] = X['prefill_score_bucket'].astype('int32')

            if self.model_type == ModelType.BAYESIAN_RIDGE and scaler is not None:
                X = scaler.transform(X)

            y_true = df_raw[target_col].values
            y_pred = model.predict(X)

            # Apply appropriate quantile adjustment based on model type
            if self.model_type == ModelType.BAYESIAN_RIDGE:
                # For Bayesian Ridge: estimate quantile by adding factor to mean prediction
                std_factor = 1.28 if self.quantile == 0.9 else (2.0 if self.quantile == 0.95 else 0.674)
                _, y_std = model.predict(X, return_std=True)
                y_pred = y_pred + std_factor * y_std
            elif settings.USE_TREELITE and (self.model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]):
                # For TreeLite mode: model predicts MEAN, apply conformal adjustment if available
                if model_name == "ttft" and self.ttft_conformal:
                    # Apply conformal adjustment to convert mean → quantile
                    y_pred = np.array([self.ttft_conformal.conformalize(pred) for pred in y_pred])
                    logging.debug(f"Applied conformal adjustment to TTFT test predictions")
                elif model_name == "tpot" and self.tpot_conformal:
                    # Apply conformal adjustment to convert mean → quantile
                    y_pred = np.array([self.tpot_conformal.conformalize(pred) for pred in y_pred])
                    logging.debug(f"Applied conformal adjustment to TPOT test predictions")
                else:
                    # No conformal predictor available yet - use crude statistical adjustment
                    from scipy.stats import norm
                    z = norm.ppf(self.quantile)
                    sigma_estimate = 0.12
                    y_pred = y_pred * (1.0 + sigma_estimate * z)
                    logging.debug(f"Applied bootstrap quantile adjustment (conformal not yet calibrated)")
            # else: Native quantile regression (XGBoost/LightGBM) - predictions are already quantiles

            # Calculate quantile-specific metrics
            ql = quantile_loss(y_true, y_pred, self.quantile)
            coverage = quantile_coverage(y_true, y_pred, self.quantile)
            violation_rate = quantile_violation_rate(y_true, y_pred, self.quantile)
            
            return ql, coverage, violation_rate
            
        except Exception as e:
            logging.error(f"Error calculating quantile metrics: {e}", exc_info=True)
            return None, None, None

    def _create_default_model(self, model_type: str) -> Union[Tuple[BayesianRidge, StandardScaler], xgb.XGBRegressor, lgb.LGBMRegressor]:

        """Creates and trains a simple default model with initial priors."""
        try:
            logging.info(f"Creating default '{model_type}' model with priors.")
            if model_type == "ttft":
                features = pd.DataFrame({
                    'kv_cache_percentage': [0.0, ],
                    'input_token_length': [1, ],
                    'num_request_waiting': [0, ],
                    'num_request_running': [0, ],
                    'prefix_cache_score': [0.0, ]  # Added prefix_cache_score
                })
                features = self._prepare_features_with_interaction(features, "ttft")
                target = pd.Series([10.0])
              
            else:
                features = pd.DataFrame({
                    'kv_cache_percentage': [0.0],
                    'input_token_length': [1],  # Added input_token_length
                    'num_request_waiting': [0, ],
                    'num_request_running': [0, ],
                    'num_tokens_generated': [1,]
                })
                target = pd.Series([10.0])
            return self._train_model_with_scaling(features, target, model_name=model_type)
        except Exception as e:
            logging.error(f"Error creating default model for {model_type}: {e}", exc_info=True)
            raise

    def _calibrate_conformal_predictors(self):
        """
        Calibrate conformal predictors using test data.
        Only called in TreeLite mode for XGBoost/LightGBM models.
        """
        try:
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("Calibrating conformal predictors for TreeLite mode...")

            # Calibrate TTFT conformal predictor
            if self.ttft_model and self.ttft_test_data and len(self.ttft_test_data) >= 10:
                try:
                    # Prepare test data
                    df_raw = pd.DataFrame(list(self.ttft_test_data)).dropna()
                    df_raw = df_raw[df_raw['actual_ttft_ms'] > 0]

                    if len(df_raw) >= 10:
                        # Apply feature engineering
                        df_features = self._prepare_features_with_interaction(df_raw.copy(), model_type="ttft")

                        # Get feature columns (same as used in training)
                        feature_cols = [
                            'kv_cache_percentage', 'input_token_length', 'num_request_waiting',
                            'num_request_running', 'prefix_cache_score', 'effective_input_tokens', 'prefill_score_bucket'
                        ]
                        X = df_features[feature_cols]
                        y_true = df_raw['actual_ttft_ms'].values

                        # Defensive check: Ensure prefill_score_bucket is int32 in TreeLite mode
                        if settings.USE_TREELITE and 'prefill_score_bucket' in X.columns:
                            if str(X['prefill_score_bucket'].dtype) == 'category':
                                logging.warning("TTFT conformal: prefill_score_bucket is categorical - converting to int32")
                                X = X.copy()
                                X['prefill_score_bucket'] = X['prefill_score_bucket'].cat.codes.astype('int32')
                            elif str(X['prefill_score_bucket'].dtype) != 'int32':
                                X = X.copy()
                                X['prefill_score_bucket'] = X['prefill_score_bucket'].astype('int32')

                        # Get mean predictions from model
                        y_pred_mean = self.ttft_model.predict(X)

                        # Create and calibrate conformal predictor
                        self.ttft_conformal = ConformalQuantilePredictor(
                            quantile=self.quantile,
                            max_calibration_samples=settings.MAX_TEST_DATA_SIZE
                        )
                        self.ttft_conformal.calibrate(y_pred_mean, y_true)

                        # Get coverage stats
                        stats = self.ttft_conformal.get_coverage_stats(y_pred_mean, y_true)
                        logging.debug(
                            f"TTFT conformal calibration complete: "
                            f"coverage={stats['coverage_percent']:.1f}% (target={stats['target_coverage_percent']:.0f}%), "
                            f"violation_rate={stats['violation_rate_percent']:.1f}%, "
                            f"quantile_adjustment=+{stats['quantile_adjustment']:.2f}ms, "
                            f"samples={stats['calibration_samples']}"
                        )
                except Exception as e:
                    logging.error(
                        f"❌ CRITICAL: TTFT conformal calibration FAILED in TreeLite mode!\n"
                        f"  Error: {e}\n"
                        f"  Test samples available: {len(self.ttft_test_data)}\n"
                        f"  Model type: {self.model_type}\n"
                        f"  Impact: Bundle will have 0 calibration samples → prediction servers will fail conformal adjustment\n"
                        f"  This is a FATAL error for TreeLite mode - predictions will not have correct coverage!",
                        exc_info=True
                    )
                    self.ttft_conformal = None

            # Calibrate TPOT conformal predictor
            if self.tpot_model and self.tpot_test_data and len(self.tpot_test_data) >= 10:
                try:
                    # Prepare test data
                    df_raw = pd.DataFrame(list(self.tpot_test_data)).dropna()
                    df_raw = df_raw[df_raw['actual_tpot_ms'] > 0]

                    if len(df_raw) >= 10:
                        # Get feature columns
                        feature_cols = ['kv_cache_percentage', 'input_token_length',
                                      'num_request_waiting', 'num_request_running', 'num_tokens_generated']
                        X = df_raw[feature_cols]
                        y_true = df_raw['actual_tpot_ms'].values

                        # Get mean predictions from model
                        y_pred_mean = self.tpot_model.predict(X)

                        # Create and calibrate conformal predictor
                        self.tpot_conformal = ConformalQuantilePredictor(
                            quantile=self.quantile,
                            max_calibration_samples=settings.MAX_TEST_DATA_SIZE
                        )
                        self.tpot_conformal.calibrate(y_pred_mean, y_true)

                        # Get coverage stats
                        stats = self.tpot_conformal.get_coverage_stats(y_pred_mean, y_true)
                        logging.debug(
                            f"TPOT conformal calibration complete: "
                            f"coverage={stats['coverage_percent']:.1f}% (target={stats['target_coverage_percent']:.0f}%), "
                            f"violation_rate={stats['violation_rate_percent']:.1f}%, "
                            f"quantile_adjustment=+{stats['quantile_adjustment']:.2f}ms, "
                            f"samples={stats['calibration_samples']}"
                        )
                except Exception as e:
                    logging.error(
                        f"❌ CRITICAL: TPOT conformal calibration FAILED in TreeLite mode!\n"
                        f"  Error: {e}\n"
                        f"  Test samples available: {len(self.tpot_test_data)}\n"
                        f"  Model type: {self.model_type}\n"
                        f"  Impact: Bundle will have 0 calibration samples → prediction servers will fail conformal adjustment\n"
                        f"  This is a FATAL error for TreeLite mode - predictions will not have correct coverage!",
                        exc_info=True
                    )
                    self.tpot_conformal = None

        except Exception as e:
            logging.error(f"Error in _calibrate_conformal_predictors: {e}", exc_info=True)

    def train(self):
        try:
            with self.lock:
                ttft_snap = list(self._all_samples(self.ttft_data_buckets))
                tpot_snap = list(self._all_samples(self.tpot_data_buckets))
                total = len(ttft_snap) + len(tpot_snap)
                if total < settings.MIN_SAMPLES_FOR_RETRAIN:
                    logging.debug(f"Skipping training: only {total} samples (< {settings.MIN_SAMPLES_FOR_RETRAIN}).")
                    return
                logging.debug(f"Initiating training with {total} samples using {self.model_type} for quantile {self.quantile}.")

            new_ttft_model = new_ttft_scaler = None
            new_tpot_model = new_tpot_scaler = None

            # Train TTFT
            if ttft_snap:
                raw_ttft = pd.DataFrame(ttft_snap).dropna()
                raw_ttft = raw_ttft[raw_ttft['actual_ttft_ms'] > 0]
                df_ttft = self._prepare_features_with_interaction(raw_ttft.copy(), model_type="ttft")
                logging.debug(f"TTFT training data size: {len(df_ttft)} with sample data: {df_ttft.columns.tolist()}")
                if len(df_ttft) >= settings.MIN_SAMPLES_FOR_RETRAIN:
                    # Updated TTFT features to include prefix_cache_score
                    ttft_feature_cols_tree = [
                    'kv_cache_percentage','input_token_length','num_request_waiting',
                    'num_request_running','prefix_cache_score','effective_input_tokens','prefill_score_bucket'
                ]
                    ttft_feature_cols_br = [
                    'kv_cache_percentage','input_token_length','num_request_waiting',
                    'num_request_running','prefix_cache_score','effective_input_tokens'
                ]

                    # Build X_ttft for all model types, then trim for BR
                    X_ttft = df_ttft[ttft_feature_cols_tree]
                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        X_ttft = X_ttft[ttft_feature_cols_br]

                    y_ttft = raw_ttft['actual_ttft_ms']

                    try:
                        # raw_ttft still has the original columns including 'prefix_cache_score'
                        raw_ttft['_prefix_bucket'] = raw_ttft['prefix_cache_score'].clip(0, 1).apply(
                            lambda s: min(int(s * self.prefix_buckets), self.prefix_buckets - 1)
                        )

                        bucket_counts = raw_ttft['_prefix_bucket'].value_counts().to_dict()
                        total_ttft = len(raw_ttft)
                        num_buckets = max(1, len(bucket_counts))
                        epsilon = 1.0
                        bucket_weights = {p: total_ttft / (num_buckets * (cnt + epsilon)) for p, cnt in bucket_counts.items()}
                        sample_weight_ttft = None
                        if settings.SAMPLE_WEIGHTING_FOR_PREFIX_CACHE:
                            sample_weight_ttft = raw_ttft['_prefix_bucket'].map(bucket_weights).astype(float).to_numpy()
                            sample_weight_ttft *= (len(sample_weight_ttft) / sample_weight_ttft.sum())

                        result = self._train_model_with_scaling(X_ttft, y_ttft, model_name="ttft", sample_weight=sample_weight_ttft)
                        if self.model_type == ModelType.BAYESIAN_RIDGE:
                            new_ttft_model, new_ttft_scaler = result
                        else:
                            new_ttft_model = result
                            new_ttft_scaler = None

                        # Quantile metrics on test set
                        ql = coverage = violation_rate = None
                        if self.ttft_test_data:
                            ql, coverage, violation_rate = self._calculate_quantile_metrics_on_test(
                                new_ttft_model, new_ttft_scaler,
                                list(self.ttft_test_data),  # Pass raw data
                                "ttft",                      # Pass model name instead of feature_cols
                                'actual_ttft_ms'
                            )       


                        
                        if ql is not None:
                            self.ttft_quantile_loss_scores.append(ql)
                            self.ttft_coverage_scores.append(coverage)
                            self.ttft_violation_rates.append(violation_rate)
                            if logging.getLogger().isEnabledFor(logging.DEBUG):
                                logging.debug(f"TTFT model trained on {len(df_ttft)} samples. "
                                           f"Quantile Loss = {ql:.4f}, "
                                           f"Coverage = {coverage:.2f}% (target: {self.quantile*100:.0f}%), "
                                           f"Violation Rate = {violation_rate:.2f}% (target: {(1-self.quantile)*100:.0f}%)")
                        else:
                            if logging.getLogger().isEnabledFor(logging.DEBUG):
                                logging.debug(f"TTFT model trained on {len(df_ttft)} samples. Quantile metrics = N/A (insufficient test data)")

                    except Exception:
                        logging.error("Error training TTFT model", exc_info=True)


            # Train TPOT
            if tpot_snap:
                df_tpot = pd.DataFrame(tpot_snap).dropna()
                df_tpot = df_tpot[df_tpot['actual_tpot_ms'] > 0]
                if len(df_tpot) >= settings.MIN_SAMPLES_FOR_RETRAIN:
                    # TPOT features remain unchanged
                    X_tpot = df_tpot[['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'num_tokens_generated']]
                    y_tpot = df_tpot['actual_tpot_ms']
                    try:
                        result = self._train_model_with_scaling(X_tpot, y_tpot, model_name="tpot")
                        if self.model_type == ModelType.BAYESIAN_RIDGE:
                            new_tpot_model, new_tpot_scaler = result
                        else:
                            new_tpot_model = result
                            new_tpot_scaler = None
                        
                        # Calculate quantile metrics on test data
                        ql, coverage, violation_rate = self._calculate_quantile_metrics_on_test(
                            new_tpot_model, new_tpot_scaler, 
                            list(self.tpot_test_data),  # Pass raw data
                        "tpot",                      # Pass model name instead of feature_cols
                        'actual_tpot_ms'
                )
                        
                        if ql is not None:
                            self.tpot_quantile_loss_scores.append(ql)
                            self.tpot_coverage_scores.append(coverage)
                            self.tpot_violation_rates.append(violation_rate)
                            if logging.getLogger().isEnabledFor(logging.DEBUG):
                                logging.debug(f"TPOT model trained on {len(df_tpot)} samples. "
                                           f"Quantile Loss = {ql:.4f}, "
                                           f"Coverage = {coverage:.2f}% (target: {self.quantile*100:.0f}%), "
                                           f"Violation Rate = {violation_rate:.2f}% (target: {(1-self.quantile)*100:.0f}%)")
                        else:
                            if logging.getLogger().isEnabledFor(logging.DEBUG):
                                logging.debug(f"TPOT model trained on {len(df_tpot)} samples. Quantile metrics = N/A (insufficient test data)")
                            
                    except Exception:
                        logging.error("Error training TPOT model", exc_info=True)
                else:
                    logging.warning("Not enough TPOT samples, skipping TPOT training.")

            with self.lock:
                if new_ttft_model:
                    self.ttft_model = new_ttft_model
                    if new_ttft_scaler is not None:
                        self.ttft_scaler = new_ttft_scaler
                    
                    # Store descaled coefficients for Bayesian Ridge
                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        ttft_features = ttft_feature_cols_br  # no 'prefill_score_bucket' for BR
                        self.ttft_coefficients = self._store_descaled_coefficients(
                        new_ttft_model, new_ttft_scaler, ttft_features, "TTFT"
                )
                        
                if new_tpot_model:
                    self.tpot_model = new_tpot_model
                    if new_tpot_scaler is not None:
                        self.tpot_scaler = new_tpot_scaler

                    # Store descaled coefficients for Bayesian Ridge
                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        tpot_features = ['kv_cache_percentage', 'input_token_length',
                                       'num_request_waiting', 'num_request_running', 'num_tokens_generated']
                        self.tpot_coefficients = self._store_descaled_coefficients(
                            new_tpot_model, new_tpot_scaler, tpot_features, "TPOT"
                        )

                # Calibrate conformal predictors (only in TreeLite mode)
                if settings.USE_TREELITE and CONFORMAL_AVAILABLE:
                    if self.model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
                        self._calibrate_conformal_predictors()

                if self.is_ready:
                    self.last_retrain_time = datetime.now(timezone.utc)
                    try:
                        self._save_models_with_bundle()
                    except Exception:
                        logging.error("Error saving models after training.", exc_info=True)
        except Exception as e:
            logging.error(f"Critical error in train(): {e}", exc_info=True)

    def predict(self, features: dict) -> Tuple[float, float, float, float]:
        try:
            with self.lock:
                if not self.is_ready:
                    raise HTTPException(status_code=503, detail="Models not ready")
                required = ['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'num_tokens_generated', 'prefix_cache_score']
                for f in required:
                    if f not in features:
                        raise ValueError(f"Missing required feature: {f}")
                    if not isinstance(features[f], (int, float)):
                        raise ValueError(f"Invalid type for feature {f}: expected number")

                # Updated TTFT features to include prefix_cache_score
                ttft_cols = ['kv_cache_percentage','input_token_length','num_request_waiting','num_request_running','prefix_cache_score']
                tpot_cols = ['kv_cache_percentage','input_token_length','num_request_waiting','num_request_running','num_tokens_generated']
                
                # Create DataFrames for predictions
                df_ttft = pd.DataFrame([{col: features[col] for col in ttft_cols}])
                # Add interaction term for TTFT
                df_ttft = self._prepare_features_with_interaction(df_ttft, model_type="ttft")
                df_tpot = pd.DataFrame([{col: features[col] for col in tpot_cols}])

                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    # Use scaling for Bayesian Ridge
                    df_ttft = df_ttft.drop(columns=['prefill_score_bucket'], errors='ignore')
                    ttft_scaled = self.ttft_scaler.transform(df_ttft)
                    tpot_scaled = self.tpot_scaler.transform(df_tpot)

                    ttft_pred_mean, ttft_std = self.ttft_model.predict(ttft_scaled, return_std=True)
                    tpot_pred_mean, tpot_std = self.tpot_model.predict(tpot_scaled, return_std=True)
                    
                    # Approximate quantile prediction by adding factor to mean
                    std_factor = 1.28 if self.quantile == 0.9 else (2.0 if self.quantile == 0.95 else 0.674)
                    ttft_pred = ttft_pred_mean[0] + std_factor * ttft_std[0]
                    tpot_pred = tpot_pred_mean[0] + std_factor * tpot_std[0]
                    
                    return ttft_pred, tpot_pred, ttft_std[0], tpot_std[0]
                
                elif self.model_type == ModelType.XGBOOST:
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
        except Exception as e:
            logging.error("Error in predict():", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during prediction")

    def add_training_sample(self, sample: dict):
        try:
            required = ['kv_cache_percentage', 'actual_ttft_ms', 'actual_tpot_ms', 'num_tokens_generated', 'input_token_length', 'num_request_waiting', 'num_request_running', 'prefix_cache_score']
            for field in required:
                if field not in sample or not isinstance(sample[field], (int, float)):
                    logging.warning(f"Invalid sample field: {field}")
                    return
        
            # Use hash-based deterministic split to ensure consistent train/test assignment
            # This ensures the same sample always goes to the same split
            sample_hash = hash(str(sorted(sample.items())))
            is_test = (sample_hash % 100) < (settings.TEST_TRAIN_RATIO * 100)
        
            # Create subsets based on conditions
            ttft_valid = sample['actual_ttft_ms'] > 0
            tpot_valid = sample['actual_tpot_ms'] > 0
        
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


    def _compute_model_hash(self, model) -> str:
        """
        Compute a hash of the model to detect if it has actually changed.

        Args:
            model: The trained model (XGBoost or LightGBM)

        Returns:
            MD5 hash string of the model's serialized form
        """
        try:
            # Serialize model to bytes using BytesIO buffer
            buffer = BytesIO()
            joblib.dump(model, buffer)
            model_bytes = buffer.getvalue()
            model_hash = hashlib.md5(model_bytes).hexdigest()
            return model_hash
        except Exception as e:
            logging.error(f"Error computing model hash: {e}", exc_info=True)
            # Return a random hash to force compilation if hashing fails
            return hashlib.md5(str(time.time()).encode()).hexdigest()

    def _cleanup_old_versioned_models(self, versioned_dir: str, current_hash: str, model_name: str):
        """
        Clean up old versioned TreeLite models to prevent disk bloat.

        Keeps the most recent N versions (configured via MAX_VERSIONED_MODELS_TO_KEEP).

        Args:
            versioned_dir: Directory containing versioned .so files
            current_hash: Hash of the current model (to protect from deletion)
            model_name: Name for logging (e.g., "TTFT" or "TPOT")
        """
        try:
            if not os.path.exists(versioned_dir):
                return

            # Find all versioned .so files
            import glob
            pattern = os.path.join(versioned_dir, f"{model_name.lower()}_*.so")
            versioned_files = glob.glob(pattern)

            if len(versioned_files) <= settings.MAX_VERSIONED_MODELS_TO_KEEP:
                return  # Nothing to clean up yet

            # Sort by modification time (newest first)
            versioned_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

            # Keep the most recent N versions
            files_to_keep = versioned_files[:settings.MAX_VERSIONED_MODELS_TO_KEEP]
            files_to_delete = versioned_files[settings.MAX_VERSIONED_MODELS_TO_KEEP:]

            # Also protect current hash from deletion
            current_filename = f"{model_name.lower()}_{current_hash[:8]}.so"
            current_filepath = os.path.join(versioned_dir, current_filename)

            for filepath in files_to_delete:
                if filepath == current_filepath:
                    logging.info(f"Skipping deletion of current {model_name} model: {filepath}")
                    continue

                try:
                    os.remove(filepath)
                    logging.info(f"Cleaned up old {model_name} versioned model: {filepath}")
                except Exception as e:
                    logging.warning(f"Failed to delete old {model_name} model {filepath}: {e}")

        except Exception as e:
            logging.error(f"Error cleaning up old {model_name} versioned models: {e}", exc_info=True)

    def _should_recompile_treelite(self, model, previous_hash: Optional[str], model_name: str) -> Tuple[bool, str]:
        """
        Determine if TreeLite recompilation is needed by comparing model hash.

        Args:
            model: The trained model
            previous_hash: Hash of the previous model (None if first time)
            model_name: Name for logging (e.g., "TTFT" or "TPOT")

        Returns:
            Tuple of (should_recompile: bool, current_hash: str)
        """
        try:
            current_hash = self._compute_model_hash(model)

            if previous_hash is None:
                logging.info(f"{model_name}: First compilation (no previous hash)")
                return True, current_hash

            if current_hash == previous_hash:
                self.cache_hits += 1
                logging.debug(f"{model_name}: Model unchanged (hash: {current_hash[:8]}...), skipping compilation")

                # Periodic cache efficiency report (every 60 seconds)
                if logging.getLogger().isEnabledFor(logging.DEBUG) and time.time() - self.last_cache_report_time > 60:
                    total = self.cache_hits + self.cache_misses
                    hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
                    logging.debug(f"TreeLite compilation cache: {self.cache_hits} hits, {self.cache_misses} misses ({hit_rate:.1f}% hit rate)")
                    self.last_cache_report_time = time.time()

                return False, current_hash

            self.cache_misses += 1
            logging.debug(f"{model_name}: Model changed (old: {previous_hash[:8]}..., new: {current_hash[:8]}...), triggering TreeLite compilation")
            return True, current_hash

        except Exception as e:
            logging.error(f"Error checking if recompilation needed for {model_name}: {e}", exc_info=True)
            # On error, force recompilation to be safe
            return True, hashlib.md5(str(time.time()).encode()).hexdigest()

    def _launch_treelite_compilation(self, model_path: str, output_path: str, model_name: str, versioned_path: str = None):
        """
        Launch TreeLite compilation in a background subprocess (non-blocking).

        IMPORTANT: Caller must hold self.compilation_lock and set the in_progress flag!

        Args:
            model_path: Path to the .joblib model file
            output_path: Where to save the compiled .so file (legacy path for compatibility)
            model_name: Name for logging (e.g., "TTFT" or "TPOT")
            versioned_path: Optional versioned path for runtime updates (e.g., /models/treelite/ttft/ttft_a1b2c3d4.so)
        """
        try:
            # Set the in-progress flag (caller already holds compilation_lock)
            if model_name == "TTFT":
                self.ttft_compilation_in_progress = True
            elif model_name == "TPOT":
                self.tpot_compilation_in_progress = True

            # Path to our background compilation script
            script_path = os.path.join(os.path.dirname(__file__), 'compile_treelite_background.py')

            # Build command args - pass versioned path if provided
            cmd_args = [sys.executable, script_path, model_path, output_path, model_name]
            if versioned_path:
                cmd_args.append(versioned_path)

            # Launch subprocess in background (fire and forget)
            # Use Popen instead of run() to avoid blocking
            process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.DEVNULL,  # Don't capture output (goes to log file)
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent process
            )

            logging.info(f"🚀 Launched background TreeLite compilation for {model_name} (PID: {process.pid})")
            logging.info(f"   Model: {model_path} -> {output_path}")
            if versioned_path:
                logging.info(f"   Versioned: {versioned_path}")
            logging.info(f"   Check /tmp/treelite_compilation.log for compilation progress")

            # Reset the in-progress flag after a timeout (in case process crashes without cleanup)
            # This prevents permanent deadlock if subprocess fails silently
            def reset_flag_after_timeout():
                time.sleep(30)  # Wait 30 seconds (compilation should finish by then)
                with self.compilation_lock:
                    if model_name == "TTFT":
                        if self.ttft_compilation_in_progress:
                            logging.warning(f"TTFT compilation flag still set after 30s, resetting (compilation may have failed)")
                            self.ttft_compilation_in_progress = False
                    elif model_name == "TPOT":
                        if self.tpot_compilation_in_progress:
                            logging.warning(f"TPOT compilation flag still set after 30s, resetting (compilation may have failed)")
                            self.tpot_compilation_in_progress = False

            # Launch watchdog thread to reset flag after timeout
            threading.Thread(target=reset_flag_after_timeout, daemon=True).start()

        except Exception as e:
            # On error, reset the flag
            if model_name == "TTFT":
                self.ttft_compilation_in_progress = False
            elif model_name == "TPOT":
                self.tpot_compilation_in_progress = False
            logging.error(f"Failed to launch background TreeLite compilation for {model_name}: {e}", exc_info=True)

    def _save_models_with_bundle(self):
        """
        Save models using bundle registry (new architectural approach).
        Creates immutable versioned bundles with atomic operations and TreeLite compilation tracking.

        Requires bundle manager to be initialized.
        """
        if not self.bundle_manager:
            raise RuntimeError("Bundle manager not initialized - cannot save models")

        try:
            # Calculate training and test sample counts
            training_samples = {
                "ttft": sum(len(bucket) for bucket in self.ttft_data_buckets.values()),
                "tpot": sum(len(bucket) for bucket in self.tpot_data_buckets.values())
            }
            test_samples = {
                "ttft": len(self.ttft_test_data),
                "tpot": len(self.tpot_test_data)
            }

            # Start a new training session - creates bundle in TRAINING state
            bundle = self.bundle_manager.start_training(
                model_name="dual_ttft_tpot",
                model_type=self.model_type.value,
                quantile=self.quantile,
                use_treelite=settings.USE_TREELITE
            )

            # Increment training cycle (0 for default models, 1+ for trained models)
            # This helps distinguish untrained bundles from real training failures
            previous_cycle = self.current_bundle.manifest.training_cycle if self.current_bundle else 0
            is_real_training = training_samples.get('ttft', 0) > 0 or training_samples.get('tpot', 0) > 0

            if is_real_training:
                # Real training with data - increment cycle
                bundle.manifest.training_cycle = previous_cycle + 1
                logging.info(
                    f"Started bundle {bundle.manifest.bundle_id[:8]} "
                    f"(state: {bundle.manifest.state.value}, training_cycle: {bundle.manifest.training_cycle})"
                )
            else:
                # Default model creation (startup) - keep cycle at 0
                bundle.manifest.training_cycle = 0
                logging.info(
                    f"Started bundle {bundle.manifest.bundle_id[:8]} with default models "
                    f"(state: {bundle.manifest.state.value}, training_cycle: 0)"
                )

            # Save models to temp directory first, then add to bundle atomically
            with tempfile.TemporaryDirectory() as temp_dir:
                # === TTFT MODEL ===
                if self.ttft_model:
                    temp_ttft_path = os.path.join(temp_dir, "ttft.joblib")
                    joblib.dump(self.ttft_model, temp_ttft_path)
                    self.bundle_manager.save_model_file(bundle, TTFT_MODEL_FILENAME, temp_ttft_path)
                    logging.debug(f"Bundle {bundle.manifest.bundle_id[:8]}: Added TTFT model")

                    # TTFT model-specific exports
                    if self.model_type == ModelType.XGBOOST:
                        # Save XGBoost trees as JSON
                        booster = self.ttft_model.get_booster()
                        raw_trees = booster.get_dump(dump_format="json")
                        trees = [json.loads(t) for t in raw_trees]
                        temp_ttft_trees_path = os.path.join(temp_dir, "ttft_trees.json")
                        with open(temp_ttft_trees_path, 'w') as f:
                            json.dump(trees, f, indent=2)
                        self.bundle_manager.save_model_file(bundle, "ttft_trees", temp_ttft_trees_path)

                    elif self.model_type == ModelType.LIGHTGBM:
                        # Save LightGBM model as text and feature importances
                        temp_ttft_txt_path = os.path.join(temp_dir, "ttft_lgb.txt")
                        self.ttft_model.booster_.save_model(temp_ttft_txt_path)
                        self.bundle_manager.save_model_file(bundle, "ttft_lgb.txt", temp_ttft_txt_path)

                        # Feature importances
                        feature_names = ['kv_cache_percentage', 'input_token_length',
                                       'num_request_waiting', 'num_request_running', 'prefix_cache_score',
                                       'effective_input_tokens', 'prefill_score_bucket']
                        importances = {name: float(imp) for name, imp in zip(feature_names, self.ttft_model.feature_importances_)}
                        temp_ttft_imp_path = os.path.join(temp_dir, "ttft_importances.json")
                        with open(temp_ttft_imp_path, 'w') as f:
                            json.dump(importances, f, indent=2)
                        self.bundle_manager.save_model_file(bundle, "ttft_importances.json", temp_ttft_imp_path)

                # === TPOT MODEL ===
                if self.tpot_model:
                    temp_tpot_path = os.path.join(temp_dir, "tpot.joblib")
                    joblib.dump(self.tpot_model, temp_tpot_path)
                    self.bundle_manager.save_model_file(bundle, TPOT_MODEL_FILENAME, temp_tpot_path)
                    logging.debug(f"Bundle {bundle.manifest.bundle_id[:8]}: Added TPOT model")

                    # TPOT model-specific exports
                    if self.model_type == ModelType.XGBOOST:
                        # Save XGBoost trees as JSON
                        booster = self.tpot_model.get_booster()
                        raw_trees = booster.get_dump(dump_format="json")
                        trees = [json.loads(t) for t in raw_trees]
                        temp_tpot_trees_path = os.path.join(temp_dir, "tpot_trees.json")
                        with open(temp_tpot_trees_path, 'w') as f:
                            json.dump(trees, f, indent=2)
                        self.bundle_manager.save_model_file(bundle, "tpot_trees", temp_tpot_trees_path)

                    elif self.model_type == ModelType.LIGHTGBM:
                        # Save LightGBM model as text and feature importances
                        temp_tpot_txt_path = os.path.join(temp_dir, "tpot_lgb.txt")
                        self.tpot_model.booster_.save_model(temp_tpot_txt_path)
                        self.bundle_manager.save_model_file(bundle, "tpot_lgb.txt", temp_tpot_txt_path)

                        # Feature importances
                        feature_names = ['kv_cache_percentage', 'input_token_length',
                                       'num_request_waiting', 'num_request_running', 'num_tokens_generated']
                        importances = {name: float(imp) for name, imp in zip(feature_names, self.tpot_model.feature_importances_)}
                        temp_tpot_imp_path = os.path.join(temp_dir, "tpot_importances.json")
                        with open(temp_tpot_imp_path, 'w') as f:
                            json.dump(importances, f, indent=2)
                        self.bundle_manager.save_model_file(bundle, "tpot_importances.json", temp_tpot_imp_path)

                # === SCALERS (Bayesian Ridge only) ===
                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    if self.ttft_scaler:
                        temp_ttft_scaler_path = os.path.join(temp_dir, "ttft_scaler.joblib")
                        joblib.dump(self.ttft_scaler, temp_ttft_scaler_path)
                        self.bundle_manager.save_model_file(bundle, TTFT_SCALER_FILENAME, temp_ttft_scaler_path)

                    if self.tpot_scaler:
                        temp_tpot_scaler_path = os.path.join(temp_dir, "tpot_scaler.joblib")
                        joblib.dump(self.tpot_scaler, temp_tpot_scaler_path)
                        self.bundle_manager.save_model_file(bundle, TPOT_SCALER_FILENAME, temp_tpot_scaler_path)

                # === CONFORMAL CALIBRATION (TreeLite mode only) ===
                if settings.USE_TREELITE and CONFORMAL_AVAILABLE:
                    if self.ttft_conformal:
                        temp_ttft_conf_path = os.path.join(temp_dir, "ttft_conformal.json")
                        # Use get_state() + manual JSON save (no .save() method exists)
                        with open(temp_ttft_conf_path, 'w') as f:
                            json.dump(self.ttft_conformal.get_state(), f, indent=2)
                        self.bundle_manager.save_model_file(bundle, TTFT_CONFORMAL_FILENAME, temp_ttft_conf_path)
                    else:
                        # TreeLite mode but no conformal calibration
                        # Use training_cycle to determine if this is expected or an error
                        if bundle.manifest.training_cycle == 0:
                            # Default models (training_cycle=0) - expected to have no conformal
                            logging.info(
                                f"TreeLite mode: No TTFT conformal calibration (training_cycle=0, default models)"
                            )
                        elif test_samples.get('ttft', 0) < 10:
                            # Insufficient test data - can't create conformal, but not critical yet
                            logging.warning(
                                f"TreeLite mode: Insufficient test data for TTFT conformal calibration\n"
                                f"  Training cycle: {bundle.manifest.training_cycle}\n"
                                f"  Test samples: {test_samples.get('ttft', 0)} (required: ≥10)\n"
                                f"  Bundle will use statistical adjustment until more test data is available"
                            )
                        else:
                            # Real error - trained model with sufficient data but conformal failed
                            logging.error(
                                f"❌ CRITICAL: TreeLite mode enabled but TTFT conformal predictor is None!\n"
                                f"  Training cycle: {bundle.manifest.training_cycle}\n"
                                f"  Training samples: {training_samples.get('ttft', 0)}\n"
                                f"  Test samples: {test_samples.get('ttft', 0)} (required: ≥10)\n"
                                f"  Model type: {self.model_type}\n"
                                f"  ⚠️  Bundle will be published WITHOUT TTFT conformal calibration!\n"
                                f"  Impact: Prediction servers will have 0 calibration samples → INCORRECT COVERAGE\n"
                                f"  Check logs above for conformal calibration errors"
                            )

                    if self.tpot_conformal:
                        temp_tpot_conf_path = os.path.join(temp_dir, "tpot_conformal.json")
                        # Use get_state() + manual JSON save (no .save() method exists)
                        with open(temp_tpot_conf_path, 'w') as f:
                            json.dump(self.tpot_conformal.get_state(), f, indent=2)
                        self.bundle_manager.save_model_file(bundle, TPOT_CONFORMAL_FILENAME, temp_tpot_conf_path)
                    else:
                        # TreeLite mode but no conformal calibration
                        # Use training_cycle to determine if this is expected or an error
                        if bundle.manifest.training_cycle == 0:
                            # Default models (training_cycle=0) - expected to have no conformal
                            logging.info(
                                f"TreeLite mode: No TPOT conformal calibration (training_cycle=0, default models)"
                            )
                        elif test_samples.get('tpot', 0) < 10:
                            # Insufficient test data - can't create conformal, but not critical yet
                            logging.warning(
                                f"TreeLite mode: Insufficient test data for TPOT conformal calibration\n"
                                f"  Training cycle: {bundle.manifest.training_cycle}\n"
                                f"  Test samples: {test_samples.get('tpot', 0)} (required: ≥10)\n"
                                f"  Bundle will use statistical adjustment until more test data is available"
                            )
                        else:
                            # Real error - trained model with sufficient data but conformal failed
                            logging.error(
                                f"❌ CRITICAL: TreeLite mode enabled but TPOT conformal predictor is None!\n"
                                f"  Training cycle: {bundle.manifest.training_cycle}\n"
                                f"  Training samples: {training_samples.get('tpot', 0)}\n"
                                f"  Test samples: {test_samples.get('tpot', 0)} (required: ≥10)\n"
                                f"  Model type: {self.model_type}\n"
                                f"  ⚠️  Bundle will be published WITHOUT TPOT conformal calibration!\n"
                                f"  Impact: Prediction servers will have 0 calibration samples → INCORRECT COVERAGE\n"
                                f"  Check logs above for conformal calibration errors"
                            )

            # === TREELITE COMPILATION ===
            # Note: XGBoost uses background compilation, LightGBM uses synchronous
            if settings.USE_TREELITE and TREELITE_AVAILABLE and TL2CGEN_AVAILABLE:
                if self.model_type == ModelType.XGBOOST:
                    # XGBoost: Background compilation
                    bundle.manifest.transition_state(BundleState.COMPILING, "Starting background TreeLite compilation for XGBoost")
                    bundle.save_manifest()

                    # Get model paths from bundle
                    ttft_model_path = str(bundle.path / TTFT_MODEL_FILENAME)
                    tpot_model_path = str(bundle.path / TPOT_MODEL_FILENAME)

                    # Launch background compilations
                    with self.compilation_lock:
                        # TTFT compilation
                        if self.ttft_model and not self.ttft_compilation_in_progress:
                            ttft_treelite_path = str(bundle.path / TTFT_TREELITE_FILENAME)
                            script_path = os.path.join(os.path.dirname(__file__), 'compile_treelite_background.py')
                            cmd_args = [sys.executable, script_path, ttft_model_path, ttft_treelite_path, "TTFT"]

                            process = subprocess.Popen(
                                cmd_args,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True
                            )

                            self.bundle_manager.start_compilation(bundle, "ttft", process)
                            logging.info(f"Launched background TreeLite compilation for TTFT (PID: {process.pid})")

                        # TPOT compilation
                        if self.tpot_model and not self.tpot_compilation_in_progress:
                            tpot_treelite_path = str(bundle.path / TPOT_TREELITE_FILENAME)
                            script_path = os.path.join(os.path.dirname(__file__), 'compile_treelite_background.py')
                            cmd_args = [sys.executable, script_path, tpot_model_path, tpot_treelite_path, "TPOT"]

                            process = subprocess.Popen(
                                cmd_args,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True
                            )

                            self.bundle_manager.start_compilation(bundle, "tpot", process)
                            logging.info(f"Launched background TreeLite compilation for TPOT (PID: {process.pid})")

                    # CRITICAL: Wait for background compilation to complete before publishing
                    # Otherwise bundles get published without .so files, causing prediction servers
                    # to use mean regression models instead of TreeLite + conformal
                    logging.info("Waiting for background TreeLite compilation to complete...")
                    max_wait_seconds = 30  # XGBoost compilation should finish within 30 seconds
                    poll_interval = 0.5  # Check every 500ms
                    elapsed = 0

                    while elapsed < max_wait_seconds:
                        import time
                        time.sleep(poll_interval)
                        elapsed += poll_interval

                        # Check compilation status for both models
                        ttft_status = self.bundle_manager.check_compilation_status(bundle, "ttft")
                        tpot_status = self.bundle_manager.check_compilation_status(bundle, "tpot")

                        # Check if both completed successfully
                        if ttft_status and tpot_status and ttft_status.value == "completed" and tpot_status.value == "completed":
                            logging.info(f"✓ XGBoost TreeLite compilation completed after {elapsed:.1f}s")

                            # Add .so files to bundle manifest (mark_compilation_complete does this)
                            ttft_treelite_path = str(bundle.path / TTFT_TREELITE_FILENAME)
                            tpot_treelite_path = str(bundle.path / TPOT_TREELITE_FILENAME)
                            self.bundle_manager.mark_compilation_complete(bundle, "ttft", ttft_treelite_path, True)
                            self.bundle_manager.mark_compilation_complete(bundle, "tpot", tpot_treelite_path, True)

                            bundle.manifest.transition_state(BundleState.READY, "All files including TreeLite compiled and verified")
                            break

                        # Check if any failed
                        if (ttft_status and ttft_status.value == "failed") or (tpot_status and tpot_status.value == "failed"):
                            ttft_val = ttft_status.value if ttft_status else "unknown"
                            tpot_val = tpot_status.value if tpot_status else "unknown"
                            logging.error(
                                f"TreeLite compilation FAILED: TTFT={ttft_val}, TPOT={tpot_val}. "
                                f"Bundle will be marked as FAILED and NOT published. "
                                f"Check /tmp/treelite_compilation.log for details. "
                            )
                            bundle.manifest.transition_state(BundleState.FAILED, f"TreeLite compilation failed (TTFT={ttft_val}, TPOT={tpot_val})")
                            break
                    else:
                        # Timeout - compilation taking too long
                        ttft_val = ttft_status.value if ttft_status else "unknown"
                        tpot_val = tpot_status.value if tpot_status else "unknown"
                        logging.error(
                            f"TreeLite compilation TIMEOUT after {max_wait_seconds}s: TTFT={ttft_val}, TPOT={tpot_val}. "
                            f"Bundle will be marked as FAILED and NOT published. "
                            f"Check /tmp/treelite_compilation.log for details."
                        )
                        bundle.manifest.transition_state(BundleState.FAILED, f"TreeLite compilation timeout (TTFT={ttft_val}, TPOT={tpot_val})")

                elif self.model_type == ModelType.LIGHTGBM:
                    # LightGBM: Synchronous compilation (fast enough)
                    bundle.manifest.transition_state(BundleState.COMPILING, "Starting TreeLite compilation for LightGBM")
                    bundle.save_manifest()

                    try:
                        # TTFT LightGBM compilation
                        if self.ttft_model:
                            tl_model = treelite.frontend.from_lightgbm(self.ttft_model.booster_)
                            ttft_treelite_path = str(bundle.path / TTFT_TREELITE_FILENAME)
                            tl2cgen.export_lib(
                                model=tl_model,
                                toolchain='gcc',
                                libpath=ttft_treelite_path,
                                params={'parallel_comp': 8},
                                verbose=False
                            )
                            self.bundle_manager.mark_compilation_complete(bundle, "ttft", ttft_treelite_path, True)
                            logging.info(f"TTFT LightGBM TreeLite compilation completed")

                        # TPOT LightGBM compilation
                        if self.tpot_model:
                            tl_model = treelite.frontend.from_lightgbm(self.tpot_model.booster_)
                            tpot_treelite_path = str(bundle.path / TPOT_TREELITE_FILENAME)
                            tl2cgen.export_lib(
                                model=tl_model,
                                toolchain='gcc',
                                libpath=tpot_treelite_path,
                                params={'parallel_comp': 8},
                                verbose=False
                            )
                            self.bundle_manager.mark_compilation_complete(bundle, "tpot", tpot_treelite_path, True)
                            logging.info(f"TPOT LightGBM TreeLite compilation completed")

                        bundle.manifest.transition_state(BundleState.READY, "All files including TreeLite compiled and verified")
                    except Exception as e:
                        logging.error(
                            f"LightGBM TreeLite compilation FAILED: {e}. "
                            f"Bundle will be marked as FAILED and NOT published.",
                            exc_info=True
                        )
                        bundle.manifest.transition_state(BundleState.FAILED, f"TreeLite compilation failed: {str(e)}")
            else:
                # No TreeLite - mark as READY immediately
                bundle.manifest.transition_state(BundleState.READY, "All files written and verified (no TreeLite)")

            bundle.save_manifest()

            # CRITICAL: Only finalize and publish if bundle is in READY state
            # If compilation failed, bundle will be in FAILED state and should NOT be published
            if bundle.manifest.state == BundleState.FAILED:
                logging.error(
                    f"Bundle {bundle.manifest.bundle_id[:8]} is in FAILED state - will NOT be published. "
                    f"Check logs above for error details."
                )
                return  # Early return - do not publish failed bundles

            # Finalize and publish bundle atomically (raises exception on failure)
            self.bundle_manager.finalize_bundle(bundle, training_samples, test_samples)

            logging.info(f"Bundle {bundle.manifest.bundle_id[:8]} published successfully at {settings.BUNDLE_CURRENT_SYMLINK}")
            self.current_bundle = bundle

            # CRITICAL: Update model hashes so prediction servers can detect changes
            # Without this, /model/{name}/hash will return "initial" forever
            # and prediction servers will never download the new models
            if self.ttft_model:
                self.ttft_model_hash = self._compute_model_hash(self.ttft_model)
                logging.debug(f"Updated TTFT model hash: {self.ttft_model_hash[:8]}...")
            if self.tpot_model:
                self.tpot_model_hash = self._compute_model_hash(self.tpot_model)
                logging.debug(f"Updated TPOT model hash: {self.tpot_model_hash[:8]}...")

        except Exception as e:
            logging.error(f"Error saving models with bundle registry: {e}", exc_info=True)
            raise

    def flush_training_data(self, flush_training: bool = True, flush_test: bool = True, 
                   flush_metrics: bool = True, reason: str = None) -> dict:
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
                    logging.info(
                        f"Flushed {ttft_training_count} TTFT and {tpot_training_count} TPOT training samples"
                    )
        
                # Flush test data
                if flush_test:
                    self.ttft_test_data.clear()
                    self.tpot_test_data.clear()
                    logging.info(
                        f"Flushed {ttft_test_count} TTFT and {tpot_test_count} TPOT test samples"
                    )
        
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

                # CRITICAL: Clear conformal calibration when flushing test data
                # This ensures test isolation - conformal predictors are recalibrated on fresh test data
                conformal_cleared = False
                if flush_test and settings.USE_TREELITE:
                    if self.ttft_conformal:
                        # Reset conformal predictor to empty state
                        self.ttft_conformal = ConformalQuantilePredictor(
                            quantile=self.quantile,
                            max_calibration_samples=settings.MAX_TEST_DATA_SIZE
                        ) if CONFORMAL_AVAILABLE else None
                        conformal_cleared = True
                    if self.tpot_conformal:
                        # Reset conformal predictor to empty state
                        self.tpot_conformal = ConformalQuantilePredictor(
                            quantile=self.quantile,
                            max_calibration_samples=settings.MAX_TEST_DATA_SIZE
                        ) if CONFORMAL_AVAILABLE else None
                        conformal_cleared = True
                    if conformal_cleared:
                        logging.info("Cleared conformal calibration (will be recalibrated on next training cycle)")
        
                return {
                    "success": True,
                    "ttft_training_samples_flushed": ttft_training_count if flush_training else 0,
                    "tpot_training_samples_flushed": tpot_training_count if flush_training else 0,
                    "ttft_test_samples_flushed": ttft_test_count if flush_test else 0,
                    "tpot_test_samples_flushed": tpot_test_count if flush_test else 0,
                    "metrics_cleared": metrics_cleared,
                    "conformal_cleared": conformal_cleared
                }
        
        except Exception as e:
            logging.error(f"Error flushing data: {e}", exc_info=True)
            raise
    
    def load_models(self):
        """
        Load models from existing bundle or create initial default models.

        This method is bundle-native: it tries to load from the active bundle first,
        and only creates default models if no bundle exists (first startup).
        """
        try:
            with self.lock:
                # Try to load from existing bundle
                active_bundle = self.bundle_manager.get_active_bundle()

                if active_bundle:
                    # Bundle exists - load models from it
                    logging.info(f"Loading models from existing bundle {active_bundle.manifest.bundle_id[:8]}")

                    # Load TTFT model
                    ttft_model_path = active_bundle.path / TTFT_MODEL_FILENAME
                    if ttft_model_path.exists():
                        self.ttft_model = joblib.load(ttft_model_path)
                        logging.info(f"Loaded TTFT model from bundle")

                        # Load TTFT scaler if Bayesian Ridge
                        if self.model_type == ModelType.BAYESIAN_RIDGE:
                            ttft_scaler_path = active_bundle.path / TTFT_SCALER_FILENAME
                            if ttft_scaler_path.exists():
                                self.ttft_scaler = joblib.load(ttft_scaler_path)
                                logging.info(f"Loaded TTFT scaler from bundle")

                    # Load TPOT model
                    tpot_model_path = active_bundle.path / TPOT_MODEL_FILENAME
                    if tpot_model_path.exists():
                        self.tpot_model = joblib.load(tpot_model_path)
                        logging.info(f"Loaded TPOT model from bundle")

                        # Load TPOT scaler if Bayesian Ridge
                        if self.model_type == ModelType.BAYESIAN_RIDGE:
                            tpot_scaler_path = active_bundle.path / TPOT_SCALER_FILENAME
                            if tpot_scaler_path.exists():
                                self.tpot_scaler = joblib.load(tpot_scaler_path)
                                logging.info(f"Loaded TPOT scaler from bundle")

                    # Load conformal predictors if TreeLite mode
                    if settings.USE_TREELITE and CONFORMAL_AVAILABLE:
                        ttft_conformal_path = active_bundle.path / TTFT_CONFORMAL_FILENAME
                        if ttft_conformal_path.exists():
                            with open(ttft_conformal_path, 'r') as f:
                                conformal_state = json.load(f)
                            self.ttft_conformal = ConformalQuantilePredictor(
                                quantile=self.quantile,
                                max_calibration_samples=settings.MAX_TEST_DATA_SIZE
                            )
                            self.ttft_conformal.load_state(conformal_state)
                            logging.info(f"Loaded TTFT conformal predictor from bundle")

                        tpot_conformal_path = active_bundle.path / TPOT_CONFORMAL_FILENAME
                        if tpot_conformal_path.exists():
                            with open(tpot_conformal_path, 'r') as f:
                                conformal_state = json.load(f)
                            self.tpot_conformal = ConformalQuantilePredictor(
                                quantile=self.quantile,
                                max_calibration_samples=settings.MAX_TEST_DATA_SIZE
                            )
                            self.tpot_conformal.load_state(conformal_state)
                            logging.info(f"Loaded TPOT conformal predictor from bundle")

                # If no bundle exists or models missing, create defaults
                if not self.ttft_model:
                    logging.info("No TTFT model in bundle - creating default model")
                    result = self._create_default_model("ttft")
                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        self.ttft_model, self.ttft_scaler = result
                    else:
                        self.ttft_model = result
                    settings.MIN_SAMPLES_FOR_RETRAIN = settings.MIN_SAMPLES_FOR_RETRAIN_FRESH

                if not self.tpot_model:
                    logging.info("No TPOT model in bundle - creating default model")
                    result = self._create_default_model("tpot")
                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        self.tpot_model, self.tpot_scaler = result
                    else:
                        self.tpot_model = result
                    settings.MIN_SAMPLES_FOR_RETRAIN = settings.MIN_SAMPLES_FOR_RETRAIN_FRESH

                # If we created any default models, save initial bundle
                if not active_bundle or not self.ttft_model or not self.tpot_model:
                    logging.info("Creating initial bundle with default models")
                    self._save_models_with_bundle()

                if not self.is_ready:
                    raise RuntimeError("Failed to initialize models/scalers")

                logging.info(f"Models loaded successfully. Ready for training.")

        except Exception as e:
            logging.error(f"Critical error in load_models: {e}", exc_info=True)
            raise
        
    def get_metrics(self) -> str:
        """Render Prometheus-style metrics: model, coefficients/importances, bucket counts, and quantile-specific scores."""
        try:
        # Snapshot models & scalers
            ttft_model, tpot_model = self.ttft_model, self.tpot_model
            ttft_scaler, tpot_scaler = self.ttft_scaler, self.tpot_scaler

            lines: List[str] = []
            # 1) Model type and quantile info
            lines.append(f'model_type{{type="{self.model_type.value}"}} 1')
            lines.append(f'model_quantile{{}} {self.quantile}')

            # Helper: emit linear‐model coefs or tree importances
            def emit_metrics(model, coefficients, feats, prefix):
                if model is None:
                    # placeholders
                    lines.append(f'{prefix}_intercept{{}} 0.0')
                    kind = "coef" if self.model_type == ModelType.BAYESIAN_RIDGE else "importance"
                    for f in feats:
                        lines.append(f'{prefix}_{kind}{{feature="{f}"}} 0.0')
                    return

                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    # Use stored descaled coefficients
                    if coefficients:
                        lines.append(f'{prefix}_intercept{{}} {coefficients.get("intercept", 0.0):.6f}')
                        for f in feats:
                            coef_value = coefficients.get(f, 0.0)
                            lines.append(f'{prefix}_coef{{feature="{f}"}} {coef_value:.6f}')
                    else:
                        # Fallback to zeros if coefficients not available
                        lines.append(f'{prefix}_intercept{{}} 0.0')
                        for f in feats:
                            lines.append(f'{prefix}_coef{{feature="{f}"}} 0.0')
                else:
                    # XGBoost/LightGBM importances
                    try:
                        imps = model.feature_importances_
                    except Exception:
                        imps = [0.0]*len(feats)
                    lines.append(f'{prefix}_intercept{{}} 0.0')
                    for f, imp in zip(feats, imps):
                        lines.append(f'{prefix}_importance{{feature="{f}"}} {imp:.6f}')

            if self.model_type == ModelType.BAYESIAN_RIDGE:
                ttft_feats = ["kv_cache_percentage","input_token_length","num_request_waiting",
                  "num_request_running","prefix_cache_score","effective_input_tokens"]
            else:
                ttft_feats = ["kv_cache_percentage","input_token_length","num_request_waiting",
                  "num_request_running","prefix_cache_score","effective_input_tokens","prefill_score_bucket"]

            tpot_feats = ["kv_cache_percentage","input_token_length","num_request_waiting",
              "num_request_running","num_tokens_generated"]
            emit_metrics(ttft_model, self.ttft_coefficients, ttft_feats, "ttft")
            emit_metrics(tpot_model, self.tpot_coefficients, tpot_feats, "tpot")

            # 3) Multi-dimensional bucket counts with 3D keys
            for (queue_bucket, cache_bucket, prefix_bucket), bucket_deque in self.ttft_data_buckets.items():
                count = len(bucket_deque)
                lines.append(f'training_samples_count{{model="ttft",queue_bucket="{queue_bucket}",cache_bucket="{cache_bucket}",prefix_bucket="{prefix_bucket}"}} {count}')
    
            for (queue_bucket, cache_bucket, prefix_bucket), bucket_deque in self.tpot_data_buckets.items():
                count = len(bucket_deque)
                lines.append(f'training_samples_count{{model="tpot",queue_bucket="{queue_bucket}",cache_bucket="{cache_bucket}",prefix_bucket="{prefix_bucket}"}} {count}')
    
            # Summary metrics by queue state
            for q in range(self.queue_buckets):
                ttft_total = sum(len(self.ttft_data_buckets[(q, c, p)]) 
                               for c in range(self.cache_buckets) 
                               for p in range(self.prefix_buckets))
                tpot_total = sum(len(self.tpot_data_buckets[(q, c, p)]) 
                               for c in range(self.cache_buckets) 
                               for p in range(self.prefix_buckets))
                lines.append(f'training_samples_queue_total{{model="ttft",queue_bucket="{q}"}} {ttft_total}')
                lines.append(f'training_samples_queue_total{{model="tpot",queue_bucket="{q}"}} {tpot_total}')

            # Summary metrics by cache state
            for c in range(self.cache_buckets):
                ttft_total = sum(len(self.ttft_data_buckets[(q, c, p)]) 
                               for q in range(self.queue_buckets) 
                               for p in range(self.prefix_buckets))
                tpot_total = sum(len(self.tpot_data_buckets[(q, c, p)]) 
                               for q in range(self.queue_buckets) 
                               for p in range(self.prefix_buckets))
                lines.append(f'training_samples_cache_total{{model="ttft",cache_bucket="{c}"}} {ttft_total}')
                lines.append(f'training_samples_cache_total{{model="tpot",cache_bucket="{c}"}} {tpot_total}')

            # Summary metrics by prefix state
            for p in range(self.prefix_buckets):
                ttft_total = sum(len(self.ttft_data_buckets[(q, c, p)]) 
                               for q in range(self.queue_buckets) 
                               for c in range(self.cache_buckets))
                tpot_total = sum(len(self.tpot_data_buckets[(q, c, p)]) 
                               for q in range(self.queue_buckets) 
                               for c in range(self.cache_buckets))
            
                # Calculate prefix range for this bucket
                prefix_low = p / self.prefix_buckets
                prefix_high = (p + 1) / self.prefix_buckets
            
                lines.append(f'training_samples_prefix_total{{model="ttft",prefix_bucket="{p}",range="{prefix_low:.2f}-{prefix_high:.2f}"}} {ttft_total}')
                lines.append(f'training_samples_prefix_total{{model="tpot",prefix_bucket="{p}",range="{prefix_low:.2f}-{prefix_high:.2f}"}} {tpot_total}')

            # Add prefix score distribution statistics
            all_ttft_samples = self._all_samples(self.ttft_data_buckets)
            if all_ttft_samples:
                prefix_scores = [s['prefix_cache_score'] for s in all_ttft_samples]
                ttfts = [s['actual_ttft_ms'] for s in all_ttft_samples]
            
                lines.append(f'prefix_score_mean{{}} {np.mean(prefix_scores):.4f}')
                lines.append(f'prefix_score_std{{}} {np.std(prefix_scores):.4f}')
                lines.append(f'prefix_score_min{{}} {np.min(prefix_scores):.4f}')
                lines.append(f'prefix_score_max{{}} {np.max(prefix_scores):.4f}')
            
                # Average TTFT by prefix bucket
                for p in range(self.prefix_buckets):
                    prefix_low = p / self.prefix_buckets
                    prefix_high = (p + 1) / self.prefix_buckets
                
                    if p == self.prefix_buckets - 1:
                        mask = [(prefix_low <= score <= prefix_high) for score in prefix_scores]  # include 1.0
                    else:
                        mask = [(prefix_low <= score <  prefix_high) for score in prefix_scores]
                    bucket_ttfts = [t for t, m in zip(ttfts, mask) if m]
                
                    if bucket_ttfts:
                        avg_ttft = np.mean(bucket_ttfts)
                        median_ttft = np.median(bucket_ttfts)
                        lines.append(f'avg_ttft_by_prefix{{prefix_bucket="{p}",range="{prefix_low:.2f}-{prefix_high:.2f}"}} {avg_ttft:.2f}')
                        lines.append(f'median_ttft_by_prefix{{prefix_bucket="{p}",range="{prefix_low:.2f}-{prefix_high:.2f}"}} {median_ttft:.2f}')

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
            lines.append(f'target_coverage_percent{{}} {target_coverage:.1f}')
            lines.append(f'target_violation_rate_percent{{}} {target_violation_rate:.1f}')

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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PredictionRequest(BaseModel):
    kv_cache_percentage: float = Field(..., ge=0.0, le=1.0)
    input_token_length: int = Field(..., ge=0)
    num_request_waiting: int = Field(..., ge=0)
    num_request_running: int = Field(..., ge=0)
    num_tokens_generated: int = Field(..., ge=0)
    prefix_cache_score: float = Field(..., ge=0.0, le=1.0, description="Prefix cache hit ratio score (0.0 to 1.0)")

class PredictionResponse(BaseModel):
    ttft_ms: float = Field(..., description=f"Predicted {settings.QUANTILE_ALPHA:.0%} quantile TTFT in milliseconds")
    tpot_ms: float = Field(..., description=f"Predicted {settings.QUANTILE_ALPHA:.0%} quantile TPOT in milliseconds")
    ttft_uncertainty: float = Field(..., description="Uncertainty estimate for TTFT prediction")
    tpot_uncertainty: float = Field(..., description="Uncertainty estimate for TPOT prediction")
    ttft_prediction_bounds: Tuple[float, float] = Field(..., description="Approximate prediction bounds for TTFT")
    tpot_prediction_bounds: Tuple[float, float] = Field(..., description="Approximate prediction bounds for TPOT")
    predicted_at: datetime
    model_type: ModelType = Field(default=predictor.model_type.value, description="Type of model used for prediction")
    quantile: float = Field(default=settings.QUANTILE_ALPHA, description="Quantile being predicted")
    
class BulkTrainingRequest(BaseModel):
    entries: List[TrainingEntry]



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
        ttft_bounds = (max(0, ttft_pred - 2*ttft_std), ttft_pred + 2*ttft_std)
        tpot_bounds = (max(0, tpot_pred - 2*tpot_std), tpot_pred + 2*tpot_std)
        return PredictionResponse(
            ttft_ms=ttft_pred,
            tpot_ms=tpot_pred,
            ttft_uncertainty=ttft_std,
            tpot_uncertainty=tpot_std,
            ttft_prediction_bounds=ttft_bounds,
            tpot_prediction_bounds=tpot_bounds,
            predicted_at=datetime.now(timezone.utc),
            model_type=predictor.model_type.value,
            quantile=predictor.quantile
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
        "description": f"Predicting {predictor.quantile:.0%} quantile for TTFT and TPOT latencies"
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
            reason=request.reason
        )
        
        total_flushed = (
            result["ttft_training_samples_flushed"] + 
            result["tpot_training_samples_flushed"] +
            result["ttft_test_samples_flushed"] + 
            result["tpot_test_samples_flushed"]
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
            flushed_at=datetime.now(timezone.utc),
            reason=request.reason,
            ttft_training_samples_flushed=result["ttft_training_samples_flushed"],
            tpot_training_samples_flushed=result["tpot_training_samples_flushed"],
            ttft_test_samples_flushed=result["ttft_test_samples_flushed"],
            tpot_test_samples_flushed=result["tpot_test_samples_flushed"],
            metrics_cleared=result["metrics_cleared"],
            message=message
        )
        
    except Exception as e:
        logging.error(f"Error in flush endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to flush data: {str(e)}"
        )


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
            "total_samples": ttft_training_count + tpot_training_count
        },
        "test_data": {
            "ttft_samples": len(predictor.ttft_test_data),
            "tpot_samples": len(predictor.tpot_test_data),
            "total_samples": len(predictor.ttft_test_data) + len(predictor.tpot_test_data)
        },
        "metrics": {
            "ttft_scores_count": len(predictor.ttft_quantile_loss_scores),
            "tpot_scores_count": len(predictor.tpot_quantile_loss_scores)
        },
        "bucket_distribution": bucket_distribution,
        "model_ready": predictor.is_ready,
        "last_retrain": predictor.last_retrain_time.isoformat() if predictor.last_retrain_time else None
    }

@app.get("/model/download/info")
async def model_download_info():
    """
    Get information about available model downloads and coefficients.
    """
    info = {
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile,
        "available_endpoints": {}
    }
    
    if predictor.model_type == ModelType.BAYESIAN_RIDGE:
        info["available_endpoints"]["coefficients"] = "/metrics"
        info["coefficients_info"] = {
            "ttft_coefficients_available": predictor.ttft_coefficients is not None,
            "tpot_coefficients_available": predictor.tpot_coefficients is not None,
            "description": "Descaled coefficients available in Prometheus metrics endpoint"
        }
    elif predictor.model_type == ModelType.XGBOOST:
        info["available_endpoints"]["trees"] = {
            "ttft_trees": "/model/ttft/xgb/json",
            "tpot_trees": "/model/tpot/xgb/json"
        }
    else: 
        info["available_endpoints"]["lightgbm"] = {
            "ttft_model_txt": "/model/ttft/lgb/txt",
            "tpot_model_txt": "/model/tpot/lgb/txt",
            "ttft_importances": "/model/ttft/lgb/importances",
            "tpot_importances": "/model/tpot/lgb/importances"
        }
    
    info["model_status"] = {
        "ttft_model_ready": predictor.ttft_model is not None,
        "tpot_model_ready": predictor.tpot_model is not None,
    }
    
    if predictor.model_type == ModelType.BAYESIAN_RIDGE:
        info["model_status"]["ttft_coefficients_ready"] = predictor.ttft_coefficients is not None
        info["model_status"]["tpot_coefficients_ready"] = predictor.tpot_coefficients is not None
    
    # Add quantile-specific evaluation info
    info["evaluation_info"] = {
        "quantile_loss": "Pinball loss for quantile regression evaluation",
        "coverage_percent": f"Percentage of actual values below predicted {predictor.quantile:.0%} quantile (target: {predictor.quantile*100:.1f}%)",
        "violation_rate_percent": f"Percentage of actual values above predicted {predictor.quantile:.0%} quantile (target: {(1-predictor.quantile)*100:.1f}%)"
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



@app.get("/model/{model_name}/hash")
async def get_model_hash(model_name: str):
    """
    Get the current hash of a model without downloading it.

    This allows prediction servers to check if a model has changed
    before downloading, reducing unnecessary network traffic.
    """
    if model_name == "ttft":
        if not predictor.ttft_model:
            raise HTTPException(status_code=404, detail="TTFT model not available")

        return {
            "model_name": "ttft",
            "hash": predictor.ttft_model_hash or "initial",
            "last_retrain": predictor.last_retrain_time.isoformat() if predictor.last_retrain_time else None,
            "model_type": predictor.model_type.value,
            "use_treelite": settings.USE_TREELITE
        }

    elif model_name == "tpot":
        if not predictor.tpot_model:
            raise HTTPException(status_code=404, detail="TPOT model not available")

        return {
            "model_name": "tpot",
            "hash": predictor.tpot_model_hash or "initial",
            "last_retrain": predictor.last_retrain_time.isoformat() if predictor.last_retrain_time else None,
            "model_type": predictor.model_type.value,
            "use_treelite": settings.USE_TREELITE
        }

    else:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}. Use 'ttft' or 'tpot'")


@app.get("/model/{model_name}/info")
async def model_info(model_name: str):
    """Get model file information from active bundle including readiness status."""
    # Map model names to bundle file names
    bundle_file_mapping = {
        "ttft": TTFT_MODEL_FILENAME,
        "tpot": TPOT_MODEL_FILENAME,
        "ttft_scaler": TTFT_SCALER_FILENAME,
        "tpot_scaler": TPOT_SCALER_FILENAME,
        "ttft_treelite": TTFT_TREELITE_FILENAME,
        "tpot_treelite": TPOT_TREELITE_FILENAME,
        "ttft_conformal": TTFT_CONFORMAL_FILENAME,
        "tpot_conformal": TPOT_CONFORMAL_FILENAME,
    }

    if model_name not in bundle_file_mapping:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    bundle_file_name = bundle_file_mapping[model_name]

    # Get file path from active bundle
    active_bundle = predictor.bundle_manager.get_active_bundle()
    if not active_bundle or bundle_file_name not in active_bundle.manifest.files:
        # Bundle or file not ready yet
        return {
            "model_name": model_name,
            "ready": False,
            "status": "pending_training",
            "message": "Model will be available after training completes",
            "model_type": predictor.model_type.value,
            "expected_after_seconds": settings.RETRAINING_INTERVAL_SEC if predictor.last_retrain_time is None else 0,
            "quantile": predictor.quantile if model_name in ["ttft", "tpot", "ttft_treelite", "tpot_treelite", "ttft_conformal", "tpot_conformal"] else None
        }

    model_path = str(active_bundle.path / bundle_file_name)

    # Check if file exists (should always exist if in manifest)
    if not os.path.exists(model_path):
        # This shouldn't happen (file in manifest but not on disk)
        raise HTTPException(status_code=500, detail=f"Bundle integrity error: {bundle_file_name} missing")

    # File exists - return full info with ready=true
    stat = os.stat(model_path)
    last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    return {
        "model_name": model_name,
        "path": model_path,
        "ready": True,
        "status": "available",
        "size_bytes": stat.st_size,
        "last_modified": last_modified.isoformat(),
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile if model_name in ["ttft", "tpot", "ttft_treelite", "tpot_treelite", "ttft_conformal", "tpot_conformal"] else None
    }


@app.get("/model/{model_name}/download")
async def download_model(model_name: str):
    """Download a model file from the active bundle."""
    # Map model names to bundle file names
    bundle_file_mapping = {
        "ttft": TTFT_MODEL_FILENAME,
        "tpot": TPOT_MODEL_FILENAME,
        "ttft_scaler": TTFT_SCALER_FILENAME,
        "tpot_scaler": TPOT_SCALER_FILENAME,
        "ttft_treelite": TTFT_TREELITE_FILENAME,
        "tpot_treelite": TPOT_TREELITE_FILENAME,
        "ttft_conformal": TTFT_CONFORMAL_FILENAME,
        "tpot_conformal": TPOT_CONFORMAL_FILENAME,
    }

    if model_name not in bundle_file_mapping:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    bundle_file_name = bundle_file_mapping[model_name]

    # Get file path from active bundle
    active_bundle = predictor.bundle_manager.get_active_bundle()
    if not active_bundle or bundle_file_name not in active_bundle.manifest.files:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not available yet")

    model_path = str(active_bundle.path / bundle_file_name)

    if not os.path.exists(model_path):
        # Bundle integrity error
        raise HTTPException(status_code=500, detail=f"Bundle integrity error: {bundle_file_name} missing")

    # Return the file
    if model_name.endswith('_treelite'):
        filename = f"{model_name}.so"
        media_type = 'application/octet-stream'
    elif model_name.endswith('_conformal'):
        filename = f"{model_name}.json"
        media_type = 'application/json'
    else:
        filename = f"{model_name}.joblib"
        media_type = 'application/octet-stream'

    return FileResponse(
        model_path,
        media_type=media_type,
        filename=filename
    )


@app.get("/models/list")
async def list_models():
    """List all available models from the active bundle."""
    models = {}

    # Map of model names to bundle file names
    bundle_file_mapping = {
        "ttft": TTFT_MODEL_FILENAME,
        "tpot": TPOT_MODEL_FILENAME,
        "ttft_scaler": TTFT_SCALER_FILENAME,
        "tpot_scaler": TPOT_SCALER_FILENAME,
        "ttft_treelite": TTFT_TREELITE_FILENAME,
        "tpot_treelite": TPOT_TREELITE_FILENAME,
        "ttft_conformal": TTFT_CONFORMAL_FILENAME,
        "tpot_conformal": TPOT_CONFORMAL_FILENAME
    }

    # Get active bundle
    active_bundle = predictor.bundle_manager.get_active_bundle()

    for model_name, bundle_file_name in bundle_file_mapping.items():
        if active_bundle and bundle_file_name in active_bundle.manifest.files:
            model_path = active_bundle.path / bundle_file_name
            if model_path.exists():
                stat = os.stat(model_path)
                models[model_name] = {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "bundle_id": active_bundle.manifest.bundle_id[:8],
                    "checksum": active_bundle.manifest.files[bundle_file_name][:8]
                }
            else:
                models[model_name] = {
                    "exists": False,
                    "size_bytes": 0,
                    "last_modified": None,
                    "bundle_id": active_bundle.manifest.bundle_id[:8],
                    "checksum": None,
                    "error": "Bundle integrity error"
                }
        else:
            models[model_name] = {
                "exists": False,
                "size_bytes": 0,
                "last_modified": None
            }

    return {
        "models": models,
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile,
        "active_bundle": active_bundle.manifest.bundle_id[:8] if active_bundle else None,
        "bundle_state": active_bundle.manifest.state.value if active_bundle else None,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "evaluation_metrics": {
            "quantile_loss": "Lower is better",
            "coverage_percent": f"Target: {predictor.quantile*100:.1f}%",
            "violation_rate_percent": f"Target: {(1-predictor.quantile)*100:.1f}%"
        }
    }


# Bundle sync endpoints for prediction servers
@app.get("/bundle/current/info")
async def get_current_bundle_info():
    """
    Get information about the current active bundle.

    This endpoint allows prediction servers to check if a new bundle is available
    without downloading all files.
    """
    active_bundle = predictor.bundle_manager.get_active_bundle()

    if not active_bundle:
        raise HTTPException(status_code=404, detail="No active bundle available")

    return {
        "bundle_id": active_bundle.manifest.bundle_id,
        "bundle_id_short": active_bundle.manifest.bundle_id[:8],
        "state": active_bundle.manifest.state.value,
        "training_cycle": active_bundle.manifest.training_cycle,
        "model_type": active_bundle.manifest.model_type,
        "quantile": active_bundle.manifest.quantile,
        "use_treelite": active_bundle.manifest.use_treelite,
        "created_at": active_bundle.manifest.created_at,
        "training_samples": active_bundle.manifest.training_samples,
        "test_samples": active_bundle.manifest.test_samples,
        "files": {
            name: {
                "checksum": checksum[:8],
                "size_bytes": os.path.getsize(active_bundle.path / name) if (active_bundle.path / name).exists() else 0
            }
            for name, checksum in active_bundle.manifest.files.items()
        }
    }


@app.get("/bundle/{bundle_id}/manifest")
async def get_bundle_manifest(bundle_id: str):
    """
    Get the manifest for a specific bundle.

    Args:
        bundle_id: Full or short (8-char) bundle ID
    """
    # Load bundle from registry
    try:
        bundle = ModelBundle.load(predictor.bundle_manager.registry.bundle_dir, bundle_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Bundle {bundle_id} not found: {e}")

    return {
        "bundle_id": bundle.manifest.bundle_id,
        "state": bundle.manifest.state.value,
        "training_cycle": bundle.manifest.training_cycle,
        "model_type": bundle.manifest.model_type,
        "quantile": bundle.manifest.quantile,
        "use_treelite": bundle.manifest.use_treelite,
        "created_at": bundle.manifest.created_at.isoformat(),
        "files": bundle.manifest.files
    }


@app.get("/bundle/{bundle_id}/file/{file_name}")
async def download_bundle_file(bundle_id: str, file_name: str):
    """
    Download a specific file from a bundle.

    Args:
        bundle_id: Full or short (8-char) bundle ID
        file_name: Name of file in bundle (e.g., "ttft_model", "ttft_treelite")
    """
    # Load bundle from registry
    try:
        bundle = ModelBundle.load(predictor.bundle_manager.registry.bundle_dir, bundle_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Bundle {bundle_id} not found: {e}")

    if file_name not in bundle.manifest.files:
        raise HTTPException(status_code=404, detail=f"File {file_name} not found in bundle {bundle_id[:8]}")

    file_path = bundle.path / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {file_name} missing from bundle {bundle_id[:8]} (integrity error)")

    # Map file types to media types
    media_type_map = {
        TTFT_MODEL_FILENAME: "application/octet-stream",
        TPOT_MODEL_FILENAME: "application/octet-stream",
        TTFT_SCALER_FILENAME: "application/octet-stream",
        TPOT_SCALER_FILENAME: "application/octet-stream",
        TTFT_TREELITE_FILENAME: "application/x-sharedlib",
        TPOT_TREELITE_FILENAME: "application/x-sharedlib",
        TTFT_CONFORMAL_FILENAME: "application/json",
        TPOT_CONFORMAL_FILENAME: "application/json",
    }

    media_type = media_type_map.get(file_name, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=f"{file_name}.bin"
    )


# Add new API endpoints for LightGBM model exports
@app.get("/model/ttft/lgb/txt")
async def ttft_lgb_txt():
    """
    Download the TTFT LightGBM model as text format from active bundle.
    """
    if predictor.model_type != ModelType.LIGHTGBM:
        raise HTTPException(status_code=404, detail="TTFT model is not LightGBM")

    if not predictor.ttft_model:
        raise HTTPException(status_code=404, detail="TTFT model not available")

    # Get active bundle
    active_bundle = predictor.bundle_manager.get_active_bundle()
    if not active_bundle:
        raise HTTPException(status_code=404, detail="No active bundle available")

    # LightGBM text export filename (stored in bundle during training)
    txt_filename = "ttft_lgb.txt"
    txt_path = active_bundle.path / txt_filename

    if not txt_path.exists():
        raise HTTPException(status_code=404, detail="TTFT LightGBM text model not found in bundle")

    return FileResponse(
        str(txt_path),
        media_type='text/plain',
        filename='ttft_lgb_model.txt'
    )

@app.get("/model/tpot/lgb/txt")
async def tpot_lgb_txt():
    """
    Download the TPOT LightGBM model as text format from active bundle.
    """
    if predictor.model_type != ModelType.LIGHTGBM:
        raise HTTPException(status_code=404, detail="TPOT model is not LightGBM")

    if not predictor.tpot_model:
        raise HTTPException(status_code=404, detail="TPOT model not available")

    # Get active bundle
    active_bundle = predictor.bundle_manager.get_active_bundle()
    if not active_bundle:
        raise HTTPException(status_code=404, detail="No active bundle available")

    # LightGBM text export filename (stored in bundle during training)
    txt_filename = "tpot_lgb.txt"
    txt_path = active_bundle.path / txt_filename

    if not txt_path.exists():
        raise HTTPException(status_code=404, detail="TPOT LightGBM text model not found in bundle")

    return FileResponse(
        str(txt_path),
        media_type='text/plain',
        filename='tpot_lgb_model.txt'
    )

@app.get("/model/ttft/lgb/importances")
async def ttft_lgb_importances():
    """
    Get TTFT LightGBM feature importances as JSON from active bundle.
    """
    if predictor.model_type != ModelType.LIGHTGBM:
        raise HTTPException(status_code=404, detail="TTFT model is not LightGBM")

    if not predictor.ttft_model:
        raise HTTPException(status_code=404, detail="TTFT model not available")

    # Get active bundle
    active_bundle = predictor.bundle_manager.get_active_bundle()
    if not active_bundle:
        raise HTTPException(status_code=404, detail="No active bundle available")

    # LightGBM importances filename (stored in bundle during training)
    imp_filename = "ttft_importances.json"
    imp_path = active_bundle.path / imp_filename

    if not imp_path.exists():
        raise HTTPException(status_code=404, detail="TTFT LightGBM importances not found in bundle")

    with open(imp_path, 'r') as f:
        importances = json.load(f)

    return JSONResponse(content=importances)

@app.get("/model/tpot/lgb/importances")
async def tpot_lgb_importances():
    """
    Get TPOT LightGBM feature importances as JSON from active bundle.
    """
    if predictor.model_type != ModelType.LIGHTGBM:
        raise HTTPException(status_code=404, detail="TPOT model is not LightGBM")

    if not predictor.tpot_model:
        raise HTTPException(status_code=404, detail="TPOT model not available")

    # Get active bundle
    active_bundle = predictor.bundle_manager.get_active_bundle()
    if not active_bundle:
        raise HTTPException(status_code=404, detail="No active bundle available")

    # LightGBM importances filename (stored in bundle during training)
    imp_filename = "tpot_importances.json"
    imp_path = active_bundle.path / imp_filename

    if not imp_path.exists():
        raise HTTPException(status_code=404, detail="TPOT LightGBM importances not found in bundle")

    with open(imp_path, 'r') as f:
        importances = json.load(f)

    return JSONResponse(content=importances)

@app.get("/debug/prefix_distribution")
async def prefix_distribution():
    """
    Debug endpoint to analyze the relationship between prefix_cache_score and TTFT.
    This helps verify that the model is seeing the data it needs to learn.
    """
    all_samples = predictor._all_samples(predictor.ttft_data_buckets)
    if not all_samples:
        return {"error": "No training samples available"}
    
    prefix_scores = [s['prefix_cache_score'] for s in all_samples]
    ttfts = [s['actual_ttft_ms'] for s in all_samples]
    
    # Group by prefix score ranges
    ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    distribution = {}

    for low, high in ranges:
        # include the right edge only for the final bin so 1.0 is counted
        if high == 1.0:
            mask = [(low <= p <= high) for p in prefix_scores]
        else:
            mask = [(low <= p <  high) for p in prefix_scores]

        range_ttfts  = [t for t, m in zip(ttfts, mask) if m]
        range_prefix = [p for p, m in zip(prefix_scores, mask) if m]
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
        "correlation": float(np.corrcoef(prefix_scores, ttfts)[0, 1])
    }
    
    return {
        "overall_stats": overall,
        "distribution_by_prefix_range": distribution,
        "interpretation": {
            "correlation": "Negative correlation means higher prefix score → lower TTFT (good!)",
            "check_distribution": "All ranges should have samples. Empty ranges mean missing data.",
            "expected_pattern": "TTFT should decrease significantly as prefix score increases"
        }
    }

if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)
