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
import os
import shutil
import time
import logging
import threading
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Optional, List, Any
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

import joblib
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Try to import XGBoost; fall back if unavailable
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    import tl2cgen
    TL2CGEN_AVAILABLE = True
except ImportError:
    TL2CGEN_AVAILABLE = False
    logging.warning("TL2cgen not available. Install with: pip install tl2cgen")

try:
    from conformal_quantile import ConformalQuantilePredictor
    CONFORMAL_AVAILABLE = True
except ImportError:
    CONFORMAL_AVAILABLE = False
    logging.warning("ConformalQuantilePredictor not available. Check conformal_quantile.py exists.")


@dataclass(frozen=True)
class ModelBundle:
    """
    Immutable container for all models and predictors.

    This dataclass ensures atomic swapping of models - when we update models,
    we create a new bundle and swap the reference atomically. This allows
    lock-free reads during predictions while ensuring consistency.

    All models in a bundle are always consistent with each other (trained together).
    """
    # Core models (always present after initialization)
    ttft_model: Optional[Any]
    tpot_model: Optional[Any]

    # Scalers (only for Bayesian Ridge)
    ttft_scaler: Optional[Any]
    tpot_scaler: Optional[Any]

    # TreeLite predictors (compiled models for fast inference)
    ttft_predictor: Optional[Any]
    tpot_predictor: Optional[Any]

    # Conformal predictors (for TreeLite mode quantile adjustment)
    ttft_conformal: Optional[Any]
    tpot_conformal: Optional[Any]

    # Metadata
    model_type: str  # "bayesian_ridge", "xgboost", or "lightgbm"
    use_treelite: bool
    quantile: float
    last_load: Optional[datetime]

    def is_ready(self, model_type_enum) -> bool:
        """Check if all required models are loaded for the given model type."""
        if model_type_enum.value == "bayesian_ridge":
            return all([self.ttft_model, self.tpot_model, self.ttft_scaler, self.tpot_scaler])
        elif self.use_treelite:
            # TreeLite mode: prefer TreeLite models, but fall back to base models during bootstrap
            has_treelite = all([self.ttft_predictor, self.tpot_predictor])
            has_base_models = all([self.ttft_model, self.tpot_model])
            return has_treelite or has_base_models
        else:  # XGBoost or LightGBM without TreeLite
            return all([self.ttft_model, self.tpot_model])


class ModelType(str, Enum):
    BAYESIAN_RIDGE = "bayesian_ridge"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

class PredictSettings:
    """Configuration for the prediction server."""

    # Training server URL
    TRAINING_SERVER_URL: str = os.getenv("TRAINING_SERVER_URL", "http://training-service:8000")

    # Local model paths
    LOCAL_TTFT_MODEL_PATH: str = os.getenv("LOCAL_TTFT_MODEL_PATH", "/local_models/ttft.joblib")
    LOCAL_TPOT_MODEL_PATH: str = os.getenv("LOCAL_TPOT_MODEL_PATH", "/local_models/tpot.joblib")
    LOCAL_TTFT_SCALER_PATH: str = os.getenv("LOCAL_TTFT_SCALER_PATH", "/local_models/ttft_scaler.joblib")
    LOCAL_TPOT_SCALER_PATH: str = os.getenv("LOCAL_TPOT_SCALER_PATH", "/local_models/tpot_scaler.joblib")
    LOCAL_TTFT_TREELITE_PATH: str = os.getenv("LOCAL_TTFT_TREELITE_PATH", "/local_models/ttft_treelite.so")
    LOCAL_TPOT_TREELITE_PATH: str = os.getenv("LOCAL_TPOT_TREELITE_PATH", "/local_models/tpot_treelite.so")
    LOCAL_TTFT_CONFORMAL_PATH: str = os.getenv("LOCAL_TTFT_CONFORMAL_PATH", "/local_models/ttft_conformal.json")
    LOCAL_TPOT_CONFORMAL_PATH: str = os.getenv("LOCAL_TPOT_CONFORMAL_PATH", "/local_models/tpot_conformal.json")

    # Use TreeLite for inference (preferred for production)
    USE_TREELITE: bool = os.getenv("USE_TREELITE", "true").lower() == "true"

    # Sync interval and model type
    MODEL_SYNC_INTERVAL_SEC: int = int(os.getenv("MODEL_SYNC_INTERVAL_SEC", "10"))
    MODEL_TYPE: ModelType = ModelType(os.getenv("LATENCY_MODEL_TYPE", "xgboost"))
    
    # Quantile configuration (should match training server)
    QUANTILE_ALPHA: float = float(os.getenv("LATENCY_QUANTILE_ALPHA", "0.9"))  # p90 quantile

    # Server host/port
    HOST: str = os.getenv("PREDICT_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PREDICT_PORT", "8001"))

    # HTTP timeout
    HTTP_TIMEOUT: int = int(os.getenv("HTTP_TIMEOUT", "30"))


settings = PredictSettings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelSyncer:
    """Downloads models from a training server via HTTP."""

    def __init__(self):
        self._shutdown_event = threading.Event()
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_lock = threading.Lock()

        # Track model hashes to avoid unnecessary downloads
        self.ttft_hash: Optional[str] = None
        self.tpot_hash: Optional[str] = None

        # Ensure local directories
        for path in [
            settings.LOCAL_TTFT_MODEL_PATH,
            settings.LOCAL_TPOT_MODEL_PATH,
            settings.LOCAL_TTFT_SCALER_PATH,
            settings.LOCAL_TPOT_SCALER_PATH,
            settings.LOCAL_TTFT_TREELITE_PATH,
            settings.LOCAL_TPOT_TREELITE_PATH,
            settings.LOCAL_TTFT_CONFORMAL_PATH,
            settings.LOCAL_TPOT_CONFORMAL_PATH
        ]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def _check_model_hash(self, model_name: str) -> Optional[str]:
        """
        Check the current hash of a model on the training server.

        Args:
            model_name: Name of the model ("ttft" or "tpot")

        Returns:
            Current hash from server, or None if unavailable
        """
        try:
            hash_url = f"{settings.TRAINING_SERVER_URL}/model/{model_name}/hash"
            r = requests.get(hash_url, timeout=5)  # Short timeout for hash check
            if r.status_code == 200:
                data = r.json()
                return data.get("hash")
            else:
                logging.debug(f"Hash endpoint not available for {model_name} (status {r.status_code})")
                return None
        except requests.RequestException as e:
            logging.debug(f"Error checking hash for {model_name}: {e}")
            return None

    def _download_model_if_newer(self, name: str, dest: str) -> bool:
        """
        Download a model file only if it has changed (hash-based).

        For core models (ttft/tpot), uses hash comparison for efficiency.
        For other files, falls back to timestamp-based checking.
        """
        try:
            # For core models, use hash-based checking
            if name in ["ttft", "tpot"]:
                server_hash = self._check_model_hash(name)

                if server_hash:
                    # Get current local hash (thread-safe read)
                    with self._sync_lock:
                        local_hash = self.ttft_hash if name == "ttft" else self.tpot_hash

                    if server_hash == local_hash and local_hash is not None:
                        logging.debug(f"Model {name} unchanged (hash: {server_hash[:8]}...), skipping download")
                        return False

                    logging.info(f"Model {name} changed (old: {local_hash[:8] if local_hash else 'none'}..., new: {server_hash[:8]}...), downloading")
                else:
                    # Hash endpoint not available, fall back to timestamp check
                    logging.debug(f"Hash check unavailable for {name}, using timestamp fallback")

            # Download logic (used when hash indicates change, or for non-core models, or fallback)
            info_url = f"{settings.TRAINING_SERVER_URL}/model/{name}/info"
            r = requests.get(info_url, timeout=settings.HTTP_TIMEOUT)
            if r.status_code != 200:
                # Debug logging for conformal files
                if 'conformal' in name:
                    logging.debug(f"Model {name} not available on server (status {r.status_code})")
                return False

            info = r.json()
            mtime = info.get("last_modified")

            # Timestamp-based check for non-core models (conformal, treelite, scalers)
            # Skip for core models that already did hash checking above
            if name not in ["ttft", "tpot"]:
                if mtime and os.path.exists(dest):
                    server_time = datetime.fromisoformat(mtime.replace('Z', '+00:00'))
                    local_time = datetime.fromtimestamp(os.path.getmtime(dest), tz=timezone.utc)
                    if local_time >= server_time:
                        logging.debug(f"Model {name} is up-to-date (timestamp): {dest}")
                        return False

            # Perform download
            dl_url = f"{settings.TRAINING_SERVER_URL}/model/{name}/download"
            dl = requests.get(dl_url, timeout=settings.HTTP_TIMEOUT, stream=True)
            if dl.status_code != 200:
                logging.error(f"Failed download {name}: {dl.status_code}")
                return False

            tmp = dest + ".tmp"
            with open(tmp, 'wb') as f:
                for chunk in dl.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            if os.path.getsize(tmp) == 0:
                os.remove(tmp)
                return False

            # Atomic replace
            os.replace(tmp, dest)

            # Update local hash for core models (thread-safe with lock)
            if name in ["ttft", "tpot"] and server_hash:
                with self._sync_lock:
                    if name == "ttft":
                        self.ttft_hash = server_hash
                    elif name == "tpot":
                        self.tpot_hash = server_hash

            # Extra logging for conformal files
            if 'conformal' in name:
                logging.info(f"✓ Downloaded conformal file: {name} -> {dest} (size: {os.path.getsize(dest)} bytes)")
            else:
                logging.info(f"Downloaded {name} -> {dest}")
            return True

        except requests.RequestException as e:
            logging.error(f"Network error for {name}: {e}")
            return False
        except OSError as e:
            logging.error(f"Filesystem error for {name}: {e}")
            return False

    def sync_models(self) -> bool:
        """Sync all relevant models; returns True if any updated.

        Note: Downloads happen WITHOUT holding sync_lock to avoid blocking predictions.
        Only hash tracking is protected by the lock.
        """
        # Build list of files to sync (no lock needed for this)
        updated = False
        to_sync = [
            ("ttft", settings.LOCAL_TTFT_MODEL_PATH),
            ("tpot", settings.LOCAL_TPOT_MODEL_PATH),
        ]

        # Sync TreeLite models if enabled
        if settings.USE_TREELITE and TL2CGEN_AVAILABLE:
            to_sync += [
                ("ttft_treelite", settings.LOCAL_TTFT_TREELITE_PATH),
                ("tpot_treelite", settings.LOCAL_TPOT_TREELITE_PATH),
            ]
            # Also sync conformal calibration for TreeLite mode
            if CONFORMAL_AVAILABLE:
                to_sync += [
                    ("ttft_conformal", settings.LOCAL_TTFT_CONFORMAL_PATH),
                    ("tpot_conformal", settings.LOCAL_TPOT_CONFORMAL_PATH),
                ]

        # Sync scalers only for Bayesian Ridge
        if settings.MODEL_TYPE == ModelType.BAYESIAN_RIDGE:
            to_sync += [
                ("ttft_scaler", settings.LOCAL_TTFT_SCALER_PATH),
                ("tpot_scaler", settings.LOCAL_TPOT_SCALER_PATH),
            ]

        # Download models WITHOUT holding lock (allows predictions to continue)
        for name, path in to_sync:
            if self._download_model_if_newer(name, path):
                updated = True
        return updated

    def _sync_loop(self):
        while not self._shutdown_event.is_set():
            try:
                if self.sync_models():
                    predictor.load_models()
            except Exception as e:
                logging.error(f"Error in sync loop: {e}")
            self._shutdown_event.wait(timeout=settings.MODEL_SYNC_INTERVAL_SEC)
        logging.info("Model sync loop exited")

    def start(self):
        if self._sync_thread:
            return
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logging.info(f"Sync thread started (interval {settings.MODEL_SYNC_INTERVAL_SEC}s)")

    def shutdown(self):
        self._shutdown_event.set()
        if self._sync_thread:
            self._sync_thread.join()


class LightweightPredictor:
    """Handles inference using loaded quantile regression models."""

    def __init__(self):
        mt = settings.MODEL_TYPE
        self.prefix_buckets = 4

        # Add LightGBM fallback logic
        if mt == ModelType.XGBOOST and not XGBOOST_AVAILABLE:
            logging.warning("XGBoost not available. Falling back to Bayesian Ridge")
            mt = ModelType.BAYESIAN_RIDGE
        elif mt == ModelType.LIGHTGBM and not LIGHTGBM_AVAILABLE:
            logging.warning("LightGBM not available. Falling back to Bayesian Ridge")
            mt = ModelType.BAYESIAN_RIDGE

        self.model_type = mt
        self.quantile = settings.QUANTILE_ALPHA
        self.use_treelite = settings.USE_TREELITE and TL2CGEN_AVAILABLE and mt in [ModelType.XGBOOST, ModelType.LIGHTGBM]

        # Initialize with empty ModelBundle (will be populated by load_models)
        self.models = ModelBundle(
            ttft_model=None,
            tpot_model=None,
            ttft_scaler=None,
            tpot_scaler=None,
            ttft_predictor=None,
            tpot_predictor=None,
            ttft_conformal=None,
            tpot_conformal=None,
            model_type=self.model_type.value,
            use_treelite=self.use_treelite,
            quantile=self.quantile,
            last_load=None
        )

        # Lock only for model swapping (not for predictions!)
        self.lock = threading.RLock()
        logging.info(f"Predictor type: {self.model_type}, quantile: {self.quantile}, use_treelite: {self.use_treelite}")

    @property
    def is_ready(self) -> bool:
        # No lock needed - atomic read of bundle reference
        models = self.models
        return models.is_ready(self.model_type)

    def _prepare_features_with_interaction(self, df: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """
        Prepare features with interaction terms to match training server.
        
        Args:
            df: DataFrame with raw features
            model_type: 'ttft' or 'tpot'
        
        Returns:
            DataFrame with engineered features including interactions
        """
        if model_type == "ttft":
            # Create interaction: prefix score * input length
            df['effective_input_tokens'] = (1-df['prefix_cache_score']) * df['input_token_length']
            df['prefill_score_bucket'] = (
            (df['prefix_cache_score'].clip(0, 1) * self.prefix_buckets)
            .astype(int)
            .clip(upper=self.prefix_buckets - 1)
        )

            # TreeLite mode: keep as int32 (categorical encoder disabled during training)
            # Non-TreeLite mode: convert to categorical for XGBoost/LightGBM
            if not self.use_treelite:
                df['prefill_score_bucket'] = pd.Categorical(df['prefill_score_bucket'], categories=[0,1,2,3], ordered=True)
            else:
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

    def load_models(self) -> bool:
        """
        Load models from disk and swap them atomically.

        This method builds a new ModelBundle outside the lock, then swaps
        it atomically with a brief lock. This minimizes lock hold time and
        allows predictions to continue using the old bundle while loading.
        """
        try:
            # Load models WITHOUT holding lock (I/O can be slow)
            # Always load base XGBoost/LightGBM models first (needed for fallback)
            new_ttft = joblib.load(settings.LOCAL_TTFT_MODEL_PATH) if os.path.exists(settings.LOCAL_TTFT_MODEL_PATH) else None
            new_tpot = joblib.load(settings.LOCAL_TPOT_MODEL_PATH) if os.path.exists(settings.LOCAL_TPOT_MODEL_PATH) else None

            if self.model_type == ModelType.BAYESIAN_RIDGE:
                new_ttft_scaler = joblib.load(settings.LOCAL_TTFT_SCALER_PATH) if os.path.exists(settings.LOCAL_TTFT_SCALER_PATH) else None
                new_tpot_scaler = joblib.load(settings.LOCAL_TPOT_SCALER_PATH) if os.path.exists(settings.LOCAL_TPOT_SCALER_PATH) else None
            else:
                new_ttft_scaler = new_tpot_scaler = None

            # Load TreeLite models if enabled (for performance)
            new_ttft_predictor = None
            new_tpot_predictor = None
            new_ttft_conformal = None
            new_tpot_conformal = None

            if self.use_treelite:
                if os.path.exists(settings.LOCAL_TTFT_TREELITE_PATH):
                    # Use 1 thread for CPU-constrained environments (avoid thread pool overhead)
                    new_ttft_predictor = tl2cgen.Predictor(settings.LOCAL_TTFT_TREELITE_PATH, nthread=1)
                    logging.debug("TTFT TreeLite model loaded via TL2cgen")
                else:
                    logging.warning(f"TreeLite model not found: {settings.LOCAL_TTFT_TREELITE_PATH}")

                if os.path.exists(settings.LOCAL_TPOT_TREELITE_PATH):
                    # Use 1 thread for CPU-constrained environments (avoid thread pool overhead)
                    new_tpot_predictor = tl2cgen.Predictor(settings.LOCAL_TPOT_TREELITE_PATH, nthread=1)
                    logging.debug("TPOT TreeLite model loaded via TL2cgen")
                else:
                    logging.warning(f"TreeLite model not found: {settings.LOCAL_TPOT_TREELITE_PATH}")

                # Load conformal calibration for TreeLite mode
                if CONFORMAL_AVAILABLE:
                    import json
                    # Load TTFT conformal calibration
                    if os.path.exists(settings.LOCAL_TTFT_CONFORMAL_PATH):
                        try:
                            with open(settings.LOCAL_TTFT_CONFORMAL_PATH, 'r') as f:
                                ttft_conf_state = json.load(f)
                            new_ttft_conformal = ConformalQuantilePredictor.from_state(ttft_conf_state)
                            logging.debug(f"TTFT conformal calibration loaded ({len(ttft_conf_state.get('calibration_residuals', []))} samples)")
                        except Exception as e:
                            logging.error(f"Error loading TTFT conformal calibration: {e}")
                            new_ttft_conformal = None
                    else:
                        logging.warning(f"TTFT conformal calibration not found: {settings.LOCAL_TTFT_CONFORMAL_PATH}")
                        new_ttft_conformal = None

                    # Load TPOT conformal calibration
                    if os.path.exists(settings.LOCAL_TPOT_CONFORMAL_PATH):
                        try:
                            with open(settings.LOCAL_TPOT_CONFORMAL_PATH, 'r') as f:
                                tpot_conf_state = json.load(f)
                            new_tpot_conformal = ConformalQuantilePredictor.from_state(tpot_conf_state)
                            logging.debug(f"TPOT conformal calibration loaded ({len(tpot_conf_state.get('calibration_residuals', []))} samples)")
                        except Exception as e:
                            logging.error(f"Error loading TPOT conformal calibration: {e}")
                            new_tpot_conformal = None
                    else:
                        logging.warning(f"TPOT conformal calibration not found: {settings.LOCAL_TPOT_CONFORMAL_PATH}")
                        new_tpot_conformal = None

            # Build new bundle with all loaded models
            load_time = datetime.now(timezone.utc)
            new_bundle = ModelBundle(
                ttft_model=new_ttft or self.models.ttft_model,  # Keep old if load failed
                tpot_model=new_tpot or self.models.tpot_model,
                ttft_scaler=new_ttft_scaler or self.models.ttft_scaler,
                tpot_scaler=new_tpot_scaler or self.models.tpot_scaler,
                ttft_predictor=new_ttft_predictor or self.models.ttft_predictor,
                tpot_predictor=new_tpot_predictor or self.models.tpot_predictor,
                ttft_conformal=new_ttft_conformal or self.models.ttft_conformal,
                tpot_conformal=new_tpot_conformal or self.models.tpot_conformal,
                model_type=self.model_type.value,
                use_treelite=self.use_treelite,
                quantile=self.quantile,
                last_load=load_time
            )

            # Atomic swap with brief lock (just to prevent concurrent load_models calls)
            with self.lock:
                self.models = new_bundle

            # Log success (after lock released)
            if new_bundle.is_ready(self.model_type):
                if self.use_treelite:
                    if new_ttft_predictor and new_tpot_predictor:
                        has_conformal = new_ttft_conformal and new_tpot_conformal
                        logging.info(f"Models loaded: Using TreeLite compiled models (conformal calibration: {has_conformal})")
                    else:
                        logging.info(f"Models loaded: Using base XGBoost models (TreeLite models not yet available)")
                else:
                    logging.info(f"Models loaded successfully (mode: {self.model_type.value})")
                return True

            logging.warning("Models missing after load")
            return False

        except Exception as e:
            logging.error(f"Load error: {e}")
            return False

    def predict(self, features: dict) -> Tuple[float, float]:
        """
        Make quantile predictions using the loaded models.

        This method is completely lock-free! It atomically reads the model bundle
        at the start, then uses that consistent snapshot for the entire prediction.
        This allows maximum parallelism for predictions.
        """
        try:
            # Atomic read of model bundle (lock-free!)
            models = self.models

            # Check readiness
            if not models.is_ready(self.model_type):
                raise HTTPException(status_code=503, detail="Models not ready")

            # Validation
            required = ['kv_cache_percentage', 'input_token_length', 'num_request_waiting',
                       'num_request_running', 'num_tokens_generated', 'prefix_cache_score']
            for f in required:
                if f not in features:
                    raise ValueError(f"Missing required feature: {f}")
                if not isinstance(features[f], (int, float)):
                    raise ValueError(f"Invalid type for feature {f}: expected number")

            # Create raw DataFrames (without interaction)
            ttft_raw_data = {
                'kv_cache_percentage': features['kv_cache_percentage'],
                'input_token_length': features['input_token_length'],
                'num_request_waiting': features['num_request_waiting'],
                'num_request_running': features['num_request_running'],
                'prefix_cache_score': features['prefix_cache_score']
            }

            tpot_raw_data = {
                'kv_cache_percentage': features['kv_cache_percentage'],
                'input_token_length': features['input_token_length'],
                'num_request_waiting': features['num_request_waiting'],
                'num_request_running': features['num_request_running'],
                'num_tokens_generated': features['num_tokens_generated']
            }

            # TreeLite mode: use compiled models with conformal prediction
            # Fall back to base XGBoost models if TreeLite not available yet (bootstrap)
            if models.use_treelite and models.ttft_predictor and models.tpot_predictor:
                # For TreeLite single prediction, construct feature array directly (no pandas overhead)
                # TTFT features: kv_cache_percentage, input_token_length, num_request_waiting,
                #                num_request_running, prefix_cache_score, effective_input_tokens, prefill_score_bucket
                effective_input_tokens = (1 - ttft_raw_data['prefix_cache_score']) * ttft_raw_data['input_token_length']
                prefill_score_bucket = int(min(ttft_raw_data['prefix_cache_score'] * self.prefix_buckets, self.prefix_buckets - 1))

                ttft_features = np.array([[
                    ttft_raw_data['kv_cache_percentage'],
                    ttft_raw_data['input_token_length'],
                    ttft_raw_data['num_request_waiting'],
                    ttft_raw_data['num_request_running'],
                    ttft_raw_data['prefix_cache_score'],
                    effective_input_tokens,
                    prefill_score_bucket
                ]], dtype=np.float32)

                # TPOT features: kv_cache_percentage, input_token_length, num_request_waiting,
                #                num_request_running, num_tokens_generated
                tpot_features = np.array([[
                    tpot_raw_data['kv_cache_percentage'],
                    tpot_raw_data['input_token_length'],
                    tpot_raw_data['num_request_waiting'],
                    tpot_raw_data['num_request_running'],
                    tpot_raw_data['num_tokens_generated']
                ]], dtype=np.float32)

                # TL2cgen v1.0.0 requires DMatrix
                ttft_dmat = tl2cgen.DMatrix(ttft_features)
                tpot_dmat = tl2cgen.DMatrix(tpot_features)

                # Get mean predictions from TL2cgen
                ttft_pred_array = models.ttft_predictor.predict(ttft_dmat)
                tpot_pred_array = models.tpot_predictor.predict(tpot_dmat)

                # Extract scalar values (handles both 1D and 2D arrays)
                ttft_mean = float(np.ravel(ttft_pred_array)[0])
                tpot_mean = float(np.ravel(tpot_pred_array)[0])

                # Apply conformal correction if available, otherwise return mean (bootstrap phase)
                if models.ttft_conformal and models.tpot_conformal:
                    ttft_pred = models.ttft_conformal.conformalize(ttft_mean)
                    tpot_pred = models.tpot_conformal.conformalize(tpot_mean)
                else:
                    # Bootstrap phase: return mean predictions to allow training data collection
                    ttft_pred = ttft_mean
                    tpot_pred = tpot_mean

                return ttft_pred, tpot_pred

            elif self.model_type == ModelType.BAYESIAN_RIDGE:
                ttft_for_scale = df_ttft.drop(columns=['prefill_score_bucket'], errors='ignore')
                ttft_scaled = models.ttft_scaler.transform(ttft_for_scale)
                tpot_scaled = models.tpot_scaler.transform(df_tpot)

                ttft_pred_mean, ttft_std = models.ttft_model.predict(ttft_scaled, return_std=True)
                tpot_pred_mean, tpot_std = models.tpot_model.predict(tpot_scaled, return_std=True)

                std_factor = 1.28 if models.quantile == 0.9 else (2.0 if models.quantile == 0.95 else 0.674)
                ttft_pred = ttft_pred_mean[0] + std_factor * ttft_std[0]
                tpot_pred = tpot_pred_mean[0] + std_factor * tpot_std[0]

                return ttft_pred, tpot_pred

            elif self.model_type == ModelType.XGBOOST:
                # Convert categorical columns to int if present (XGBoost needs enable_categorical or numeric dtypes)
                df_ttft_numeric = df_ttft.copy()
                if 'prefill_score_bucket' in df_ttft_numeric.columns:
                    if df_ttft_numeric['prefill_score_bucket'].dtype.name == 'category':
                        df_ttft_numeric['prefill_score_bucket'] = df_ttft_numeric['prefill_score_bucket'].astype('int32')

                ttft_pred = models.ttft_model.predict(df_ttft_numeric)
                tpot_pred = models.tpot_model.predict(df_tpot)

                # Note: Base XGBoost models are native quantile regression (reg:quantileerror)
                # so predictions are already quantiles. No adjustment needed.
                return ttft_pred[0], tpot_pred[0]

            else:  # LightGBM
                ttft_pred = models.ttft_model.predict(df_ttft)
                tpot_pred = models.tpot_model.predict(df_tpot)

                # Note: Base LightGBM models are native quantile regression
                # so predictions are already quantiles. No adjustment needed.
                return ttft_pred[0], tpot_pred[0]

        except ValueError as ve:
            logging.warning(f"Client error in predict(): {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except HTTPException:
            raise
        except Exception as e:
            logging.error("Error in predict():", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during prediction")

    def predict_batch(self, features_list: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make batch quantile predictions using the loaded models.

        This method is completely lock-free! It atomically reads the model bundle
        at the start, then uses that consistent snapshot for the entire batch.
        This allows maximum parallelism for batch predictions.
        """
        batch_start = time.time()
        try:
            # Atomic read of model bundle (lock-free!)
            models = self.models
            batch_size = len(features_list)

            # Check readiness
            if not models.is_ready(self.model_type):
                raise HTTPException(status_code=503, detail="Models not ready")

            # Validation
            validation_start = time.time()
            required = ['kv_cache_percentage', 'input_token_length', 'num_request_waiting',
                       'num_request_running', 'num_tokens_generated', 'prefix_cache_score']
            for i, features in enumerate(features_list):
                for f in required:
                    if f not in features:
                        raise ValueError(f"Missing required feature '{f}' in request {i}")
                    if not isinstance(features[f], (int, float)):
                        raise ValueError(f"Invalid type for feature '{f}' in request {i}: expected number")
            validation_time = (time.time() - validation_start) * 1000

            # Create raw feature data (without interaction)
            prep_start = time.time()
            ttft_raw_data = []
            tpot_raw_data = []

            for features in features_list:
                ttft_raw_data.append({
                    'kv_cache_percentage': features['kv_cache_percentage'],
                    'input_token_length': features['input_token_length'],
                    'num_request_waiting': features['num_request_waiting'],
                    'num_request_running': features['num_request_running'],
                    'prefix_cache_score': features['prefix_cache_score']
                })

                tpot_raw_data.append({
                    'kv_cache_percentage': features['kv_cache_percentage'],
                    'input_token_length': features['input_token_length'],
                    'num_request_waiting': features['num_request_waiting'],
                    'num_request_running': features['num_request_running'],
                    'num_tokens_generated': features['num_tokens_generated']
                })
            prep_time = (time.time() - prep_start) * 1000

            # TreeLite mode: use compiled models with conformal prediction
            # Fall back to base XGBoost models if TreeLite not available yet (bootstrap)
            if models.use_treelite and models.ttft_predictor and models.tpot_predictor:
                # For TreeLite batch prediction, construct feature arrays using vectorized operations
                # This eliminates the Python loop and GIL contention bottleneck
                array_start = time.time()

                # Extract all feature values into numpy arrays in one pass (no Python loops!)
                # TTFT features: 7 columns
                ttft_features = np.column_stack([
                    np.array([d['kv_cache_percentage'] for d in ttft_raw_data], dtype=np.float32),
                    np.array([d['input_token_length'] for d in ttft_raw_data], dtype=np.float32),
                    np.array([d['num_request_waiting'] for d in ttft_raw_data], dtype=np.float32),
                    np.array([d['num_request_running'] for d in ttft_raw_data], dtype=np.float32),
                    np.array([d['prefix_cache_score'] for d in ttft_raw_data], dtype=np.float32),
                ])

                # Compute interaction term vectorized (entire column at once)
                effective_input_tokens = (1.0 - ttft_features[:, 4]) * ttft_features[:, 1]

                # Compute bucket vectorized (entire column at once)
                prefill_score_bucket = np.clip(
                    (ttft_features[:, 4] * self.prefix_buckets).astype(np.int32),
                    0,
                    self.prefix_buckets - 1
                )

                # Add computed columns to feature matrix
                ttft_features = np.column_stack([
                    ttft_features,
                    effective_input_tokens,
                    prefill_score_bucket
                ])

                # TPOT features: 5 columns (no interaction term)
                tpot_features = np.column_stack([
                    np.array([d['kv_cache_percentage'] for d in tpot_raw_data], dtype=np.float32),
                    np.array([d['input_token_length'] for d in tpot_raw_data], dtype=np.float32),
                    np.array([d['num_request_waiting'] for d in tpot_raw_data], dtype=np.float32),
                    np.array([d['num_request_running'] for d in tpot_raw_data], dtype=np.float32),
                    np.array([d['num_tokens_generated'] for d in tpot_raw_data], dtype=np.float32),
                ])

                array_time = (time.time() - array_start) * 1000

                # TL2cgen v1.0.0 requires DMatrix
                dmat_start = time.time()
                ttft_dmat = tl2cgen.DMatrix(ttft_features)
                tpot_dmat = tl2cgen.DMatrix(tpot_features)
                dmat_time = (time.time() - dmat_start) * 1000

                # Get batch mean predictions from TL2cgen
                predict_start = time.time()
                ttft_pred_array = models.ttft_predictor.predict(ttft_dmat)
                tpot_pred_array = models.tpot_predictor.predict(tpot_dmat)
                predict_time = (time.time() - predict_start) * 1000

                # Flatten to 1D array (handles both 1D and 2D output)
                flatten_start = time.time()
                ttft_mean_preds = np.ravel(ttft_pred_array)
                tpot_mean_preds = np.ravel(tpot_pred_array)
                flatten_time = (time.time() - flatten_start) * 1000

                # Apply conformal correction if available, otherwise return mean (bootstrap phase)
                conformal_start = time.time()
                if models.ttft_conformal and models.tpot_conformal:
                    # Use vectorized batch conformalization (much faster than loop)
                    ttft_pred = models.ttft_conformal.conformalize_batch(ttft_mean_preds)
                    tpot_pred = models.tpot_conformal.conformalize_batch(tpot_mean_preds)
                else:
                    # Bootstrap phase: return mean predictions to allow training data collection
                    ttft_pred = ttft_mean_preds
                    tpot_pred = tpot_mean_preds
                conformal_time = (time.time() - conformal_start) * 1000

                total_time = (time.time() - batch_start) * 1000

                # Log detailed timing for batches >= 10 (potential bottleneck detection)
                if batch_size >= 10:
                    logging.info(
                        f"Batch prediction timing (size={batch_size}): "
                        f"total={total_time:.1f}ms, "
                        f"validation={validation_time:.1f}ms, "
                        f"prep={prep_time:.1f}ms, "
                        f"array_build={array_time:.1f}ms, "
                        f"dmat_create={dmat_time:.1f}ms, "
                        f"treelite_predict={predict_time:.1f}ms, "
                        f"flatten={flatten_time:.1f}ms, "
                        f"conformal={conformal_time:.1f}ms"
                    )

                return ttft_pred, tpot_pred

            # For non-TreeLite paths (Bayesian Ridge, XGBoost/LightGBM without TreeLite),
            # prepare DataFrames with feature engineering
            df_ttft_raw = pd.DataFrame(ttft_raw_data)
            df_ttft_batch = self._prepare_features_with_interaction(df_ttft_raw, "ttft")

            df_tpot_raw = pd.DataFrame(tpot_raw_data)
            df_tpot_batch = self._prepare_features_with_interaction(df_tpot_raw, "tpot")

            if self.model_type == ModelType.BAYESIAN_RIDGE:
                ttft_for_scale = df_ttft_batch.drop(columns=['prefill_score_bucket'], errors='ignore')
                ttft_scaled = models.ttft_scaler.transform(ttft_for_scale)
                tpot_scaled = models.tpot_scaler.transform(df_tpot_batch)

                ttft_pred_mean, ttft_std = models.ttft_model.predict(ttft_scaled, return_std=True)
                tpot_pred_mean, tpot_std = models.tpot_model.predict(tpot_scaled, return_std=True)

                std_factor = 1.28 if models.quantile == 0.9 else (2.0 if models.quantile == 0.95 else 0.674)
                ttft_pred = ttft_pred_mean + std_factor * ttft_std
                tpot_pred = tpot_pred_mean + std_factor * tpot_std

                return ttft_pred, tpot_pred

            elif self.model_type == ModelType.XGBOOST:
                # Convert categorical columns to int if present (XGBoost needs enable_categorical or numeric dtypes)
                df_ttft_batch_numeric = df_ttft_batch.copy()
                if 'prefill_score_bucket' in df_ttft_batch_numeric.columns:
                    if df_ttft_batch_numeric['prefill_score_bucket'].dtype.name == 'category':
                        df_ttft_batch_numeric['prefill_score_bucket'] = df_ttft_batch_numeric['prefill_score_bucket'].astype('int32')

                ttft_pred = models.ttft_model.predict(df_ttft_batch_numeric)
                tpot_pred = models.tpot_model.predict(df_tpot_batch)

                # Note: Base XGBoost models are native quantile regression (reg:quantileerror)
                # so predictions are already quantiles. No adjustment needed.
                return ttft_pred, tpot_pred

            else:  # LightGBM
                ttft_pred = models.ttft_model.predict(df_ttft_batch)
                tpot_pred = models.tpot_model.predict(df_tpot_batch)

                # Note: Base LightGBM models are native quantile regression
                # so predictions are already quantiles. No adjustment needed.
                return ttft_pred, tpot_pred

        except ValueError as ve:
            logging.warning(f"Client error in predict_batch(): {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except HTTPException:
            raise
        except Exception as e:
            logging.error("Error in predict_batch():", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during batch prediction")


# Instantiate
model_syncer = ModelSyncer()
predictor = LightweightPredictor()

# Thread pool for running sync predictions in async context
# Use 4 workers to avoid context switching overhead on 1-core pods
# The async event loop will queue requests, not the thread pool
prediction_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="prediction")

# FastAPI app
app = FastAPI(
    title="HTTP-based Quantile Latency Predictor",
    description="A prediction service that downloads quantile regression models from training server via HTTP.",
    version="1.0.0"
)


# Pydantic models
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
    predicted_at: datetime
    model_type: str = Field(..., description="Type of model used for prediction")
    quantile: float = Field(..., description="Quantile being predicted")
    last_model_load: Optional[datetime]


class StatusResponse(BaseModel):
    is_ready: bool
    model_type: str
    quantile: float = Field(..., description="Quantile being predicted")
    last_model_load: Optional[datetime]
    training_server_url: str
    models_exist: dict


class BulkPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(..., min_items=1, max_items=10000, description="List of prediction requests (max 10000)")

class BulkPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of prediction responses")
    total_requests: int = Field(..., description="Total number of requests processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")

class BulkPredictionError(BaseModel):
    index: int = Field(..., description="Index of the failed request in the original batch")
    error: str = Field(..., description="Error message")
    request: PredictionRequest = Field(..., description="The original request that failed")

class BulkPredictionResponseWithErrors(BaseModel):
    predictions: List[Optional[PredictionResponse]] = Field(..., description="List of prediction responses (None for failed predictions)")
    errors: List[BulkPredictionError] = Field(..., description="List of errors for failed predictions")
    total_requests: int = Field(..., description="Total number of requests processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


# API endpoints

@app.get("/status", response_model=StatusResponse)
async def status_endpoint():
    """Get server status and model information."""
    # Atomic read of model bundle
    models = predictor.models

    models_exist = {
        "ttft_model": os.path.exists(settings.LOCAL_TTFT_MODEL_PATH),
        "tpot_model": os.path.exists(settings.LOCAL_TPOT_MODEL_PATH),
    }

    if predictor.model_type == ModelType.BAYESIAN_RIDGE:
        models_exist.update({
            "ttft_scaler": os.path.exists(settings.LOCAL_TTFT_SCALER_PATH),
            "tpot_scaler": os.path.exists(settings.LOCAL_TPOT_SCALER_PATH),
        })

    return StatusResponse(
        is_ready=predictor.is_ready,
        model_type=predictor.model_type.value,
        quantile=predictor.quantile,
        last_model_load=models.last_load,
        training_server_url=settings.TRAINING_SERVER_URL,
        models_exist=models_exist
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    """Make quantile latency predictions."""
    start_time = time.time()
    try:
        # Direct synchronous call - now optimized with no pandas overhead
        predict_start = time.time()
        ttft_pred, tpot_pred = predictor.predict(request.model_dump())
        predict_time = (time.time() - predict_start) * 1000

        # Log slow predictions for debugging (50ms threshold)
        if predict_time > 50:
            logging.warning(f"Slow prediction: {predict_time:.1f}ms")

        # Ensure non-negative predictions
        ttft_pred = max(0, ttft_pred)
        tpot_pred = max(0, tpot_pred)

        # Atomic read for last_load timestamp
        models = predictor.models

        total_time = (time.time() - start_time) * 1000
        if total_time > 100:
            logging.warning(f"Slow endpoint total: {total_time:.1f}ms (predict: {predict_time:.1f}ms)")

        return PredictionResponse(
            ttft_ms=ttft_pred,
            tpot_ms=tpot_pred,
            predicted_at=datetime.now(timezone.utc),
            model_type=predictor.model_type.value,
            quantile=predictor.quantile,
            last_model_load=models.last_load
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction")


@app.post("/predict/bulk/strict", response_model=BulkPredictionResponse)
def predict_bulk_strict_endpoint(request: BulkPredictionRequest):
    """Make bulk quantile latency predictions using batch processing (fails on any single error)."""
    start_time = time.time()

    try:
        # Convert all requests to dict format
        features_list = [pred_request.model_dump() for pred_request in request.requests]

        # Direct synchronous batch prediction
        ttft_preds, tpot_preds = predictor.predict_batch(features_list)

        # Atomic read for metadata
        models = predictor.models

        # Build response list
        predictions = []
        current_time = datetime.now(timezone.utc)

        for i in range(len(request.requests)):
            # Ensure non-negative predictions
            ttft_pred = max(0, ttft_preds[i])
            tpot_pred = max(0, tpot_preds[i])

            prediction_response = PredictionResponse(
                ttft_ms=ttft_pred,
                tpot_ms=tpot_pred,
                predicted_at=current_time,
                model_type=predictor.model_type.value,
                quantile=predictor.quantile,
                last_model_load=models.last_load
            )

            predictions.append(prediction_response)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BulkPredictionResponse(
            predictions=predictions,
            total_requests=len(request.requests),
            successful_predictions=len(predictions),
            failed_predictions=0,
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Bulk prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Bulk prediction failed")


@app.post("/predict/bulk", response_model=BulkPredictionResponseWithErrors)
def predict_bulk_endpoint(request: BulkPredictionRequest):
    """Make bulk quantile latency predictions using batch processing with error handling."""
    start_time = time.time()

    # Separate valid and invalid requests
    valid_requests = []
    valid_indices = []
    errors = []

    # Pre-validate all requests
    required = ['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'num_tokens_generated', 'prefix_cache_score']

    for i, pred_request in enumerate(request.requests):
        try:
            features = pred_request.model_dump()
            # Validate features
            for f in required:
                if f not in features:
                    raise ValueError(f"Missing required feature: {f}")
                if not isinstance(features[f], (int, float)):
                    raise ValueError(f"Invalid type for feature {f}: expected number")

            valid_requests.append(features)
            valid_indices.append(i)

        except Exception as e:
            errors.append(BulkPredictionError(
                index=i,
                error=str(e),
                request=pred_request
            ))

    # Initialize predictions list with None values
    predictions = [None] * len(request.requests)
    successful_count = len(valid_requests)
    failed_count = len(errors)

    # Process valid requests in batch if any exist
    if valid_requests:
        try:
            # Direct synchronous batch prediction
            ttft_preds, tpot_preds = predictor.predict_batch(valid_requests)

            # Atomic read for metadata
            models = predictor.models
            current_time = datetime.now(timezone.utc)

            # Fill in predictions for valid requests
            for batch_idx, original_idx in enumerate(valid_indices):
                # Ensure non-negative predictions
                ttft_pred = max(0, ttft_preds[batch_idx])
                tpot_pred = max(0, tpot_preds[batch_idx])

                prediction_response = PredictionResponse(
                    ttft_ms=ttft_pred,
                    tpot_ms=tpot_pred,
                    predicted_at=current_time,
                    model_type=predictor.model_type.value,
                    quantile=predictor.quantile,
                    last_model_load=models.last_load
                )
                
                predictions[original_idx] = prediction_response
                
        except Exception as e:
            # If batch prediction fails, mark all valid requests as failed
            for original_idx in valid_indices:
                errors.append(BulkPredictionError(
                    index=original_idx,
                    error=f"Batch prediction error: {str(e)}",
                    request=request.requests[original_idx]
                ))
                predictions[original_idx] = None
            
            successful_count = 0
            failed_count = len(request.requests)
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return BulkPredictionResponseWithErrors(
        predictions=predictions,
        errors=errors,
        total_requests=len(request.requests),
        successful_predictions=successful_count,
        failed_predictions=failed_count,
        processing_time_ms=processing_time_ms
    )

@app.post("/reload")
async def reload_models():
    """Manually trigger model reload."""
    try:
        # First sync from training server
        synced = model_syncer.sync_models()
        
        # Then load models
        loaded = predictor.load_models()
        
        return {
            "synced": synced,
            "loaded": loaded,
            "is_ready": predictor.is_ready,
            "model_type": predictor.model_type.value,
            "quantile": predictor.quantile,
            "last_load_time": predictor.models.last_load
        }
    except Exception as e:
        logging.error(f"Error reloading models: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")

@app.get("/calibration/stats")
async def get_calibration_stats():
    """Get conformal calibration statistics (TreeLite mode only)."""
    if not predictor.use_treelite:
        return {
            "message": "Conformal calibration only used in TreeLite mode",
            "use_treelite": False,
            "model_type": predictor.model_type.value
        }

    # Atomic read of model bundle
    models = predictor.models

    stats = {
        "use_treelite": True,
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile,
        "ttft_conformal": None,
        "tpot_conformal": None
    }

    if models.ttft_conformal:
        stats["ttft_conformal"] = {
            "calibration_samples": len(models.ttft_conformal.calibration_residuals),
            "quantile_adjustment_ms": models.ttft_conformal._cached_quantile_value if not models.ttft_conformal._cache_dirty else None,
            "target_quantile": models.ttft_conformal.quantile,
            "max_calibration_samples": models.ttft_conformal.max_calibration_samples
        }
    else:
        stats["ttft_conformal"] = {"error": "TTFT conformal calibration not loaded"}

    if models.tpot_conformal:
        stats["tpot_conformal"] = {
            "calibration_samples": len(models.tpot_conformal.calibration_residuals),
            "quantile_adjustment_ms": models.tpot_conformal._cached_quantile_value if not models.tpot_conformal._cache_dirty else None,
            "target_quantile": models.tpot_conformal.quantile,
            "max_calibration_samples": models.tpot_conformal.max_calibration_samples
        }
    else:
        stats["tpot_conformal"] = {"error": "TPOT conformal calibration not loaded"}

    return stats


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "http-based-quantile-latency-predictor"}


@app.get("/readyz", status_code=status.HTTP_200_OK)
async def readiness_check():
    """Readiness check endpoint."""
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Models are not ready"
        )
    return {
        "status": "ready", 
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile
    }


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "message": "HTTP-based Quantile Latency Predictor is running",
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile,
        "description": f"Predicting {predictor.quantile:.0%} quantile for TTFT and TPOT latencies",
        "is_ready": predictor.is_ready,
        "sync_interval": settings.MODEL_SYNC_INTERVAL_SEC,
        "training_server": settings.TRAINING_SERVER_URL
    }


@app.on_event("startup")
async def startup():
    logging.info("Starting up...")
    # initial sync & load
    model_syncer.sync_models()
    predictor.load_models()
    model_syncer.start()

@app.on_event("shutdown")
async def shutdown():
    logging.info("Shutting down...")
    model_syncer.shutdown()
    # Shutdown thread pool gracefully
    prediction_executor.shutdown(wait=True, cancel_futures=False)


if __name__ == "__main__":
    uvicorn.run("__main__:app", host=settings.HOST, port=settings.PORT, reload=True)
