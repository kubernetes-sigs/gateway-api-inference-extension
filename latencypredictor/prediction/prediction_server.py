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
import time
import logging
import threading
import requests
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Optional, List, Any
from enum import Enum

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

    # Bundle cache directory
    BUNDLE_CACHE_DIR: str = os.getenv("BUNDLE_CACHE_DIR", "/local_models/bundles")

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

# Configure logging level from environment variable (default: INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")


class ModelSyncer:
    """Downloads models from a training server via HTTP using bundle-aware syncing."""

    def __init__(self):
        self._shutdown_event = threading.Event()
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_lock = threading.Lock()

        # Track current bundle ID to avoid re-downloading same bundle
        self.current_bundle_id: Optional[str] = None

        # Cache bundle metadata (training_samples, test_samples, etc.)
        self.current_bundle_info: Optional[dict] = None

        # Bundle cache directory
        self.bundle_cache_dir = settings.BUNDLE_CACHE_DIR
        os.makedirs(self.bundle_cache_dir, exist_ok=True)

    def sync_bundle(self) -> bool:
        """
        Sync models using bundle-aware approach.

        This eliminates race conditions by downloading all model files atomically
        from the same bundle.

        Returns:
            True if a new bundle was downloaded, False otherwise
        """
        try:
            # 1. Get current bundle info from training server
            bundle_info_url = f"{settings.TRAINING_SERVER_URL}/bundle/current/info"
            r = requests.get(bundle_info_url, timeout=10)

            if r.status_code != 200:
                logging.error(f"Bundle info not available (status {r.status_code}) - training server must support bundle system")
                raise Exception(f"Training server does not support bundle system (status {r.status_code})")

            bundle_info = r.json()
            server_bundle_id = bundle_info["bundle_id"]

            # 2. Check if we already have this bundle
            if server_bundle_id == self.current_bundle_id:
                logging.debug(f"Bundle {server_bundle_id[:8]} already synced")
                return False

            logging.info(f"New bundle available: {server_bundle_id[:8]} (current: {self.current_bundle_id[:8] if self.current_bundle_id else 'none'})")

            # 3. Create bundle directory in cache
            bundle_dir = os.path.join(self.bundle_cache_dir, server_bundle_id[:8])
            os.makedirs(bundle_dir, exist_ok=True)

            # 4. Download all files from bundle
            files_downloaded = 0
            for file_name, file_info in bundle_info["files"].items():
                file_url = f"{settings.TRAINING_SERVER_URL}/bundle/{server_bundle_id}/file/{file_name}"
                file_dest = os.path.join(bundle_dir, file_name)

                # Skip if already downloaded (idempotent)
                if os.path.exists(file_dest):
                    expected_size = file_info["size_bytes"]
                    actual_size = os.path.getsize(file_dest)
                    if actual_size == expected_size:
                        logging.debug(f"  ✓ {file_name} already cached ({actual_size} bytes)")
                        continue

                # Download file
                try:
                    file_r = requests.get(file_url, timeout=settings.HTTP_TIMEOUT, stream=True)
                    if file_r.status_code == 200:
                        # Write to temp file first, then atomic rename
                        temp_dest = file_dest + ".tmp"
                        with open(temp_dest, 'wb') as f:
                            for chunk in file_r.iter_content(8192):
                                if chunk:
                                    f.write(chunk)

                        # Atomic rename
                        os.replace(temp_dest, file_dest)
                        files_downloaded += 1
                        logging.debug(f"  ✓ Downloaded {file_name} ({os.path.getsize(file_dest)} bytes)")
                    else:
                        logging.warning(f"  ✗ Failed to download {file_name}: HTTP {file_r.status_code}")
                except Exception as e:
                    logging.error(f"  ✗ Error downloading {file_name}: {e}")

            # 5. Update current bundle ID and cache bundle info
            with self._sync_lock:
                self.current_bundle_id = server_bundle_id
                self.current_bundle_info = bundle_info  # Cache full bundle info including training_samples

            logging.info(f"✓ Bundle {server_bundle_id[:8]} synced ({files_downloaded} files downloaded, {len(bundle_info['files']) - files_downloaded} cached)")
            return True

        except requests.RequestException as e:
            logging.debug(f"Bundle sync failed (network): {e}")
            return False
        except Exception as e:
            logging.error(f"Bundle sync failed: {e}", exc_info=True)
            return False

    def get_bundle_path(self, bundle_id: Optional[str] = None) -> Optional[str]:
        """
        Get the path to a bundle directory.

        Args:
            bundle_id: Bundle ID (uses current if None)

        Returns:
            Path to bundle directory, or None if not available
        """
        if bundle_id is None:
            bundle_id = self.current_bundle_id

        if bundle_id is None:
            return None

        bundle_dir = os.path.join(self.bundle_cache_dir, bundle_id[:8])
        if os.path.exists(bundle_dir):
            return bundle_dir

        return None

    def _sync_loop(self):
        while not self._shutdown_event.is_set():
            try:
                # Bundle-aware sync (eliminates race conditions)
                bundle_synced = self.sync_bundle()

                if bundle_synced:
                    # Bundle synced successfully - load models from bundle
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

            # Dtype handling for prefill_score_bucket:
            # - XGBoost in TreeLite mode: int32 (TreeLite doesn't support categorical)
            # - LightGBM (all modes): categorical (required for consistency with training)
            # - XGBoost native quantile mode: categorical (enable_categorical=True)
            if self.use_treelite and self.model_type == ModelType.XGBOOST:
                # XGBoost TreeLite mode: convert to int32
                df['prefill_score_bucket'] = df['prefill_score_bucket'].astype('int32')
            else:
                # All other cases: keep as categorical
                df['prefill_score_bucket'] = pd.Categorical(df['prefill_score_bucket'], categories=[0,1,2,3], ordered=True)

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

        BUNDLE-ONLY: All files loaded from bundle directory (atomic guarantee).
        """
        try:
            # Get bundle path
            bundle_path = model_syncer.get_bundle_path()

            if not bundle_path:
                logging.error("No bundle available - cannot load models")
                return False

            # Load from bundle directory (all files guaranteed consistent)
            logging.debug(f"Loading models from bundle: {bundle_path}")
            ttft_model_path = os.path.join(bundle_path, TTFT_MODEL_FILENAME)
            tpot_model_path = os.path.join(bundle_path, TPOT_MODEL_FILENAME)
            ttft_scaler_path = os.path.join(bundle_path, TTFT_SCALER_FILENAME)
            tpot_scaler_path = os.path.join(bundle_path, TPOT_SCALER_FILENAME)
            ttft_treelite_path = os.path.join(bundle_path, TTFT_TREELITE_FILENAME)
            tpot_treelite_path = os.path.join(bundle_path, TPOT_TREELITE_FILENAME)
            ttft_conformal_path = os.path.join(bundle_path, TTFT_CONFORMAL_FILENAME)
            tpot_conformal_path = os.path.join(bundle_path, TPOT_CONFORMAL_FILENAME)

            # Load models WITHOUT holding lock (I/O can be slow)
            # Always load base XGBoost/LightGBM models first (needed for fallback)
            new_ttft = joblib.load(ttft_model_path) if os.path.exists(ttft_model_path) else None
            new_tpot = joblib.load(tpot_model_path) if os.path.exists(tpot_model_path) else None

            if self.model_type == ModelType.BAYESIAN_RIDGE:
                new_ttft_scaler = joblib.load(ttft_scaler_path) if os.path.exists(ttft_scaler_path) else None
                new_tpot_scaler = joblib.load(tpot_scaler_path) if os.path.exists(tpot_scaler_path) else None
            else:
                new_ttft_scaler = new_tpot_scaler = None

            # Load TreeLite models if enabled (for performance)
            new_ttft_predictor = None
            new_tpot_predictor = None
            new_ttft_conformal = None
            new_tpot_conformal = None

            if self.use_treelite:
                # Load TTFT TreeLite model
                if os.path.exists(ttft_treelite_path):
                    # IMPORTANT: Always reload TreeLite predictor (tl2cgen caches model in memory)
                    # Even if we have an old predictor, reload from disk to get updated model
                    new_ttft_predictor = tl2cgen.Predictor(ttft_treelite_path, nthread=1)
                    logging.debug(f"✓ TTFT TreeLite model loaded from bundle: {ttft_treelite_path}")

                    # Smoke test: verify predictor works with test input
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        test_input = np.array([[0.5, 400, 4, 1, 0.7, 120, 2]], dtype=np.float32)
                        test_pred = float(np.ravel(new_ttft_predictor.predict(tl2cgen.DMatrix(test_input)))[0])
                        logging.debug(f"  TTFT smoke test: input_len=400 → {test_pred:.2f}ms (expected ~974ms for good model)")
                else:
                    # TreeLite not available yet - keep old predictor if we have one (bootstrap phase)
                    new_ttft_predictor = self.models.ttft_predictor if self.models.ttft_predictor else None
                    if not new_ttft_predictor:
                        logging.warning(f"TreeLite model not found: {ttft_treelite_path}")
                    else:
                        logging.info("Keeping existing TTFT TreeLite predictor (file not found)")

                # Load TPOT TreeLite model
                if os.path.exists(tpot_treelite_path):
                    # IMPORTANT: Always reload TreeLite predictor (tl2cgen caches model in memory)
                    # Even if we have an old predictor, reload from disk to get updated model
                    new_tpot_predictor = tl2cgen.Predictor(tpot_treelite_path, nthread=1)
                    logging.debug(f"✓ TPOT TreeLite model loaded from bundle: {tpot_treelite_path}")

                    # Smoke test: verify predictor works with test input
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        test_input = np.array([[0.5, 400, 4, 1, 10]], dtype=np.float32)
                        test_pred = float(np.ravel(new_tpot_predictor.predict(tl2cgen.DMatrix(test_input)))[0])
                        logging.debug(f"  TPOT smoke test: input_len=400 → {test_pred:.2f}ms")
                else:
                    # TreeLite not available yet - keep old predictor if we have one (bootstrap phase)
                    new_tpot_predictor = self.models.tpot_predictor if self.models.tpot_predictor else None
                    if not new_tpot_predictor:
                        logging.warning(f"TreeLite model not found: {tpot_treelite_path}")
                    else:
                        logging.info("Keeping existing TPOT TreeLite predictor (file not found)")

                # Load conformal calibration for TreeLite mode
                if CONFORMAL_AVAILABLE:
                    import json
                    # Load TTFT conformal calibration
                    # Get training_cycle from bundle metadata for intelligent logging
                    training_cycle = 0
                    if model_syncer.current_bundle_info:
                        training_cycle = model_syncer.current_bundle_info.get("training_cycle", 0)

                    if os.path.exists(ttft_conformal_path):
                        try:
                            with open(ttft_conformal_path, 'r') as f:
                                ttft_conf_state = json.load(f)
                            new_ttft_conformal = ConformalQuantilePredictor.from_state(ttft_conf_state)
                            logging.debug(f"TTFT conformal calibration loaded ({len(ttft_conf_state.get('calibration_residuals', []))} samples)")
                        except Exception as e:
                            logging.error(f"Error loading TTFT conformal calibration: {e}")
                            new_ttft_conformal = None
                    else:
                        # Training_cycle-aware logging: INFO for default models (cycle=0), WARNING for trained models (cycle≥1)
                        if training_cycle == 0:
                            logging.info(f"TTFT conformal calibration not yet available (training_cycle=0, default models - waiting for first training run)")
                        else:
                            logging.warning(f"TTFT conformal calibration missing despite training_cycle={training_cycle} - predictions may have reduced quality")
                        new_ttft_conformal = None

                    # Load TPOT conformal calibration
                    if os.path.exists(tpot_conformal_path):
                        try:
                            with open(tpot_conformal_path, 'r') as f:
                                tpot_conf_state = json.load(f)
                            new_tpot_conformal = ConformalQuantilePredictor.from_state(tpot_conf_state)
                            logging.debug(f"TPOT conformal calibration loaded ({len(tpot_conf_state.get('calibration_residuals', []))} samples)")
                        except Exception as e:
                            logging.error(f"Error loading TPOT conformal calibration: {e}")
                            new_tpot_conformal = None
                    else:
                        # Training_cycle-aware logging: INFO for default models (cycle=0), WARNING for trained models (cycle≥1)
                        if training_cycle == 0:
                            logging.info(f"TPOT conformal calibration not yet available (training_cycle=0, default models - waiting for first training run)")
                        else:
                            logging.warning(f"TPOT conformal calibration missing despite training_cycle={training_cycle} - predictions may have reduced quality")
                        new_tpot_conformal = None

            # Build new bundle with all loaded models
            load_time = datetime.now(timezone.utc)

            # Track predictor changes for debugging
            old_ttft_id = id(self.models.ttft_predictor) if self.models.ttft_predictor else None
            old_tpot_id = id(self.models.tpot_predictor) if self.models.tpot_predictor else None
            new_ttft_id = id(new_ttft_predictor) if new_ttft_predictor else None
            new_tpot_id = id(new_tpot_predictor) if new_tpot_predictor else None

            final_ttft_predictor = new_ttft_predictor or self.models.ttft_predictor
            final_tpot_predictor = new_tpot_predictor or self.models.tpot_predictor

            # Log predictor object changes (DEBUG only - technical details)
            if self.use_treelite and logging.getLogger().isEnabledFor(logging.DEBUG):
                if new_ttft_predictor and old_ttft_id != new_ttft_id:
                    logging.debug(f"TTFT predictor object CHANGED: {old_ttft_id} → {new_ttft_id}")
                elif new_ttft_predictor is None and old_ttft_id:
                    logging.debug(f"TTFT predictor object REUSED (new load failed): {old_ttft_id}")

                if new_tpot_predictor and old_tpot_id != new_tpot_id:
                    logging.debug(f"TPOT predictor object CHANGED: {old_tpot_id} → {new_tpot_id}")
                elif new_tpot_predictor is None and old_tpot_id:
                    logging.debug(f"TPOT predictor object REUSED (new load failed): {old_tpot_id}")

            # CRITICAL FIX: Never reuse old conformal predictors
            # Conformal calibration MUST match the TreeLite models
            # If new conformal failed to load, set to None (don't use stale calibration)
            # This prevents the 70% coverage bug where old conformal is used with new models
            final_ttft_conformal = new_ttft_conformal  # None if not loaded (correct behavior)
            final_tpot_conformal = new_tpot_conformal  # None if not loaded (correct behavior)

            new_bundle = ModelBundle(
                ttft_model=new_ttft or self.models.ttft_model,  # Keep old if load failed
                tpot_model=new_tpot or self.models.tpot_model,
                ttft_scaler=new_ttft_scaler or self.models.ttft_scaler,
                tpot_scaler=new_tpot_scaler or self.models.tpot_scaler,
                ttft_predictor=final_ttft_predictor,
                tpot_predictor=final_tpot_predictor,
                ttft_conformal=final_ttft_conformal,  # Never fallback to old conformal
                tpot_conformal=final_tpot_conformal,  # Never fallback to old conformal
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
                bundle_id_short = model_syncer.current_bundle_id[:8] if model_syncer.current_bundle_id else "unknown"

                if self.use_treelite:
                    if new_ttft_predictor and new_tpot_predictor:
                        has_conformal = new_ttft_conformal and new_tpot_conformal
                        logging.info(f"Models loaded from bundle {bundle_id_short}: TreeLite compiled models (conformal: {bool(has_conformal)})")
                    else:
                        logging.info(f"Models loaded from bundle {bundle_id_short}: Using base models (TreeLite not yet available)")
                else:
                    logging.info(f"Models loaded from bundle {bundle_id_short} (mode: {self.model_type.value})")
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

            # Create raw data dictionaries
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

                # Get mean predictions from TL2cgen (with error handling for corrupt .so files)
                try:
                    ttft_pred_array = models.ttft_predictor.predict(ttft_dmat)
                    tpot_pred_array = models.tpot_predictor.predict(tpot_dmat)

                    # Extract scalar values (handles both 1D and 2D arrays)
                    ttft_mean = float(np.ravel(ttft_pred_array)[0])
                    tpot_mean = float(np.ravel(tpot_pred_array)[0])
                except Exception as e:
                    logging.error(f"TreeLite prediction failed: {e}", exc_info=True)
                    raise HTTPException(
                        status_code=503,
                        detail=f"TreeLite prediction failed (model may be corrupt): {str(e)}"
                    )

                # Check if conformal calibration is expected (by checking bundle manifest)
                # This distinguishes bootstrap (conformal not yet available) from bugs (conformal missing when expected)
                conformal_expected = False
                bundle_path = model_syncer.get_bundle_path()
                if bundle_path:
                    manifest_path = os.path.join(bundle_path, "manifest.json")
                    if os.path.exists(manifest_path):
                        try:
                            with open(manifest_path, 'r') as f:
                                manifest = json.load(f)
                                # Check if bundle should contain conformal files
                                from bundle_constants import TTFT_CONFORMAL_FILENAME, TPOT_CONFORMAL_FILENAME
                                conformal_expected = (
                                    TTFT_CONFORMAL_FILENAME in manifest.get("files", {}) and
                                    TPOT_CONFORMAL_FILENAME in manifest.get("files", {})
                                )
                        except Exception as e:
                            logging.warning(f"Failed to read bundle manifest: {e}")

                # Apply conformal correction if available
                if models.ttft_conformal and models.tpot_conformal:
                    # Optimal path: TreeLite + conformal working correctly
                    ttft_pred = models.ttft_conformal.conformalize(ttft_mean)
                    tpot_pred = models.tpot_conformal.conformalize(tpot_mean)
                    logging.debug(f"  After conformal: TTFT={ttft_pred:.2f}, TPOT={tpot_pred:.2f}")
                elif conformal_expected:
                    # Bundle should have conformal but loading failed → FAIL FAST
                    logging.error(
                        "Conformal calibration expected (present in bundle manifest) but not loaded. "
                        "Bundle may be corrupt or incomplete."
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Conformal calibration expected but not loaded (bundle may be corrupt or incomplete)"
                    )
                else:
                    # Bootstrap phase: conformal not yet available, use mean predictions
                    ttft_pred = ttft_mean
                    tpot_pred = tpot_mean
                    logging.debug(f"  Bootstrap mode: using mean predictions (conformal not yet calibrated)")

                return ttft_pred, tpot_pred

            # Non-TreeLite paths: create DataFrames with feature engineering
            # This is needed for Bayesian Ridge, XGBoost native, and LightGBM
            df_ttft_raw = pd.DataFrame([ttft_raw_data])
            df_ttft = self._prepare_features_with_interaction(df_ttft_raw, "ttft")

            df_tpot_raw = pd.DataFrame([tpot_raw_data])
            df_tpot = self._prepare_features_with_interaction(df_tpot_raw, "tpot")

            if self.model_type == ModelType.BAYESIAN_RIDGE:
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
                # XGBoost handles categorical dtypes natively via enable_categorical
                # Just pass the DataFrame as-is (dtype must match training)

                # DEBUG: Log model state and input data for investigation
                logging.info(
                    f"DEBUG XGBoost prediction - TTFT model info: "
                    f"n_estimators={getattr(models.ttft_model, 'n_estimators', 'N/A')}, "
                    f"objective={getattr(models.ttft_model, 'objective', 'N/A')}, "
                    f"enable_categorical={getattr(models.ttft_model, 'enable_categorical', 'N/A')}"
                )
                logging.info(f"DEBUG XGBoost prediction - TTFT input shape: {df_ttft.shape}, dtypes: {df_ttft.dtypes.to_dict()}")
                logging.info(f"DEBUG XGBoost prediction - TTFT input values:\n{df_ttft.to_dict('records')}")

                ttft_pred = models.ttft_model.predict(df_ttft)
                tpot_pred = models.tpot_model.predict(df_tpot)

                logging.info(f"DEBUG XGBoost prediction - Raw predictions: TTFT={ttft_pred[0]:.4f}ms, TPOT={tpot_pred[0]:.4f}ms")

                # CRITICAL: Check if we're in TreeLite mode but using fallback path
                # If USE_TREELITE=true, base models use objective=reg:squarederror (mean regression)
                # and REQUIRE conformal adjustment to get quantile predictions
                if models.use_treelite:
                    # TreeLite mode: base models predict mean, need conformal adjustment
                    if models.ttft_conformal and models.tpot_conformal:
                        # Apply conformal adjustment to get quantile predictions
                        ttft_pred_quantile = models.ttft_conformal.conformalize(float(ttft_pred[0]))
                        tpot_pred_quantile = models.tpot_conformal.conformalize(float(tpot_pred[0]))
                        logging.warning(
                            f"TreeLite models not loaded, using fallback XGBoost with conformal adjustment. "
                            f"TTFT: {ttft_pred[0]:.2f}ms (mean) → {ttft_pred_quantile:.2f}ms (P{models.quantile*100:.0f}), "
                            f"TPOT: {tpot_pred[0]:.2f}ms (mean) → {tpot_pred_quantile:.2f}ms (P{models.quantile*100:.0f})"
                        )
                        return ttft_pred_quantile, tpot_pred_quantile
                    else:
                        # FAIL FAST: TreeLite mode but conformal missing
                        logging.error(
                            "TreeLite mode enabled but TreeLite models not loaded and conformal calibration missing. "
                            "Cannot provide valid quantile predictions. Check bundle integrity."
                        )
                        raise HTTPException(
                            status_code=503,
                            detail="TreeLite models not loaded and conformal calibration missing (cannot provide quantile predictions)"
                        )
                else:
                    # Native quantile mode: Base XGBoost models use objective=reg:quantileerror
                    # Predictions are already quantiles. No adjustment needed.
                    return ttft_pred[0], tpot_pred[0]

            else:  # LightGBM
                ttft_pred = models.ttft_model.predict(df_ttft)
                tpot_pred = models.tpot_model.predict(df_tpot)

                # CRITICAL: Check if we're in TreeLite mode but using fallback path
                # If USE_TREELITE=true, base models use objective=regression (mean regression)
                # and REQUIRE conformal adjustment to get quantile predictions
                if models.use_treelite:
                    # TreeLite mode: base models predict mean, need conformal adjustment
                    if models.ttft_conformal and models.tpot_conformal:
                        # Apply conformal adjustment to get quantile predictions
                        ttft_pred_quantile = models.ttft_conformal.conformalize(float(ttft_pred[0]))
                        tpot_pred_quantile = models.tpot_conformal.conformalize(float(tpot_pred[0]))
                        logging.warning(
                            f"TreeLite models not loaded, using fallback LightGBM with conformal adjustment. "
                            f"TTFT: {ttft_pred[0]:.2f}ms (mean) → {ttft_pred_quantile:.2f}ms (P{models.quantile*100:.0f}), "
                            f"TPOT: {tpot_pred[0]:.2f}ms (mean) → {tpot_pred_quantile:.2f}ms (P{models.quantile*100:.0f})"
                        )
                        return ttft_pred_quantile, tpot_pred_quantile
                    else:
                        # FAIL FAST: TreeLite mode but conformal missing
                        logging.error(
                            "TreeLite mode enabled but TreeLite models not loaded and conformal calibration missing. "
                            "Cannot provide valid quantile predictions. Check bundle integrity."
                        )
                        raise HTTPException(
                            status_code=503,
                            detail="TreeLite models not loaded and conformal calibration missing (cannot provide quantile predictions)"
                        )
                else:
                    # Native quantile mode: Base LightGBM models use objective=quantile
                    # Predictions are already quantiles. No adjustment needed.
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
        try:
            # Atomic read of model bundle (lock-free!)
            models = self.models
            batch_size = len(features_list)

            # Check readiness
            if not models.is_ready(self.model_type):
                raise HTTPException(status_code=503, detail="Models not ready")

            # Validation
            required = ['kv_cache_percentage', 'input_token_length', 'num_request_waiting',
                       'num_request_running', 'num_tokens_generated', 'prefix_cache_score']
            for i, features in enumerate(features_list):
                for f in required:
                    if f not in features:
                        raise ValueError(f"Missing required feature '{f}' in request {i}")
                    if not isinstance(features[f], (int, float)):
                        raise ValueError(f"Invalid type for feature '{f}' in request {i}: expected number")

            # Create raw feature data (without interaction)
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

            # TreeLite mode: use compiled models with conformal prediction
            # Fall back to base XGBoost models if TreeLite not available yet (bootstrap)
            if models.use_treelite and models.ttft_predictor and models.tpot_predictor:
                # For TreeLite batch prediction, construct feature arrays using vectorized operations
                # This eliminates the Python loop and GIL contention bottleneck

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

                # TL2cgen v1.0.0 requires DMatrix
                ttft_dmat = tl2cgen.DMatrix(ttft_features)
                tpot_dmat = tl2cgen.DMatrix(tpot_features)

                # Get batch mean predictions from TL2cgen (with error handling for corrupt .so files)
                try:
                    ttft_pred_array = models.ttft_predictor.predict(ttft_dmat)
                    tpot_pred_array = models.tpot_predictor.predict(tpot_dmat)

                    # Flatten to 1D array (handles both 1D and 2D output)
                    ttft_mean_preds = np.ravel(ttft_pred_array)
                    tpot_mean_preds = np.ravel(tpot_pred_array)
                except Exception as e:
                    logging.error(f"TreeLite batch prediction failed: {e}", exc_info=True)
                    raise HTTPException(
                        status_code=503,
                        detail=f"TreeLite prediction failed (model may be corrupt): {str(e)}"
                    )

                # Check if conformal calibration is expected (by checking bundle manifest)
                # This distinguishes bootstrap (conformal not yet available) from bugs (conformal missing when expected)
                conformal_expected = False
                bundle_path = model_syncer.get_bundle_path()
                if bundle_path:
                    manifest_path = os.path.join(bundle_path, "manifest.json")
                    if os.path.exists(manifest_path):
                        try:
                            with open(manifest_path, 'r') as f:
                                manifest = json.load(f)
                                # Check if bundle should contain conformal files
                                from bundle_constants import TTFT_CONFORMAL_FILENAME, TPOT_CONFORMAL_FILENAME
                                conformal_expected = (
                                    TTFT_CONFORMAL_FILENAME in manifest.get("files", {}) and
                                    TPOT_CONFORMAL_FILENAME in manifest.get("files", {})
                                )
                        except Exception as e:
                            logging.warning(f"Failed to read bundle manifest: {e}")

                # Apply conformal correction if available
                if models.ttft_conformal and models.tpot_conformal:
                    # Optimal path: TreeLite + conformal working correctly
                    # Use vectorized batch conformalization (much faster than loop)
                    ttft_pred = models.ttft_conformal.conformalize_batch(ttft_mean_preds)
                    tpot_pred = models.tpot_conformal.conformalize_batch(tpot_mean_preds)
                elif conformal_expected:
                    # Bundle should have conformal but loading failed → FAIL FAST
                    logging.error(
                        "Conformal calibration expected (present in bundle manifest) but not loaded. "
                        "Bundle may be corrupt or incomplete."
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Conformal calibration expected but not loaded (bundle may be corrupt or incomplete)"
                    )
                else:
                    # Bootstrap phase: conformal not yet available, use mean predictions
                    ttft_pred = ttft_mean_preds
                    tpot_pred = tpot_mean_preds

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
                # XGBoost handles categorical dtypes natively via enable_categorical
                # Just pass the DataFrame as-is (dtype must match training)
                ttft_pred = models.ttft_model.predict(df_ttft_batch)
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
    use_treelite: bool = Field(..., description="Whether TreeLite mode is enabled")
    bundle_info: Optional[dict] = Field(None, description="Current bundle metadata (training_samples, test_samples, bundle_id, created_at)")


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

    # Check if models exist in current bundle
    bundle_path = model_syncer.get_bundle_path()
    models_exist = {}

    if bundle_path:
        models_exist = {
            "ttft_model": os.path.exists(os.path.join(bundle_path, TTFT_MODEL_FILENAME)),
            "tpot_model": os.path.exists(os.path.join(bundle_path, TPOT_MODEL_FILENAME)),
        }

        if predictor.model_type == ModelType.BAYESIAN_RIDGE:
            models_exist.update({
                "ttft_scaler": os.path.exists(os.path.join(bundle_path, TTFT_SCALER_FILENAME)),
                "tpot_scaler": os.path.exists(os.path.join(bundle_path, TPOT_SCALER_FILENAME)),
            })
    else:
        models_exist = {
            "error": "No bundle loaded"
        }

    # Build bundle_info from cached bundle metadata
    bundle_info = None
    if model_syncer.current_bundle_info:
        bundle_info = {
            "bundle_id": model_syncer.current_bundle_info.get("bundle_id", "unknown")[:8],
            "training_samples": model_syncer.current_bundle_info.get("training_samples", {}),
            "test_samples": model_syncer.current_bundle_info.get("test_samples", {}),
            "created_at": model_syncer.current_bundle_info.get("created_at", "unknown")
        }

    return StatusResponse(
        is_ready=predictor.is_ready,
        model_type=predictor.model_type.value,
        quantile=predictor.quantile,
        last_model_load=models.last_load,
        training_server_url=settings.TRAINING_SERVER_URL,
        models_exist=models_exist,
        use_treelite=settings.USE_TREELITE,
        bundle_info=bundle_info
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    """Make quantile latency predictions."""
    try:
        ttft_pred, tpot_pred = predictor.predict(request.model_dump())

        # Ensure non-negative predictions
        ttft_pred = max(0, ttft_pred)
        tpot_pred = max(0, tpot_pred)

        # Atomic read for last_load timestamp
        models = predictor.models

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
        # First sync bundle from training server
        synced = model_syncer.sync_bundle()

        # Then load models from bundle
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
        # Ensure cache is updated before reading
        if models.ttft_conformal._cache_dirty:
            models.ttft_conformal._update_quantile_cache()

        stats["ttft_conformal"] = {
            "calibration_samples": len(models.ttft_conformal.calibration_residuals),
            "quantile_adjustment_ms": float(models.ttft_conformal._cached_quantile_value) if models.ttft_conformal._cached_quantile_value is not None else None,
            "target_quantile": models.ttft_conformal.quantile,
            "max_calibration_samples": models.ttft_conformal.max_calibration_samples
        }
    else:
        stats["ttft_conformal"] = {"error": "TTFT conformal calibration not loaded"}

    if models.tpot_conformal:
        # Ensure cache is updated before reading
        if models.tpot_conformal._cache_dirty:
            models.tpot_conformal._update_quantile_cache()

        stats["tpot_conformal"] = {
            "calibration_samples": len(models.tpot_conformal.calibration_residuals),
            "quantile_adjustment_ms": float(models.tpot_conformal._cached_quantile_value) if models.tpot_conformal._cached_quantile_value is not None else None,
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
    # Initial bundle sync & load
    model_syncer.sync_bundle()
    predictor.load_models()
    model_syncer.start()

@app.on_event("shutdown")
async def shutdown():
    logging.info("Shutting down...")
    model_syncer.shutdown()


if __name__ == "__main__":
    uvicorn.run("__main__:app", host=settings.HOST, port=settings.PORT, reload=True)
