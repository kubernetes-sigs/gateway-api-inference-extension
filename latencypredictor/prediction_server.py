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
from datetime import datetime, timezone
from typing import Tuple, Optional, List
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
    import treelite_runtime
    TREELITE_AVAILABLE = True
except ImportError:
    TREELITE_AVAILABLE = False
    logging.warning("TreeLite runtime not available. Install with: pip install treelite_runtime")

try:
    from conformal_quantile import ConformalQuantilePredictor
    CONFORMAL_AVAILABLE = True
except ImportError:
    CONFORMAL_AVAILABLE = False
    logging.warning("ConformalQuantilePredictor not available. Check conformal_quantile.py exists.")

class ModelType(str, Enum):
    BAYESIAN_RIDGE = "bayesian_ridge"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    TREELITE = "treelite"

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

        # Ensure local directories
        for path in [
            settings.LOCAL_TTFT_MODEL_PATH,
            settings.LOCAL_TPOT_MODEL_PATH,
            settings.LOCAL_TTFT_SCALER_PATH,
            settings.LOCAL_TPOT_SCALER_PATH,
            settings.LOCAL_TTFT_TREELITE_PATH,
            settings.LOCAL_TPOT_TREELITE_PATH,
        ]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def _download_model_if_newer(self, name: str, dest: str) -> bool:
        try:
            info_url = f"{settings.TRAINING_SERVER_URL}/model/{name}/info"
            r = requests.get(info_url, timeout=settings.HTTP_TIMEOUT)
            if r.status_code != 200:
                return False
            info = r.json()
            mtime = info.get("last_modified")
            if not mtime:
                return False
            server_time = datetime.fromisoformat(mtime.replace('Z', '+00:00'))

            if os.path.exists(dest):
                local_time = datetime.fromtimestamp(os.path.getmtime(dest), tz=timezone.utc)
                if local_time >= server_time:
                    logging.info(f"Model {name} is up-to-date: {dest}")
                    return False

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
            logging.info(f"Downloaded {name} -> {dest}")
            return True

        except requests.RequestException as e:
            logging.error(f"Network error for {name}: {e}")
            return False
        except OSError as e:
            logging.error(f"Filesystem error for {name}: {e}")
            return False

    def sync_models(self) -> bool:
        """Sync all relevant models; returns True if any updated."""
        with self._sync_lock:
            updated = False
            to_sync = [
                ("ttft", settings.LOCAL_TTFT_MODEL_PATH),
                ("tpot", settings.LOCAL_TPOT_MODEL_PATH),
            ]

            # Sync TreeLite models if enabled
            if settings.USE_TREELITE and TREELITE_AVAILABLE:
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
        self.use_treelite = settings.USE_TREELITE and TREELITE_AVAILABLE and mt in [ModelType.XGBOOST, ModelType.LIGHTGBM]

        # Model storage
        self.ttft_model = None
        self.tpot_model = None
        self.ttft_scaler = None
        self.tpot_scaler = None

        # TreeLite predictors (lightweight, compiled models)
        self.ttft_predictor = None
        self.tpot_predictor = None

        # Conformal predictors (for TreeLite mode quantile predictions)
        self.ttft_conformal = None
        self.tpot_conformal = None

        self.lock = threading.RLock()
        self.last_load: Optional[datetime] = None
        logging.info(f"Predictor type: {self.model_type}, quantile: {self.quantile}, use_treelite: {self.use_treelite}")

    @property
    def is_ready(self) -> bool:
        with self.lock:
            if self.model_type == ModelType.BAYESIAN_RIDGE:
                return all([self.ttft_model, self.tpot_model, self.ttft_scaler, self.tpot_scaler])
            elif self.use_treelite:
                # For TreeLite, we need the compiled predictors
                return all([self.ttft_predictor, self.tpot_predictor])
            else:  # XGBoost or LightGBM without TreeLite
                return all([self.ttft_model, self.tpot_model])

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

            # make it categorical for tree models (safe for LGB, XGB with enable_categorical)
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
        try:
            with self.lock:
                # Load TreeLite models if enabled
                if self.use_treelite:
                    if os.path.exists(settings.LOCAL_TTFT_TREELITE_PATH):
                        self.ttft_predictor = treelite_runtime.Predictor(settings.LOCAL_TTFT_TREELITE_PATH, nthread=8)
                        logging.info("TTFT TreeLite model loaded")
                    else:
                        logging.warning(f"TreeLite model not found: {settings.LOCAL_TTFT_TREELITE_PATH}")

                    if os.path.exists(settings.LOCAL_TPOT_TREELITE_PATH):
                        self.tpot_predictor = treelite_runtime.Predictor(settings.LOCAL_TPOT_TREELITE_PATH, nthread=8)
                        logging.info("TPOT TreeLite model loaded")
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
                                self.ttft_conformal = ConformalQuantilePredictor.from_state(ttft_conf_state)
                                logging.info(f"TTFT conformal calibration loaded ({len(ttft_conf_state.get('calibration_residuals', []))} samples)")
                            except Exception as e:
                                logging.error(f"Error loading TTFT conformal calibration: {e}")
                                self.ttft_conformal = None
                        else:
                            logging.warning(f"TTFT conformal calibration not found: {settings.LOCAL_TTFT_CONFORMAL_PATH}")
                            self.ttft_conformal = None

                        # Load TPOT conformal calibration
                        if os.path.exists(settings.LOCAL_TPOT_CONFORMAL_PATH):
                            try:
                                with open(settings.LOCAL_TPOT_CONFORMAL_PATH, 'r') as f:
                                    tpot_conf_state = json.load(f)
                                self.tpot_conformal = ConformalQuantilePredictor.from_state(tpot_conf_state)
                                logging.info(f"TPOT conformal calibration loaded ({len(tpot_conf_state.get('calibration_residuals', []))} samples)")
                            except Exception as e:
                                logging.error(f"Error loading TPOT conformal calibration: {e}")
                                self.tpot_conformal = None
                        else:
                            logging.warning(f"TPOT conformal calibration not found: {settings.LOCAL_TPOT_CONFORMAL_PATH}")
                            self.tpot_conformal = None
                else:
                    # Load XGBoost/LightGBM/BayesianRidge models
                    new_ttft = joblib.load(settings.LOCAL_TTFT_MODEL_PATH) if os.path.exists(settings.LOCAL_TTFT_MODEL_PATH) else None
                    new_tpot = joblib.load(settings.LOCAL_TPOT_MODEL_PATH) if os.path.exists(settings.LOCAL_TPOT_MODEL_PATH) else None

                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        new_ttft_scaler = joblib.load(settings.LOCAL_TTFT_SCALER_PATH) if os.path.exists(settings.LOCAL_TTFT_SCALER_PATH) else None
                        new_tpot_scaler = joblib.load(settings.LOCAL_TPOT_SCALER_PATH) if os.path.exists(settings.LOCAL_TPOT_SCALER_PATH) else None
                    else:
                        new_ttft_scaler = new_tpot_scaler = None

                    if new_ttft: self.ttft_model = new_ttft
                    if new_tpot: self.tpot_model = new_tpot
                    if new_ttft_scaler: self.ttft_scaler = new_ttft_scaler
                    if new_tpot_scaler: self.tpot_scaler = new_tpot_scaler

                self.last_load = datetime.now(timezone.utc)
                if self.is_ready:
                    logging.info(f"Models loaded successfully (TreeLite: {self.use_treelite})")
                    return True
                logging.warning("Models missing after load")
                return False
        except Exception as e:
            logging.error(f"Load error: {e}")
            return False

    def predict(self, features: dict) -> Tuple[float, float]:
        """Make quantile predictions using the loaded models."""
        try:
            with self.lock:
                if not self.is_ready:
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

                # Prepare features with interactions
                df_ttft_raw = pd.DataFrame([ttft_raw_data])
                df_ttft = self._prepare_features_with_interaction(df_ttft_raw, "ttft")

                df_tpot_raw = pd.DataFrame([tpot_raw_data])
                df_tpot = self._prepare_features_with_interaction(df_tpot_raw, "tpot")

                # Use TreeLite for inference if enabled
                if self.use_treelite:
                    # TreeLite expects numpy arrays
                    ttft_arr = df_ttft.values.astype('float32')
                    tpot_arr = df_tpot.values.astype('float32')

                    # Create DMatrix for TreeLite
                    ttft_dmat = treelite_runtime.DMatrix(ttft_arr)
                    tpot_dmat = treelite_runtime.DMatrix(tpot_arr)

                    # Get mean predictions from TreeLite
                    ttft_mean = float(self.ttft_predictor.predict(ttft_dmat)[0])
                    tpot_mean = float(self.tpot_predictor.predict(tpot_dmat)[0])

                    # Apply conformal correction to get quantile predictions
                    if self.ttft_conformal:
                        ttft_pred = self.ttft_conformal.conformalize(ttft_mean)
                    else:
                        # Fallback: use mean prediction (not ideal but prevents crashes)
                        logging.warning("TTFT conformal calibration not loaded, returning mean prediction")
                        ttft_pred = ttft_mean

                    if self.tpot_conformal:
                        tpot_pred = self.tpot_conformal.conformalize(tpot_mean)
                    else:
                        # Fallback: use mean prediction (not ideal but prevents crashes)
                        logging.warning("TPOT conformal calibration not loaded, returning mean prediction")
                        tpot_pred = tpot_mean

                    return ttft_pred, tpot_pred

                elif self.model_type == ModelType.BAYESIAN_RIDGE:
                    ttft_for_scale = df_ttft.drop(columns=['prefill_score_bucket'], errors='ignore')
                    ttft_scaled = self.ttft_scaler.transform(ttft_for_scale)
                    tpot_scaled = self.tpot_scaler.transform(df_tpot)

                    ttft_pred_mean, ttft_std = self.ttft_model.predict(ttft_scaled, return_std=True)
                    tpot_pred_mean, tpot_std = self.tpot_model.predict(tpot_scaled, return_std=True)

                    std_factor = 1.28 if self.quantile == 0.9 else (2.0 if self.quantile == 0.95 else 0.674)
                    ttft_pred = ttft_pred_mean[0] + std_factor * ttft_std[0]
                    tpot_pred = tpot_pred_mean[0] + std_factor * tpot_std[0]

                    return ttft_pred, tpot_pred

                elif self.model_type == ModelType.XGBOOST:
                    ttft_pred = self.ttft_model.predict(df_ttft)
                    tpot_pred = self.tpot_model.predict(df_tpot)

                    return ttft_pred[0], tpot_pred[0]

                else:  # LightGBM
                    ttft_pred = self.ttft_model.predict(df_ttft)
                    tpot_pred = self.tpot_model.predict(df_tpot)

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
        """Make batch quantile predictions using the loaded models."""
        try:
            with self.lock:
                if not self.is_ready:
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

                # Prepare features with interactions
                df_ttft_raw = pd.DataFrame(ttft_raw_data)
                df_ttft_batch = self._prepare_features_with_interaction(df_ttft_raw, "ttft")

                df_tpot_raw = pd.DataFrame(tpot_raw_data)
                df_tpot_batch = self._prepare_features_with_interaction(df_tpot_raw, "tpot")

                # Use TreeLite for batch inference if enabled
                if self.use_treelite:
                    # TreeLite expects numpy arrays
                    ttft_arr = df_ttft_batch.values.astype('float32')
                    tpot_arr = df_tpot_batch.values.astype('float32')

                    # Create DMatrix for TreeLite
                    ttft_dmat = treelite_runtime.DMatrix(ttft_arr)
                    tpot_dmat = treelite_runtime.DMatrix(tpot_arr)

                    ttft_pred = self.ttft_predictor.predict(ttft_dmat)
                    tpot_pred = self.tpot_predictor.predict(tpot_dmat)

                    return ttft_pred, tpot_pred

                elif self.model_type == ModelType.BAYESIAN_RIDGE:
                    ttft_for_scale = df_ttft_batch.drop(columns=['prefill_score_bucket'], errors='ignore')
                    ttft_scaled = self.ttft_scaler.transform(ttft_for_scale)
                    tpot_scaled = self.tpot_scaler.transform(df_tpot_batch)

                    ttft_pred_mean, ttft_std = self.ttft_model.predict(ttft_scaled, return_std=True)
                    tpot_pred_mean, tpot_std = self.tpot_model.predict(tpot_scaled, return_std=True)

                    std_factor = 1.28 if self.quantile == 0.9 else (2.0 if self.quantile == 0.95 else 0.674)
                    ttft_pred = ttft_pred_mean + std_factor * ttft_std
                    tpot_pred = tpot_pred_mean + std_factor * tpot_std

                    return ttft_pred, tpot_pred

                elif self.model_type == ModelType.XGBOOST:
                    ttft_pred = self.ttft_model.predict(df_ttft_batch)
                    tpot_pred = self.tpot_model.predict(df_tpot_batch)

                    return ttft_pred, tpot_pred

                else:  # LightGBM
                    ttft_pred = self.ttft_model.predict(df_ttft_batch)
                    tpot_pred = self.tpot_model.predict(df_tpot_batch)

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
        last_model_load=predictor.last_load,
        training_server_url=settings.TRAINING_SERVER_URL,
        models_exist=models_exist
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """Make quantile latency predictions."""
    try:
        ttft_pred, tpot_pred = predictor.predict(request.dict())
        
        # Ensure non-negative predictions
        ttft_pred = max(0, ttft_pred)
        tpot_pred = max(0, tpot_pred)
        
        return PredictionResponse(
            ttft_ms=ttft_pred,
            tpot_ms=tpot_pred,
            predicted_at=datetime.now(timezone.utc),
            model_type=predictor.model_type.value,
            quantile=predictor.quantile,
            last_model_load=predictor.last_load 
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction")


@app.post("/predict/bulk/strict", response_model=BulkPredictionResponse)
async def predict_bulk_strict_endpoint(request: BulkPredictionRequest):
    """Make bulk quantile latency predictions using batch processing (fails on any single error)."""
    start_time = time.time()
    
    try:
        # Convert all requests to dict format
        features_list = [pred_request.dict() for pred_request in request.requests]
        
        # Make batch prediction
        ttft_preds, tpot_preds = predictor.predict_batch(features_list)
        
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
                last_model_load=predictor.last_load 
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
async def predict_bulk_endpoint(request: BulkPredictionRequest):
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
            features = pred_request.dict()
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
            # Make batch prediction for all valid requests
            ttft_preds, tpot_preds = predictor.predict_batch(valid_requests)
            
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
                    last_model_load=predictor.last_load 
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
            "last_load_time": predictor.last_load 
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

    stats = {
        "use_treelite": True,
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile,
        "ttft_conformal": None,
        "tpot_conformal": None
    }

    if predictor.ttft_conformal:
        stats["ttft_conformal"] = {
            "calibration_samples": len(predictor.ttft_conformal.calibration_residuals),
            "quantile_adjustment_ms": predictor.ttft_conformal._cached_quantile_value if not predictor.ttft_conformal._cache_dirty else None,
            "target_quantile": predictor.ttft_conformal.quantile,
            "max_calibration_samples": predictor.ttft_conformal.max_calibration_samples
        }
    else:
        stats["ttft_conformal"] = {"error": "TTFT conformal calibration not loaded"}

    if predictor.tpot_conformal:
        stats["tpot_conformal"] = {
            "calibration_samples": len(predictor.tpot_conformal.calibration_residuals),
            "quantile_adjustment_ms": predictor.tpot_conformal._cached_quantile_value if not predictor.tpot_conformal._cache_dirty else None,
            "target_quantile": predictor.tpot_conformal.quantile,
            "max_calibration_samples": predictor.tpot_conformal.max_calibration_samples
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


if __name__ == "__main__":
    uvicorn.run("__main__:app", host=settings.HOST, port=settings.PORT, reload=True)
