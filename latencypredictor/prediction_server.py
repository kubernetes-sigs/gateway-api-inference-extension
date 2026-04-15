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
import tempfile
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone
from typing import Tuple, Optional, List
from enum import Enum

import joblib
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

try:
    import orjson  # noqa: F401 — imported for ORJSONResponse
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    logging.warning("orjson not available; falling back to stdlib JSON. Install with: pip install orjson")

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


class ModelType(str, Enum):
    BAYESIAN_RIDGE = "bayesian_ridge"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class ObjectiveType(str, Enum):
    QUANTILE = "quantile"
    MEAN = "mean"


class PredictSettings:
    """Configuration for the prediction server."""

    TRAINING_SERVER_URL: str = os.getenv("TRAINING_SERVER_URL", "http://training-service:8000")

    LOCAL_TTFT_MODEL_PATH: str = os.getenv("LOCAL_TTFT_MODEL_PATH", "/local_models/ttft.joblib")
    LOCAL_TPOT_MODEL_PATH: str = os.getenv("LOCAL_TPOT_MODEL_PATH", "/local_models/tpot.joblib")
    LOCAL_TTFT_SCALER_PATH: str = os.getenv("LOCAL_TTFT_SCALER_PATH", "/local_models/ttft_scaler.joblib")
    LOCAL_TPOT_SCALER_PATH: str = os.getenv("LOCAL_TPOT_SCALER_PATH", "/local_models/tpot_scaler.joblib")

    MODEL_SYNC_INTERVAL_SEC: int = int(os.getenv("MODEL_SYNC_INTERVAL_SEC", "10"))
    MODEL_TYPE: ModelType = ModelType(os.getenv("LATENCY_MODEL_TYPE", "xgboost"))

    QUANTILE_ALPHA: float = float(os.getenv("LATENCY_QUANTILE_ALPHA", "0.9"))
    OBJECTIVE_TYPE: ObjectiveType = ObjectiveType(os.getenv("LATENCY_OBJECTIVE_TYPE", "quantile"))

    HOST: str = os.getenv("PREDICT_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PREDICT_PORT", "8001"))

    HTTP_TIMEOUT: int = int(os.getenv("HTTP_TIMEOUT", "30"))
    DOWNLOAD_RETRIES: int = int(os.getenv("DOWNLOAD_RETRIES", "3"))

    ENSEMBLE_MODE: bool = os.getenv("LATENCY_ENSEMBLE_MODE", "true").lower() == "true"
    ENABLE_TOKEN_IN_FLIGHT_FEATURES: bool = os.getenv("LATENCY_ENABLE_TOKEN_IN_FLIGHT_FEATURES", "true").lower() == "true"

    # Gated ensemble model paths (each wraps noqueue + queued sub-models)
    LOCAL_TTFT_GATED_MODEL_PATH: str = os.getenv("LOCAL_TTFT_GATED_MODEL_PATH", "/local_models/ttft_gated.joblib")
    LOCAL_TPOT_GATED_MODEL_PATH: str = os.getenv("LOCAL_TPOT_GATED_MODEL_PATH", "/local_models/tpot_gated.joblib")


class QueueGatedModel:
    """Wraps noqueue + queued sub-models into one joblib-serializable object.

    At prediction time the caller checks num_request_waiting and picks the
    appropriate sub-model + scaler from inside this wrapper.
    """

    def __init__(self, noqueue_model, queued_model,
                 noqueue_scaler=None, queued_scaler=None):
        self.noqueue_model = noqueue_model
        self.queued_model = queued_model
        self.noqueue_scaler = noqueue_scaler
        self.queued_scaler = queued_scaler


settings = PredictSettings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _create_http_session() -> requests.Session:
    """Create a requests session with retry logic for transient errors."""
    session = requests.Session()
    retry = Retry(
        total=settings.DOWNLOAD_RETRIES,
        backoff_factor=1,  # 1s, 2s, 4s
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _file_checksum(path: str) -> Optional[str]:
    """Return MD5 hex digest of a file, or None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelSyncer:
    """Downloads models from a training server via HTTP."""

    def __init__(self):
        self._shutdown_event = threading.Event()
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_lock = threading.Lock()
        self._http: requests.Session = _create_http_session()

        all_paths = [
            settings.LOCAL_TTFT_MODEL_PATH,
            settings.LOCAL_TPOT_MODEL_PATH,
            settings.LOCAL_TTFT_SCALER_PATH,
            settings.LOCAL_TPOT_SCALER_PATH,
        ]
        if settings.ENSEMBLE_MODE:
            all_paths += [
                settings.LOCAL_TTFT_GATED_MODEL_PATH,
                settings.LOCAL_TPOT_GATED_MODEL_PATH,
            ]
        for path in all_paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def _download_model_if_newer(self, name: str, dest: str) -> bool:
        """Download a model file if the server has a newer version.
        
        Returns True if a new file was written to dest.
        """
        try:
            # Check server timestamp
            info_url = f"{settings.TRAINING_SERVER_URL}/model/{name}/info"
            r = self._http.get(info_url, timeout=settings.HTTP_TIMEOUT)
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
                    return False

            # Download with streaming
            dl_url = f"{settings.TRAINING_SERVER_URL}/model/{name}/download"
            dl = self._http.get(dl_url, timeout=settings.HTTP_TIMEOUT, stream=True)
            if dl.status_code != 200:
                logging.error(f"Failed download {name}: HTTP {dl.status_code}")
                return False

            # Validate Content-Length if provided
            expected_size = dl.headers.get("Content-Length")
            expected_size = int(expected_size) if expected_size else None

            # Write to a unique temp file (safe across workers)
            dir_name = os.path.dirname(dest)
            fd, tmp_path = tempfile.mkstemp(
                suffix=f".{name}.tmp",
                prefix=f"pid{os.getpid()}_",
                dir=dir_name,
            )
            try:
                bytes_written = 0
                with os.fdopen(fd, 'wb') as f:
                    for chunk in dl.iter_content(65536):
                        if chunk:
                            f.write(chunk)
                            bytes_written += len(chunk)

                # Validate download completeness
                if bytes_written == 0:
                    logging.warning(f"Empty download for {name}, skipping")
                    os.unlink(tmp_path)
                    return False

                if expected_size is not None and bytes_written != expected_size:
                    logging.error(
                        f"Incomplete download for {name}: got {bytes_written} bytes, "
                        f"expected {expected_size} bytes"
                    )
                    os.unlink(tmp_path)
                    return False

                # Verify the file is a valid joblib/pickle before replacing
                try:
                    joblib.load(tmp_path)
                except Exception as e:
                    logging.error(f"Downloaded {name} is corrupt (joblib.load failed): {e}")
                    os.unlink(tmp_path)
                    return False

                # Atomic replace
                os.replace(tmp_path, dest)
                logging.info(f"Downloaded {name} -> {dest} ({bytes_written} bytes)")
                return True

            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

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
            if settings.MODEL_TYPE == ModelType.BAYESIAN_RIDGE:
                to_sync += [
                    ("ttft_scaler", settings.LOCAL_TTFT_SCALER_PATH),
                    ("tpot_scaler", settings.LOCAL_TPOT_SCALER_PATH),
                ]
            if settings.ENSEMBLE_MODE:
                to_sync += [
                    ("ttft_gated", settings.LOCAL_TTFT_GATED_MODEL_PATH),
                    ("tpot_gated", settings.LOCAL_TPOT_GATED_MODEL_PATH),
                ]
            for name, path in to_sync:
                if self._download_model_if_newer(name, path):
                    updated = True
            return updated

    def _sync_loop(self):
        while not self._shutdown_event.is_set():
            try:
                # Download the latest model files from the training server.
                # sync_models() returns True only if THIS worker downloaded a
                # new file. In multi-worker mode only one of the N workers
                # wins that race per cycle, so we must call load_models()
                # unconditionally — otherwise the N-1 workers that lost the
                # download race would keep serving a stale in-memory model.
                # load_models() has its own checksum-based early return, so
                # the cost when the file on disk hasn't changed is just a
                # file-checksum read.
                self.sync_models()
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
        logging.info(f"Sync thread started (PID={os.getpid()}, interval={settings.MODEL_SYNC_INTERVAL_SEC}s)")

    def shutdown(self):
        self._shutdown_event.set()
        if self._sync_thread:
            self._sync_thread.join(timeout=10)
        self._http.close()


class LightweightPredictor:
    """Handles inference using loaded regression models (quantile or mean)."""

    def __init__(self):
        mt = settings.MODEL_TYPE
        self.prefix_buckets = 4

        if mt == ModelType.XGBOOST and not XGBOOST_AVAILABLE:
            logging.warning("XGBoost not available. Falling back to Bayesian Ridge")
            mt = ModelType.BAYESIAN_RIDGE
        elif mt == ModelType.LIGHTGBM and not LIGHTGBM_AVAILABLE:
            logging.warning("LightGBM not available. Falling back to Bayesian Ridge")
            mt = ModelType.BAYESIAN_RIDGE

        self.model_type = mt
        self.quantile = settings.QUANTILE_ALPHA
        self.objective_type = settings.OBJECTIVE_TYPE
        self.ttft_model = None
        self.tpot_model = None
        self.ttft_scaler = None
        self.tpot_scaler = None

        # Gated ensemble model wrappers (QueueGatedModel instances)
        self.ttft_gated: Optional[QueueGatedModel] = None
        self.tpot_gated: Optional[QueueGatedModel] = None
        self.ensemble_active: bool = False

        self.lock = threading.RLock()
        self.last_load: Optional[datetime] = None
        # Track checksums to avoid redundant reloads
        self._loaded_checksums: dict = {}
        logging.info(f"Predictor type: {self.model_type}, objective: {self.objective_type}, quantile: {self.quantile}")

    @property
    def is_ready(self) -> bool:
        with self.lock:
            if self.model_type == ModelType.BAYESIAN_RIDGE:
                return all([self.ttft_model, self.tpot_model, self.ttft_scaler, self.tpot_scaler])
            else:
                return all([self.ttft_model, self.tpot_model])

    def _prepare_features_with_interaction(self, df: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """Prepare features with interaction terms to match training server."""
        if 'pod_type' in df.columns:
            df['pod_type'] = df['pod_type'].fillna('')
            df['pod_type_cat'] = pd.Categorical(
                df['pod_type'],
                categories=['', 'prefill', 'decode'],
                ordered=False
            )
        else:
            df['pod_type_cat'] = pd.Categorical(
                [''] * len(df), categories=['', 'prefill', 'decode'], ordered=False
            )

        # Binary feature: gives the tree a clean signal to partition idle vs queued
        df['is_queued'] = (df['num_request_waiting'] > 0).astype(int)

        _tif = []
        if settings.ENABLE_TOKEN_IN_FLIGHT_FEATURES:
            if 'prefill_tokens_in_flight' not in df.columns:
                df['prefill_tokens_in_flight'] = 0
            if 'decode_tokens_in_flight' not in df.columns:
                df['decode_tokens_in_flight'] = 0
            _tif = ['prefill_tokens_in_flight', 'decode_tokens_in_flight']

        if model_type == "ttft":
            df['effective_input_tokens'] = (1 - df['prefix_cache_score']) * df['input_token_length']
            df['prefill_score_bucket'] = (
                (df['prefix_cache_score'].clip(0, 1) * self.prefix_buckets)
                .astype(int)
                .clip(upper=self.prefix_buckets - 1)
            )
            df['prefill_score_bucket'] = pd.Categorical(
                df['prefill_score_bucket'], categories=[0, 1, 2, 3], ordered=True
            )

            feature_cols = (
                ['is_queued', 'kv_cache_percentage', 'input_token_length',
                 'num_request_waiting', 'num_request_running']
                + _tif
                + ['prefix_cache_score', 'effective_input_tokens', 'prefill_score_bucket', 'pod_type_cat']
            )
            return df[feature_cols]

        else:  # tpot
            feature_cols = (
                ['is_queued', 'kv_cache_percentage', 'input_token_length',
                 'num_request_waiting', 'num_request_running']
                + _tif
                + ['num_tokens_generated', 'pod_type_cat']
            )
            return df[feature_cols]

    def _prepare_features_for_ensemble(self, df: pd.DataFrame, model_type: str, queue_regime: str) -> pd.DataFrame:
        """Prepare features for ensemble sub-models.

        Args:
            df: DataFrame with raw features
            model_type: 'ttft' or 'tpot'
            queue_regime: 'noqueue' or 'queued'
        Returns:
            DataFrame with engineered features, with queue columns dropped for noqueue regime
        """
        features = self._prepare_features_with_interaction(df, model_type)
        if queue_regime == "noqueue":
            features = features.drop(columns=['is_queued', 'num_request_waiting'], errors='ignore')
        return features

    def load_models(self) -> bool:
        """Load models from disk. Skips reload if checksums haven't changed."""
        try:
            # Compute checksums before acquiring lock to minimise lock hold time
            paths_to_check = [
                ("ttft", settings.LOCAL_TTFT_MODEL_PATH),
                ("tpot", settings.LOCAL_TPOT_MODEL_PATH),
            ]
            if self.model_type == ModelType.BAYESIAN_RIDGE:
                paths_to_check += [
                    ("ttft_scaler", settings.LOCAL_TTFT_SCALER_PATH),
                    ("tpot_scaler", settings.LOCAL_TPOT_SCALER_PATH),
                ]

            # Also check ensemble model files if ensemble mode is on
            ensemble_paths_to_check = []
            if settings.ENSEMBLE_MODE:
                ensemble_paths_to_check = [
                    ("ttft_gated", settings.LOCAL_TTFT_GATED_MODEL_PATH),
                    ("tpot_gated", settings.LOCAL_TPOT_GATED_MODEL_PATH),
                ]

            current_checksums = {}
            for name, path in paths_to_check:
                cs = _file_checksum(path)
                if cs is None:
                    # File doesn't exist yet — can't load
                    if not self.is_ready:
                        logging.debug(f"Model file {path} not found yet")
                    return False
                current_checksums[name] = cs

            # Ensemble checksums (best-effort: missing files just mean ensemble stays inactive)
            ensemble_checksums = {}
            ensemble_all_present = True
            for name, path in ensemble_paths_to_check:
                cs = _file_checksum(path)
                if cs is None:
                    ensemble_all_present = False
                else:
                    ensemble_checksums[name] = cs

            all_checksums = {**current_checksums, **ensemble_checksums}

            # Skip reload if nothing changed and we're already ready
            if self.is_ready and all_checksums == self._loaded_checksums:
                return True

            # Load all model files outside the lock — joblib.load() is slow
            # (file I/O + deserialization) and must not block predictions.
            new_ttft = joblib.load(settings.LOCAL_TTFT_MODEL_PATH)
            new_tpot = joblib.load(settings.LOCAL_TPOT_MODEL_PATH)

            new_ttft_scaler = new_tpot_scaler = None
            if self.model_type == ModelType.BAYESIAN_RIDGE:
                new_ttft_scaler = joblib.load(settings.LOCAL_TTFT_SCALER_PATH)
                new_tpot_scaler = joblib.load(settings.LOCAL_TPOT_SCALER_PATH)

            new_ttft_gated = new_tpot_gated = None
            new_ensemble_active = False
            if settings.ENSEMBLE_MODE and ensemble_all_present:
                try:
                    new_ttft_gated = joblib.load(settings.LOCAL_TTFT_GATED_MODEL_PATH)
                    new_tpot_gated = joblib.load(settings.LOCAL_TPOT_GATED_MODEL_PATH)
                    new_ensemble_active = True
                    logging.info("Gated ensemble models loaded successfully")
                except Exception as e:
                    logging.warning(f"Failed to load gated ensemble models, using single model: {e}")

            # Hold the lock only for the pointer swap — microseconds, not milliseconds.
            with self.lock:
                self.ttft_model = new_ttft
                self.tpot_model = new_tpot
                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    self.ttft_scaler = new_ttft_scaler
                    self.tpot_scaler = new_tpot_scaler
                self.ttft_gated = new_ttft_gated
                self.tpot_gated = new_tpot_gated
                self.ensemble_active = new_ensemble_active
                self._loaded_checksums = all_checksums
                self.last_load = datetime.now(timezone.utc)

            logging.info(f"Models loaded (PID={os.getpid()}, ensemble_active={self.ensemble_active})")
            return True

        except Exception as e:
            logging.error(f"Load error: {e}")
            return False

    def _predict_with_models(self, df_ttft: pd.DataFrame, df_tpot: pd.DataFrame,
                             ttft_model, tpot_model, ttft_scaler=None, tpot_scaler=None):
        """Core prediction logic using given models/scalers. Returns (ttft_preds, tpot_preds) as arrays."""
        if self.model_type == ModelType.BAYESIAN_RIDGE:
            ttft_for_scale = df_ttft.drop(columns=['prefill_score_bucket'], errors='ignore')
            if 'pod_type_cat' in ttft_for_scale.columns:
                ttft_for_scale = pd.get_dummies(
                    ttft_for_scale, columns=['pod_type_cat'],
                    prefix='pod_type', drop_first=False,
                )
            ttft_scaled = ttft_scaler.transform(ttft_for_scale)

            tpot_for_scale = df_tpot.copy()
            if 'pod_type_cat' in tpot_for_scale.columns:
                tpot_for_scale = pd.get_dummies(
                    tpot_for_scale, columns=['pod_type_cat'],
                    prefix='pod_type', drop_first=False,
                )
            tpot_scaled = tpot_scaler.transform(tpot_for_scale)

            ttft_pred_mean, ttft_std = ttft_model.predict(ttft_scaled, return_std=True)
            tpot_pred_mean, tpot_std = tpot_model.predict(tpot_scaled, return_std=True)

            if self.objective_type == ObjectiveType.MEAN:
                return ttft_pred_mean, tpot_pred_mean

            std_factor = 1.28 if self.quantile == 0.9 else (2.0 if self.quantile == 0.95 else 0.674)
            return (
                ttft_pred_mean + std_factor * ttft_std,
                tpot_pred_mean + std_factor * tpot_std,
            )
        else:  # XGBoost or LightGBM
            return ttft_model.predict(df_ttft), tpot_model.predict(df_tpot)

    def predict(self, features: dict) -> Tuple[float, float]:
        """Make predictions using the loaded models."""
        try:
            required = [
                'kv_cache_percentage', 'input_token_length', 'num_request_waiting',
                'num_request_running', 'num_tokens_generated', 'prefix_cache_score',
            ]
            for f in required:
                if f not in features:
                    raise ValueError(f"Missing required feature: {f}")
                if not isinstance(features[f], (int, float)):
                    raise ValueError(f"Invalid type for feature {f}: expected number")

            # Snapshot model references under the lock (fast path — no inference here).
            with self.lock:
                if not self.is_ready:
                    raise HTTPException(status_code=503, detail="Models not ready")
                if self.ensemble_active and features['num_request_waiting'] == 0:
                    queue_regime = "noqueue"
                    ttft_model = self.ttft_gated.noqueue_model
                    tpot_model = self.tpot_gated.noqueue_model
                    ttft_scaler = self.ttft_gated.noqueue_scaler
                    tpot_scaler = self.tpot_gated.noqueue_scaler
                elif self.ensemble_active:
                    queue_regime = "queued"
                    ttft_model = self.ttft_gated.queued_model
                    tpot_model = self.tpot_gated.queued_model
                    ttft_scaler = self.ttft_gated.queued_scaler
                    tpot_scaler = self.tpot_gated.queued_scaler
                else:
                    queue_regime = None
                    ttft_model = self.ttft_model
                    tpot_model = self.tpot_model
                    ttft_scaler = self.ttft_scaler
                    tpot_scaler = self.tpot_scaler

            # Heavy work: DataFrame construction + inference — no lock held.
            ttft_raw_data = {
                'kv_cache_percentage': features['kv_cache_percentage'],
                'input_token_length': features['input_token_length'],
                'num_request_waiting': features['num_request_waiting'],
                'num_request_running': features['num_request_running'],
                'prefix_cache_score': features['prefix_cache_score'],
            }
            tpot_raw_data = {
                'kv_cache_percentage': features['kv_cache_percentage'],
                'input_token_length': features['input_token_length'],
                'num_request_waiting': features['num_request_waiting'],
                'num_request_running': features['num_request_running'],
                'num_tokens_generated': features['num_tokens_generated'],
            }

            if queue_regime is not None:
                df_ttft_raw = pd.DataFrame([ttft_raw_data])
                if 'pod_type' in features:
                    df_ttft_raw['pod_type'] = features['pod_type']
                df_ttft = self._prepare_features_for_ensemble(df_ttft_raw, "ttft", queue_regime)

                df_tpot_raw = pd.DataFrame([tpot_raw_data])
                if 'pod_type' in features:
                    df_tpot_raw['pod_type'] = features['pod_type']
                df_tpot = self._prepare_features_for_ensemble(df_tpot_raw, "tpot", queue_regime)
            else:
                df_ttft_raw = pd.DataFrame([ttft_raw_data])
                if 'pod_type' in features:
                    df_ttft_raw['pod_type'] = features['pod_type']
                df_ttft = self._prepare_features_with_interaction(df_ttft_raw, "ttft")

                df_tpot_raw = pd.DataFrame([tpot_raw_data])
                if 'pod_type' in features:
                    df_tpot_raw['pod_type'] = features['pod_type']
                df_tpot = self._prepare_features_with_interaction(df_tpot_raw, "tpot")

            ttft_preds, tpot_preds = self._predict_with_models(
                df_ttft, df_tpot, ttft_model, tpot_model, ttft_scaler, tpot_scaler
            )
            return ttft_preds[0], tpot_preds[0]

        except ValueError as ve:
            logging.warning(f"Client error in predict(): {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except HTTPException:
            raise
        except Exception as e:
            logging.error("Error in predict():", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during prediction")

    def predict_batch(self, features_list: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Make batch predictions using the loaded models."""
        try:
            required = [
                'kv_cache_percentage', 'input_token_length', 'num_request_waiting',
                'num_request_running', 'num_tokens_generated', 'prefix_cache_score',
            ]
            for i, features in enumerate(features_list):
                for f in required:
                    if f not in features:
                        raise ValueError(f"Missing required feature '{f}' in request {i}")
                    if not isinstance(features[f], (int, float)):
                        raise ValueError(f"Invalid type for feature '{f}' in request {i}: expected number")

            # Snapshot model references under the lock (fast path — no inference here).
            with self.lock:
                if not self.is_ready:
                    raise HTTPException(status_code=503, detail="Models not ready")
                ensemble_active = self.ensemble_active
                if ensemble_active:
                    ttft_gated = self.ttft_gated
                    tpot_gated = self.tpot_gated
                else:
                    ttft_model = self.ttft_model
                    tpot_model = self.tpot_model
                    ttft_scaler = self.ttft_scaler
                    tpot_scaler = self.tpot_scaler

            # Heavy work: DataFrame construction + inference — no lock held.
            ttft_raw_data = []
            tpot_raw_data = []

            for features in features_list:
                ttft_entry = {
                    'kv_cache_percentage': features['kv_cache_percentage'],
                    'input_token_length': features['input_token_length'],
                    'num_request_waiting': features['num_request_waiting'],
                    'num_request_running': features['num_request_running'],
                    'prefix_cache_score': features['prefix_cache_score'],
                }
                if 'pod_type' in features:
                    ttft_entry['pod_type'] = features['pod_type']
                ttft_raw_data.append(ttft_entry)

                tpot_entry = {
                    'kv_cache_percentage': features['kv_cache_percentage'],
                    'input_token_length': features['input_token_length'],
                    'num_request_waiting': features['num_request_waiting'],
                    'num_request_running': features['num_request_running'],
                    'num_tokens_generated': features['num_tokens_generated'],
                }
                if 'pod_type' in features:
                    tpot_entry['pod_type'] = features['pod_type']
                tpot_raw_data.append(tpot_entry)

            n = len(features_list)
            ttft_results = np.zeros(n)
            tpot_results = np.zeros(n)

            if ensemble_active:
                noqueue_idx = [i for i, f in enumerate(features_list) if f['num_request_waiting'] == 0]
                queued_idx = [i for i, f in enumerate(features_list) if f['num_request_waiting'] > 0]

                for idx_list, regime, ttft_m, tpot_m, ttft_s, tpot_s in [
                    (noqueue_idx, "noqueue", ttft_gated.noqueue_model, tpot_gated.noqueue_model,
                     ttft_gated.noqueue_scaler, tpot_gated.noqueue_scaler),
                    (queued_idx, "queued", ttft_gated.queued_model, tpot_gated.queued_model,
                     ttft_gated.queued_scaler, tpot_gated.queued_scaler),
                ]:
                    if not idx_list:
                        continue
                    sub_ttft_raw = pd.DataFrame([ttft_raw_data[i] for i in idx_list])
                    sub_tpot_raw = pd.DataFrame([tpot_raw_data[i] for i in idx_list])
                    sub_ttft = self._prepare_features_for_ensemble(sub_ttft_raw, "ttft", regime)
                    sub_tpot = self._prepare_features_for_ensemble(sub_tpot_raw, "tpot", regime)
                    ttft_p, tpot_p = self._predict_with_models(
                        sub_ttft, sub_tpot, ttft_m, tpot_m, ttft_s, tpot_s
                    )
                    for j, orig_idx in enumerate(idx_list):
                        ttft_results[orig_idx] = ttft_p[j]
                        tpot_results[orig_idx] = tpot_p[j]

                return ttft_results, tpot_results
            else:
                df_ttft_raw = pd.DataFrame(ttft_raw_data)
                df_ttft_batch = self._prepare_features_with_interaction(df_ttft_raw, "ttft")

                df_tpot_raw = pd.DataFrame(tpot_raw_data)
                df_tpot_batch = self._prepare_features_with_interaction(df_tpot_raw, "tpot")

                ttft_preds, tpot_preds = self._predict_with_models(
                    df_ttft_batch, df_tpot_batch,
                    ttft_model, tpot_model, ttft_scaler, tpot_scaler
                )
                return ttft_preds, tpot_preds

        except ValueError as ve:
            logging.warning(f"Client error in predict_batch(): {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except HTTPException:
            raise
        except Exception as e:
            logging.error("Error in predict_batch():", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during batch prediction")

    def predict_batch_fast(self, reqs: list) -> Tuple[np.ndarray, np.ndarray]:
        """Fast-path bulk prediction from PredictionRequest objects.

        Skips r.dict() conversion and list-of-dicts DataFrame construction.
        Pydantic has already validated all fields at ingest, so no re-validation here.
        Builds DataFrames from dict-of-numpy-arrays instead of list-of-dicts,
        which avoids Pandas type-inference overhead (~5x faster DataFrame creation).
        """
        try:
            n = len(reqs)

            # Extract each feature column as a numpy array in one pass.
            # np.fromiter with count= pre-allocates and avoids intermediate lists.
            kv  = np.fromiter((r.kv_cache_percentage    for r in reqs), dtype=np.float64, count=n)
            itl = np.fromiter((r.input_token_length      for r in reqs), dtype=np.float64, count=n)
            nrw = np.fromiter((r.num_request_waiting     for r in reqs), dtype=np.float64, count=n)
            nrr = np.fromiter((r.num_request_running     for r in reqs), dtype=np.float64, count=n)
            ntg = np.fromiter((r.num_tokens_generated    for r in reqs), dtype=np.float64, count=n)
            pcs = np.fromiter((r.prefix_cache_score      for r in reqs), dtype=np.float64, count=n)
            if settings.ENABLE_TOKEN_IN_FLIGHT_FEATURES:
                pti = np.fromiter((r.prefill_tokens_in_flight for r in reqs), dtype=np.float64, count=n)
                dti = np.fromiter((r.decode_tokens_in_flight  for r in reqs), dtype=np.float64, count=n)

            # Snapshot model references under lock (no inference here).
            with self.lock:
                if not self.is_ready:
                    raise HTTPException(status_code=503, detail="Models not ready")
                ensemble_active = self.ensemble_active
                if ensemble_active:
                    ttft_gated = self.ttft_gated
                    tpot_gated = self.tpot_gated
                else:
                    ttft_model  = self.ttft_model
                    tpot_model  = self.tpot_model
                    ttft_scaler = self.ttft_scaler
                    tpot_scaler = self.tpot_scaler

            # Build DataFrames from dict-of-arrays: Pandas receives already-typed
            # numpy columns, skipping per-row type inference (the slow part of
            # pd.DataFrame(list_of_dicts)).
            df_ttft_raw = pd.DataFrame({
                'kv_cache_percentage': kv,
                'input_token_length':  itl,
                'num_request_waiting': nrw,
                'num_request_running': nrr,
                'prefix_cache_score':  pcs,
            })
            df_tpot_raw = pd.DataFrame({
                'kv_cache_percentage': kv,
                'input_token_length':  itl,
                'num_request_waiting': nrw,
                'num_request_running': nrr,
                'num_tokens_generated': ntg,
            })
            if settings.ENABLE_TOKEN_IN_FLIGHT_FEATURES:
                df_ttft_raw['prefill_tokens_in_flight'] = pti
                df_ttft_raw['decode_tokens_in_flight']  = dti
                df_tpot_raw['prefill_tokens_in_flight'] = pti
                df_tpot_raw['decode_tokens_in_flight']  = dti

            # Conditionally add pod_type column (only if any request uses it).
            if any(r.pod_type for r in reqs):
                pod_types = [r.pod_type for r in reqs]
                df_ttft_raw['pod_type'] = pod_types
                df_tpot_raw['pod_type'] = pod_types

            if ensemble_active:
                nrw_int = nrw.astype(int)
                noqueue_idx = np.where(nrw_int == 0)[0]
                queued_idx  = np.where(nrw_int  > 0)[0]

                ttft_results = np.zeros(n)
                tpot_results = np.zeros(n)

                for idx_arr, regime, ttft_m, tpot_m, ttft_s, tpot_s in [
                    (noqueue_idx, "noqueue", ttft_gated.noqueue_model, tpot_gated.noqueue_model,
                     ttft_gated.noqueue_scaler, tpot_gated.noqueue_scaler),
                    (queued_idx, "queued", ttft_gated.queued_model, tpot_gated.queued_model,
                     ttft_gated.queued_scaler, tpot_gated.queued_scaler),
                ]:
                    if len(idx_arr) == 0:
                        continue
                    # .copy() so that _prepare_features_with_interaction can
                    # add columns without triggering SettingWithCopyWarning.
                    sub_ttft = self._prepare_features_for_ensemble(
                        df_ttft_raw.iloc[idx_arr].copy(), "ttft", regime)
                    sub_tpot = self._prepare_features_for_ensemble(
                        df_tpot_raw.iloc[idx_arr].copy(), "tpot", regime)
                    ttft_p, tpot_p = self._predict_with_models(
                        sub_ttft, sub_tpot, ttft_m, tpot_m, ttft_s, tpot_s)
                    ttft_results[idx_arr] = ttft_p
                    tpot_results[idx_arr] = tpot_p

                return ttft_results, tpot_results
            else:
                df_ttft_batch = self._prepare_features_with_interaction(df_ttft_raw, "ttft")
                df_tpot_batch = self._prepare_features_with_interaction(df_tpot_raw, "tpot")
                return self._predict_with_models(
                    df_ttft_batch, df_tpot_batch,
                    ttft_model, tpot_model, ttft_scaler, tpot_scaler,
                )

        except HTTPException:
            raise
        except Exception as e:
            logging.error("Error in predict_batch_fast():", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during batch prediction")


# ---------------------------------------------------------------------------
# Instantiate singletons (each uvicorn worker fork gets its own copy)
# ---------------------------------------------------------------------------
model_syncer = ModelSyncer()
predictor = LightweightPredictor()

# FastAPI app
app = FastAPI(
    title="HTTP-based Latency Predictor",
    description="A prediction service that downloads regression models (quantile or mean) from training server via HTTP.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    kv_cache_percentage: float = Field(..., ge=0.0, le=1.0)
    input_token_length: int = Field(..., ge=0)
    num_request_waiting: int = Field(..., ge=0)
    num_request_running: int = Field(..., ge=0)
    num_tokens_generated: int = Field(..., ge=0)
    prefix_cache_score: float = Field(..., ge=0.0, le=1.0)
    pod_type: Optional[str] = Field(default="", description="Pod type: 'prefill', 'decode', or '' for monolithic")
    prefill_tokens_in_flight: int = Field(default=0, ge=0)
    decode_tokens_in_flight: int = Field(default=0, ge=0)


class PredictionResponse(BaseModel):
    ttft_ms: float = Field(..., description="Predicted TTFT in ms (mean or quantile depending on objective)")
    tpot_ms: float = Field(..., description="Predicted TPOT in ms (mean or quantile depending on objective)")
    predicted_at: datetime
    model_type: str
    objective_type: str
    quantile: float
    last_model_load: Optional[datetime]


class StatusResponse(BaseModel):
    is_ready: bool
    model_type: str
    objective_type: str
    quantile: float
    last_model_load: Optional[datetime]
    training_server_url: str
    models_exist: dict
    ensemble_active: bool = False


class BulkPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(..., min_items=1, max_items=10000)


class BulkPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    processing_time_ms: float


class BulkPredictionError(BaseModel):
    index: int
    error: str
    request: PredictionRequest


class BulkPredictionResponseWithErrors(BaseModel):
    predictions: List[Optional[PredictionResponse]]
    errors: List[BulkPredictionError]
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    processing_time_ms: float


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse)
async def status_endpoint():
    models_exist = {
        "ttft_model": os.path.exists(settings.LOCAL_TTFT_MODEL_PATH),
        "tpot_model": os.path.exists(settings.LOCAL_TPOT_MODEL_PATH),
    }
    if predictor.model_type == ModelType.BAYESIAN_RIDGE:
        models_exist.update({
            "ttft_scaler": os.path.exists(settings.LOCAL_TTFT_SCALER_PATH),
            "tpot_scaler": os.path.exists(settings.LOCAL_TPOT_SCALER_PATH),
        })
    if settings.ENSEMBLE_MODE:
        models_exist.update({
            "ttft_gated": os.path.exists(settings.LOCAL_TTFT_GATED_MODEL_PATH),
            "tpot_gated": os.path.exists(settings.LOCAL_TPOT_GATED_MODEL_PATH),
        })
    return StatusResponse(
        is_ready=predictor.is_ready,
        model_type=predictor.model_type.value,
        objective_type=predictor.objective_type.value,
        quantile=predictor.quantile,
        last_model_load=predictor.last_load,
        training_server_url=settings.TRAINING_SERVER_URL,
        models_exist=models_exist,
        ensemble_active=predictor.ensemble_active,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    try:
        ttft_pred, tpot_pred = predictor.predict(request.dict())
        return PredictionResponse(
            ttft_ms=max(0, ttft_pred),
            tpot_ms=max(0, tpot_pred),
            predicted_at=datetime.now(timezone.utc),
            model_type=predictor.model_type.value,
            objective_type=predictor.objective_type.value,
            quantile=predictor.quantile,
            last_model_load=predictor.last_load,
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during prediction")


@app.post("/predict/bulk/strict")
async def predict_bulk_strict_endpoint(request: BulkPredictionRequest):
    start_time = time.time()
    try:
        reqs = request.requests
        n = len(reqs)

        # Fast path: builds DataFrames from numpy arrays, skips r.dict() x1000.
        ttft_preds, tpot_preds = predictor.predict_batch_fast(reqs)

        # Clip negatives once via numpy, then convert to Python list in one call.
        ttft_vals = np.maximum(ttft_preds, 0.0).tolist()
        tpot_vals = np.maximum(tpot_preds, 0.0).tolist()

        # Cache metadata fields once instead of re-evaluating per prediction.
        current_time  = datetime.now(timezone.utc)
        _model_type   = predictor.model_type.value
        _obj_type     = predictor.objective_type.value
        _quantile     = predictor.quantile
        _last_load    = predictor.last_load

        # Build response as plain dicts — avoids 1000x Pydantic object construction.
        # ORJSONResponse serializes datetime, float, and str natively, much faster
        # than stdlib json used by FastAPI's default JSONResponse.
        predictions = [
            {
                "ttft_ms": ttft_vals[i],
                "tpot_ms": tpot_vals[i],
                "predicted_at": current_time,
                "model_type": _model_type,
                "objective_type": _obj_type,
                "quantile": _quantile,
                "last_model_load": _last_load,
            }
            for i in range(n)
        ]

        return ORJSONResponse({
            "predictions": predictions,
            "total_requests": n,
            "successful_predictions": n,
            "failed_predictions": 0,
            "processing_time_ms": (time.time() - start_time) * 1000,
        })
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Bulk prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Bulk prediction failed")


@app.post("/predict/bulk", response_model=BulkPredictionResponseWithErrors)
async def predict_bulk_endpoint(request: BulkPredictionRequest):
    start_time = time.time()

    valid_requests = []
    valid_indices = []
    errors = []
    required = [
        'kv_cache_percentage', 'input_token_length', 'num_request_waiting',
        'num_request_running', 'num_tokens_generated', 'prefix_cache_score',
    ]

    for i, pred_request in enumerate(request.requests):
        try:
            features = pred_request.dict()
            for f in required:
                if f not in features:
                    raise ValueError(f"Missing required feature: {f}")
                if not isinstance(features[f], (int, float)):
                    raise ValueError(f"Invalid type for feature {f}: expected number")
            valid_requests.append(features)
            valid_indices.append(i)
        except Exception as e:
            errors.append(BulkPredictionError(index=i, error=str(e), request=pred_request))

    predictions: List[Optional[PredictionResponse]] = [None] * len(request.requests)
    successful_count = len(valid_requests)
    failed_count = len(errors)

    if valid_requests:
        try:
            ttft_preds, tpot_preds = predictor.predict_batch(valid_requests)
            current_time = datetime.now(timezone.utc)
            for batch_idx, original_idx in enumerate(valid_indices):
                predictions[original_idx] = PredictionResponse(
                    ttft_ms=max(0, ttft_preds[batch_idx]),
                    tpot_ms=max(0, tpot_preds[batch_idx]),
                    predicted_at=current_time,
                    model_type=predictor.model_type.value,
                    objective_type=predictor.objective_type.value,
                    quantile=predictor.quantile,
                    last_model_load=predictor.last_load,
                )
        except Exception as e:
            for original_idx in valid_indices:
                errors.append(BulkPredictionError(
                    index=original_idx,
                    error=f"Batch prediction error: {e}",
                    request=request.requests[original_idx],
                ))
                predictions[original_idx] = None
            successful_count = 0
            failed_count = len(request.requests)

    return BulkPredictionResponseWithErrors(
        predictions=predictions,
        errors=errors,
        total_requests=len(request.requests),
        successful_predictions=successful_count,
        failed_predictions=failed_count,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


@app.post("/reload")
async def reload_models():
    try:
        synced = model_syncer.sync_models()
        loaded = predictor.load_models()
        return {
            "synced": synced,
            "loaded": loaded,
            "is_ready": predictor.is_ready,
            "model_type": predictor.model_type.value,
            "objective_type": predictor.objective_type.value,
            "quantile": predictor.quantile,
            "last_load_time": predictor.last_load,
        }
    except Exception as e:
        logging.error(f"Error reloading models: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading models: {e}")


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok", "service": "http-based-latency-predictor"}


@app.get("/readyz", status_code=status.HTTP_200_OK)
async def readiness_check():
    if not predictor.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models are not ready")
    return {"status": "ready", "model_type": predictor.model_type.value, "objective_type": predictor.objective_type.value, "quantile": predictor.quantile}


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "HTTP-based Latency Predictor is running",
        "model_type": predictor.model_type.value,
        "objective_type": predictor.objective_type.value,
        "quantile": predictor.quantile,
        "is_ready": predictor.is_ready,
        "sync_interval": settings.MODEL_SYNC_INTERVAL_SEC,
        "training_server": settings.TRAINING_SERVER_URL,
    }


@app.on_event("startup")
async def startup():
    logging.info(f"Starting up (PID={os.getpid()})...")
    # Initial sync & load — may fail if training server isn't up yet
    model_syncer.sync_models()
    predictor.load_models()
    if predictor.is_ready:
        logging.info(f"Models ready at startup (PID={os.getpid()})")
    else:
        logging.info(f"Models not ready yet, sync thread will retry (PID={os.getpid()})")
    model_syncer.start()


@app.on_event("shutdown")
async def shutdown():
    logging.info(f"Shutting down (PID={os.getpid()})...")
    model_syncer.shutdown()


if __name__ == "__main__":
    uvicorn.run("__main__:app", host=settings.HOST, port=settings.PORT, reload=True)