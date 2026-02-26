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


class ModelType(str, Enum):
    BAYESIAN_RIDGE = "bayesian_ridge"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


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

    HOST: str = os.getenv("PREDICT_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PREDICT_PORT", "8001"))

    HTTP_TIMEOUT: int = int(os.getenv("HTTP_TIMEOUT", "30"))
    DOWNLOAD_RETRIES: int = int(os.getenv("DOWNLOAD_RETRIES", "3"))


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

        for path in [
            settings.LOCAL_TTFT_MODEL_PATH,
            settings.LOCAL_TPOT_MODEL_PATH,
            settings.LOCAL_TTFT_SCALER_PATH,
            settings.LOCAL_TPOT_SCALER_PATH,
        ]:
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
            for name, path in to_sync:
                if self._download_model_if_newer(name, path):
                    updated = True
            return updated

    def _sync_loop(self):
        while not self._shutdown_event.is_set():
            try:
                updated = self.sync_models()
                # Always try to load models if predictor isn't ready yet,
                # even if no new files were downloaded (another worker may
                # have already written them to disk).
                if updated or not predictor.is_ready:
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
    """Handles inference using loaded quantile regression models."""

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
        self.ttft_model = None
        self.tpot_model = None
        self.ttft_scaler = None
        self.tpot_scaler = None
        self.lock = threading.RLock()
        self.last_load: Optional[datetime] = None
        # Track checksums to avoid redundant reloads
        self._loaded_checksums: dict = {}
        logging.info(f"Predictor type: {self.model_type}, quantile: {self.quantile}")

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

            feature_cols = [
                'kv_cache_percentage', 'input_token_length',
                'num_request_waiting', 'num_request_running',
                'prefix_cache_score', 'effective_input_tokens',
                'prefill_score_bucket', 'pod_type_cat',
            ]
            return df[feature_cols]

        else:  # tpot
            feature_cols = [
                'kv_cache_percentage', 'input_token_length',
                'num_request_waiting', 'num_request_running',
                'num_tokens_generated', 'pod_type_cat',
            ]
            return df[feature_cols]

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

            current_checksums = {}
            for name, path in paths_to_check:
                cs = _file_checksum(path)
                if cs is None:
                    # File doesn't exist yet — can't load
                    if not self.is_ready:
                        logging.debug(f"Model file {path} not found yet")
                    return False
                current_checksums[name] = cs

            # Skip reload if nothing changed and we're already ready
            if self.is_ready and current_checksums == self._loaded_checksums:
                return True

            with self.lock:
                new_ttft = joblib.load(settings.LOCAL_TTFT_MODEL_PATH)
                new_tpot = joblib.load(settings.LOCAL_TPOT_MODEL_PATH)

                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    new_ttft_scaler = joblib.load(settings.LOCAL_TTFT_SCALER_PATH)
                    new_tpot_scaler = joblib.load(settings.LOCAL_TPOT_SCALER_PATH)
                    self.ttft_scaler = new_ttft_scaler
                    self.tpot_scaler = new_tpot_scaler

                self.ttft_model = new_ttft
                self.tpot_model = new_tpot
                self._loaded_checksums = current_checksums
                self.last_load = datetime.now(timezone.utc)

                logging.info(f"Models loaded (PID={os.getpid()})")
                return True

        except Exception as e:
            logging.error(f"Load error: {e}")
            return False

    def predict(self, features: dict) -> Tuple[float, float]:
        """Make quantile predictions using the loaded models."""
        try:
            with self.lock:
                if not self.is_ready:
                    raise HTTPException(status_code=503, detail="Models not ready")

                required = [
                    'kv_cache_percentage', 'input_token_length', 'num_request_waiting',
                    'num_request_running', 'num_tokens_generated', 'prefix_cache_score',
                ]
                for f in required:
                    if f not in features:
                        raise ValueError(f"Missing required feature: {f}")
                    if not isinstance(features[f], (int, float)):
                        raise ValueError(f"Invalid type for feature {f}: expected number")

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

                df_ttft_raw = pd.DataFrame([ttft_raw_data])
                if 'pod_type' in features:
                    df_ttft_raw['pod_type'] = features['pod_type']
                df_ttft = self._prepare_features_with_interaction(df_ttft_raw, "ttft")

                df_tpot_raw = pd.DataFrame([tpot_raw_data])
                if 'pod_type' in features:
                    df_tpot_raw['pod_type'] = features['pod_type']
                df_tpot = self._prepare_features_with_interaction(df_tpot_raw, "tpot")

                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    ttft_for_scale = df_ttft.drop(columns=['prefill_score_bucket'], errors='ignore')
                    if 'pod_type_cat' in ttft_for_scale.columns:
                        ttft_for_scale = pd.get_dummies(
                            ttft_for_scale, columns=['pod_type_cat'],
                            prefix='pod_type', drop_first=False,
                        )
                    ttft_scaled = self.ttft_scaler.transform(ttft_for_scale)

                    tpot_for_scale = df_tpot.copy()
                    if 'pod_type_cat' in tpot_for_scale.columns:
                        tpot_for_scale = pd.get_dummies(
                            tpot_for_scale, columns=['pod_type_cat'],
                            prefix='pod_type', drop_first=False,
                        )
                    tpot_scaled = self.tpot_scaler.transform(tpot_for_scale)

                    ttft_pred_mean, ttft_std = self.ttft_model.predict(ttft_scaled, return_std=True)
                    tpot_pred_mean, tpot_std = self.tpot_model.predict(tpot_scaled, return_std=True)

                    std_factor = 1.28 if self.quantile == 0.9 else (2.0 if self.quantile == 0.95 else 0.674)
                    return (
                        ttft_pred_mean[0] + std_factor * ttft_std[0],
                        tpot_pred_mean[0] + std_factor * tpot_std[0],
                    )

                else:  # XGBoost or LightGBM
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

                df_ttft_raw = pd.DataFrame(ttft_raw_data)
                df_ttft_batch = self._prepare_features_with_interaction(df_ttft_raw, "ttft")

                df_tpot_raw = pd.DataFrame(tpot_raw_data)
                df_tpot_batch = self._prepare_features_with_interaction(df_tpot_raw, "tpot")

                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    ttft_for_scale = df_ttft_batch.drop(columns=['prefill_score_bucket'], errors='ignore')
                    if 'pod_type_cat' in ttft_for_scale.columns:
                        ttft_for_scale = pd.get_dummies(
                            ttft_for_scale, columns=['pod_type_cat'],
                            prefix='pod_type', drop_first=False,
                        )
                    ttft_scaled = self.ttft_scaler.transform(ttft_for_scale)

                    tpot_for_scale = df_tpot_batch.copy()
                    if 'pod_type_cat' in tpot_for_scale.columns:
                        tpot_for_scale = pd.get_dummies(
                            tpot_for_scale, columns=['pod_type_cat'],
                            prefix='pod_type', drop_first=False,
                        )
                    tpot_scaled = self.tpot_scaler.transform(tpot_for_scale)

                    ttft_pred_mean, ttft_std = self.ttft_model.predict(ttft_scaled, return_std=True)
                    tpot_pred_mean, tpot_std = self.tpot_model.predict(tpot_scaled, return_std=True)

                    std_factor = 1.28 if self.quantile == 0.9 else (2.0 if self.quantile == 0.95 else 0.674)
                    return (
                        ttft_pred_mean + std_factor * ttft_std,
                        tpot_pred_mean + std_factor * tpot_std,
                    )

                else:  # XGBoost or LightGBM
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


# ---------------------------------------------------------------------------
# Instantiate singletons (each uvicorn worker fork gets its own copy)
# ---------------------------------------------------------------------------
model_syncer = ModelSyncer()
predictor = LightweightPredictor()

# FastAPI app
app = FastAPI(
    title="HTTP-based Quantile Latency Predictor",
    description="A prediction service that downloads quantile regression models from training server via HTTP.",
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


class PredictionResponse(BaseModel):
    ttft_ms: float = Field(..., description=f"Predicted {settings.QUANTILE_ALPHA:.0%} quantile TTFT in ms")
    tpot_ms: float = Field(..., description=f"Predicted {settings.QUANTILE_ALPHA:.0%} quantile TPOT in ms")
    predicted_at: datetime
    model_type: str
    quantile: float
    last_model_load: Optional[datetime]


class StatusResponse(BaseModel):
    is_ready: bool
    model_type: str
    quantile: float
    last_model_load: Optional[datetime]
    training_server_url: str
    models_exist: dict


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
    return StatusResponse(
        is_ready=predictor.is_ready,
        model_type=predictor.model_type.value,
        quantile=predictor.quantile,
        last_model_load=predictor.last_load,
        training_server_url=settings.TRAINING_SERVER_URL,
        models_exist=models_exist,
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
            quantile=predictor.quantile,
            last_model_load=predictor.last_load,
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during prediction")


@app.post("/predict/bulk/strict", response_model=BulkPredictionResponse)
async def predict_bulk_strict_endpoint(request: BulkPredictionRequest):
    start_time = time.time()
    try:
        features_list = [r.dict() for r in request.requests]
        ttft_preds, tpot_preds = predictor.predict_batch(features_list)

        current_time = datetime.now(timezone.utc)
        predictions = [
            PredictionResponse(
                ttft_ms=max(0, ttft_preds[i]),
                tpot_ms=max(0, tpot_preds[i]),
                predicted_at=current_time,
                model_type=predictor.model_type.value,
                quantile=predictor.quantile,
                last_model_load=predictor.last_load,
            )
            for i in range(len(request.requests))
        ]

        return BulkPredictionResponse(
            predictions=predictions,
            total_requests=len(request.requests),
            successful_predictions=len(predictions),
            failed_predictions=0,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
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
            "quantile": predictor.quantile,
            "last_load_time": predictor.last_load,
        }
    except Exception as e:
        logging.error(f"Error reloading models: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading models: {e}")


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok", "service": "http-based-quantile-latency-predictor"}


@app.get("/readyz", status_code=status.HTTP_200_OK)
async def readiness_check():
    if not predictor.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models are not ready")
    return {"status": "ready", "model_type": predictor.model_type.value, "quantile": predictor.quantile}


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "HTTP-based Quantile Latency Predictor is running",
        "model_type": predictor.model_type.value,
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