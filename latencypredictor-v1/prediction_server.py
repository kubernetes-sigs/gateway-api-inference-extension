import os
import shutil
import time
import logging
import threading
import requests
from datetime import datetime, timezone
from typing import Tuple, Optional
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


class ModelType(str, Enum):
    BAYESIAN_RIDGE = "bayesian_ridge"
    XGBOOST = "xgboost"


class PredictSettings:
    """Configuration for the prediction server."""

    # Training server URL
    TRAINING_SERVER_URL: str = os.getenv("TRAINING_SERVER_URL", "http://training-service:8000")

    # Local model paths
    LOCAL_TTFT_MODEL_PATH: str = os.getenv("LOCAL_TTFT_MODEL_PATH", "/local_models/ttft.joblib")
    LOCAL_TPOT_MODEL_PATH: str = os.getenv("LOCAL_TPOT_MODEL_PATH", "/local_models/tpot.joblib")
    LOCAL_TTFT_SCALER_PATH: str = os.getenv("LOCAL_TTFT_SCALER_PATH", "/local_models/ttft_scaler.joblib")
    LOCAL_TPOT_SCALER_PATH: str = os.getenv("LOCAL_TPOT_SCALER_PATH", "/local_models/tpot_scaler.joblib")

    # Sync interval and model type
    MODEL_SYNC_INTERVAL_SEC: int = int(os.getenv("MODEL_SYNC_INTERVAL_SEC", "10"))
    MODEL_TYPE: ModelType = ModelType(os.getenv("LATENCY_MODEL_TYPE", "xgboost"))

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
    """Handles inference using loaded models."""

    def __init__(self):
        mt = settings.MODEL_TYPE
        if mt == ModelType.XGBOOST and not XGBOOST_AVAILABLE:
            logging.warning("Falling back to Bayesian Ridge")
            mt = ModelType.BAYESIAN_RIDGE
        self.model_type = mt
        self.ttft_model = None
        self.tpot_model = None
        self.ttft_scaler = None
        self.tpot_scaler = None
        self.lock = threading.RLock()
        self.last_load: Optional[datetime] = None
        logging.info(f"Predictor type: {self.model_type}")

    @property
    def is_ready(self) -> bool:
        with self.lock:
            if self.model_type == ModelType.BAYESIAN_RIDGE:
                return all([self.ttft_model, self.tpot_model, self.ttft_scaler, self.tpot_scaler])
            return all([self.ttft_model, self.tpot_model])

    def load_models(self) -> bool:
        try:
            with self.lock:
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
                    logging.info("Models loaded")
                    return True
                logging.warning("Models missing after load")
                return False
        except Exception as e:
            logging.error(f"Load error: {e}")
            return False

    def predict(self, features: dict) -> Tuple[float, float, float, float]:
        # Prediction logic unchanged...
        try:
            with self.lock:
                if not self.is_ready:
                    raise HTTPException(status_code=503, detail="Models not ready")
                required = ['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'num_tokens_generated']
                for f in required:
                    if f not in features:
                        raise ValueError(f"Missing required feature: {f}")
                    if not isinstance(features[f], (int, float)):
                        raise ValueError(f"Invalid type for feature {f}: expected number")

                ttft_cols = ['kv_cache_percentage','input_token_length','num_request_waiting','num_request_running']
                tpot_cols = ['kv_cache_percentage','input_token_length','num_request_waiting','num_request_running','num_tokens_generated']
                
                # Create DataFrames for predictions
                df_ttft = pd.DataFrame([{col: features[col] for col in ttft_cols}])
                df_tpot = pd.DataFrame([{col: features[col] for col in tpot_cols}])

                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    # Use scaling for Bayesian Ridge
                    ttft_scaled = self.ttft_scaler.transform(df_ttft)
                    tpot_scaled = self.tpot_scaler.transform(df_tpot)

                    ttft_pred, ttft_std = self.ttft_model.predict(ttft_scaled, return_std=True)
                    tpot_pred, tpot_std = self.tpot_model.predict(tpot_scaled, return_std=True)
                    return ttft_pred[0], tpot_pred[0], ttft_std[0], tpot_std[0]
                
                else:  # XGBoost
                    # XGBoost doesn't need scaling and doesn't provide uncertainty
                    ttft_pred = self.ttft_model.predict(df_ttft)
                    tpot_pred = self.tpot_model.predict(df_tpot)
                    
                    # For XGBoost, we'll estimate uncertainty as a percentage of the prediction
                    # This is a simple heuristic - in practice you might want to use quantile regression
                    # or other methods for uncertainty estimation
                    ttft_std = ttft_pred[0] * 0.1  # 10% of prediction as uncertainty
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


# Instantiate
model_syncer = ModelSyncer()
predictor = LightweightPredictor()

# FastAPI app
app = FastAPI(
    title="HTTP-based Latency Predictor",
    description="A prediction service that downloads models from training server via HTTP.",
    version="1.0.0"
)


# Pydantic models
class PredictionRequest(BaseModel):
    kv_cache_percentage: float = Field(..., ge=0.0, le=1.0)
    input_token_length: int = Field(..., ge=0)
    num_request_waiting: int = Field(..., ge=0)
    num_request_running: int = Field(..., ge=0)
    num_tokens_generated: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    ttft_ms: float
    tpot_ms: float
    ttft_uncertainty: float
    tpot_uncertainty: float
    ttft_prediction_bounds: Tuple[float, float]
    tpot_prediction_bounds: Tuple[float, float]
    predicted_at: datetime
    model_type: str
    last_model_load: Optional[datetime]


class StatusResponse(BaseModel):
    is_ready: bool
    model_type: str
    last_model_load: Optional[datetime]
    training_server_url: str
    models_exist: dict


# API endpoints


# Fix the status endpoint - change last_load_time to last_load:

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
        last_model_load=predictor.last_load,  # ✅ Fixed: changed from last_load_time to last_load
        training_server_url=settings.TRAINING_SERVER_URL,
        models_exist=models_exist
    )

# Also fix the predict endpoint:
@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """Make latency predictions."""
    try:
        ttft_pred, tpot_pred, ttft_std, tpot_std = predictor.predict(request.dict())
        
        # Ensure non-negative predictions
        ttft_pred = max(0, ttft_pred)
        tpot_pred = max(0, tpot_pred)
        
        # Calculate 95% confidence bounds (±2 standard deviations)
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
            last_model_load=predictor.last_load 
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction")

# And fix the reload endpoint:
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
            "last_load_time": predictor.last_load 
        }
    except Exception as e:
        logging.error(f"Error reloading models: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")

@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "http-based-latency-predictor"}


@app.get("/readyz", status_code=status.HTTP_200_OK)
async def readiness_check():
    """Readiness check endpoint."""
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Models are not ready"
        )
    return {"status": "ready", "model_type": predictor.model_type.value}




@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "message": "HTTP-based Latency Predictor is running",
        "model_type": predictor.model_type.value,
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