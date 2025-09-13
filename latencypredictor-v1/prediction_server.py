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
    """Handles inference using loaded quantile regression models."""

    def __init__(self):
        mt = settings.MODEL_TYPE
        if mt == ModelType.XGBOOST and not XGBOOST_AVAILABLE:
            logging.warning("Falling back to Bayesian Ridge")
            mt = ModelType.BAYESIAN_RIDGE
        self.model_type = mt
        self.quantile = settings.QUANTILE_ALPHA
        self.ttft_model = None
        self.tpot_model = None
        self.ttft_scaler = None
        self.tpot_scaler = None
        self.lock = threading.RLock()
        self.last_load: Optional[datetime] = None
        logging.info(f"Predictor type: {self.model_type}, quantile: {self.quantile}")

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
        """Make quantile predictions using the loaded models."""
        try:
            with self.lock:
                if not self.is_ready:
                    raise HTTPException(status_code=503, detail="Models not ready")
                
                # Updated required features to include prefix_cache_score
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
                df_tpot = pd.DataFrame([{col: features[col] for col in tpot_cols}])

                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    # Use scaling for Bayesian Ridge
                    ttft_scaled = self.ttft_scaler.transform(df_ttft)
                    tpot_scaled = self.tpot_scaler.transform(df_tpot)

                    ttft_pred_mean, ttft_std = self.ttft_model.predict(ttft_scaled, return_std=True)
                    tpot_pred_mean, tpot_std = self.tpot_model.predict(tpot_scaled, return_std=True)
                    
                    # Approximate quantile prediction by adding factor to mean
                    # This matches the logic in the training server
                    std_factor = 1.28 if self.quantile == 0.9 else (2.0 if self.quantile == 0.95 else 0.674)
                    ttft_pred = ttft_pred_mean[0] + std_factor * ttft_std[0]
                    tpot_pred = tpot_pred_mean[0] + std_factor * tpot_std[0]
                    
                    return ttft_pred, tpot_pred, ttft_std[0], tpot_std[0]
                
                else:  # XGBoost with true quantile regression
                    # XGBoost quantile regression directly predicts the quantile
                    ttft_pred = self.ttft_model.predict(df_ttft)
                    tpot_pred = self.tpot_model.predict(df_tpot)
                    
                    # For XGBoost quantile regression, uncertainty estimation is more complex
                    # We'll use a simple heuristic based on the quantile value and prediction
                    # This is a rough approximation - ideally you'd train additional models for uncertainty
                    ttft_std = ttft_pred[0] * 0.15  # 15% of prediction as uncertainty estimate
                    tpot_std = tpot_pred[0] * 0.15
                    
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
    ttft_uncertainty: float = Field(..., description="Uncertainty estimate for TTFT prediction")
    tpot_uncertainty: float = Field(..., description="Uncertainty estimate for TPOT prediction")
    ttft_prediction_bounds: Tuple[float, float] = Field(..., description="Approximate prediction bounds for TTFT")
    tpot_prediction_bounds: Tuple[float, float] = Field(..., description="Approximate prediction bounds for TPOT")
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
    requests: List[PredictionRequest] = Field(..., min_items=1, max_items=100, description="List of prediction requests (max 100)")

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
        ttft_pred, tpot_pred, ttft_std, tpot_std = predictor.predict(request.dict())
        
        # Ensure non-negative predictions
        ttft_pred = max(0, ttft_pred)
        tpot_pred = max(0, tpot_pred)
        
        # Calculate approximate confidence bounds
        # For quantile predictions, these represent uncertainty around the quantile estimate
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
            quantile=predictor.quantile,
            last_model_load=predictor.last_load 
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction")
    
    
# Add this endpoint after the existing predict endpoint
@app.post("/predict/bulk", response_model=BulkPredictionResponseWithErrors)
async def predict_bulk_endpoint(request: BulkPredictionRequest):
    """Make bulk quantile latency predictions."""
    start_time = time.time()
    
    predictions = []
    errors = []
    successful_count = 0
    failed_count = 0
    
    for i, pred_request in enumerate(request.requests):
        try:
            ttft_pred, tpot_pred, ttft_std, tpot_std = predictor.predict(pred_request.dict())
            
            # Ensure non-negative predictions
            ttft_pred = max(0, ttft_pred)
            tpot_pred = max(0, tpot_pred)
            
            # Calculate approximate confidence bounds
            ttft_bounds = (max(0, ttft_pred - 2*ttft_std), ttft_pred + 2*ttft_std)
            tpot_bounds = (max(0, tpot_pred - 2*tpot_std), tpot_pred + 2*tpot_std)
            
            prediction_response = PredictionResponse(
                ttft_ms=ttft_pred,
                tpot_ms=tpot_pred,
                ttft_uncertainty=ttft_std,
                tpot_uncertainty=tpot_std,
                ttft_prediction_bounds=ttft_bounds,
                tpot_prediction_bounds=tpot_bounds,
                predicted_at=datetime.now(timezone.utc),
                model_type=predictor.model_type.value,
                quantile=predictor.quantile,
                last_model_load=predictor.last_load 
            )
            
            predictions.append(prediction_response)
            successful_count += 1
            
        except HTTPException as he:
            predictions.append(None)
            errors.append(BulkPredictionError(
                index=i,
                error=he.detail,
                request=pred_request
            ))
            failed_count += 1
            
        except Exception as e:
            predictions.append(None)
            errors.append(BulkPredictionError(
                index=i,
                error=f"Internal error: {str(e)}",
                request=pred_request
            ))
            failed_count += 1
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return BulkPredictionResponseWithErrors(
        predictions=predictions,
        errors=errors,
        total_requests=len(request.requests),
        successful_predictions=successful_count,
        failed_predictions=failed_count,
        processing_time_ms=processing_time_ms
    )


# Optional: Add a simpler bulk endpoint that fails fast on any error
@app.post("/predict/bulk/strict", response_model=BulkPredictionResponse)
async def predict_bulk_strict_endpoint(request: BulkPredictionRequest):
    """Make bulk quantile latency predictions (fails on any single error)."""
    start_time = time.time()
    
    predictions = []
    
    try:
        for pred_request in request.requests:
            ttft_pred, tpot_pred, ttft_std, tpot_std = predictor.predict(pred_request.dict())
            
            # Ensure non-negative predictions
            ttft_pred = max(0, ttft_pred)
            tpot_pred = max(0, tpot_pred)
            
            # Calculate approximate confidence bounds
            ttft_bounds = (max(0, ttft_pred - 2*ttft_std), ttft_pred + 2*ttft_std)
            tpot_bounds = (max(0, tpot_pred - 2*tpot_std), tpot_pred + 2*tpot_std)
            
            prediction_response = PredictionResponse(
                ttft_ms=ttft_pred,
                tpot_ms=tpot_pred,
                ttft_uncertainty=ttft_std,
                tpot_uncertainty=tpot_std,
                ttft_prediction_bounds=ttft_bounds,
                tpot_prediction_bounds=tpot_bounds,
                predicted_at=datetime.now(timezone.utc),
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