import os
import random
import time
import logging
import threading
from datetime import datetime, timezone
from collections import deque
from typing import Any, Dict, List, Tuple

from fastapi.responses import Response  # Fixed import

import joblib
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
class Settings:
    """
    Configuration class for the latency predictor server.
    Reads settings from environment variables with sensible defaults.
    """
    TTFT_MODEL_PATH: str = os.getenv("LATENCY_TTFT_MODEL_PATH", "/tmp/models/ttft.joblib")
    TPOT_MODEL_PATH: str = os.getenv("LATENCY_TPOT_MODEL_PATH", "/tmp/models/tpot.joblib")
    TTFT_SCALER_PATH: str = os.getenv("LATENCY_TTFT_SCALER_PATH", "/tmp/models/ttft_scaler.joblib")
    TPOT_SCALER_PATH: str = os.getenv("LATENCY_TPOT_SCALER_PATH", "/tmp/models/tpot_scaler.joblib")
    RETRAINING_INTERVAL_SEC: int = int(os.getenv("LATENCY_RETRAINING_INTERVAL_SEC", 1800))
    MIN_SAMPLES_FOR_RETRAIN: int = int(os.getenv("LATENCY_MIN_SAMPLES_FOR_RETRAIN", 100))
    MAX_TRAINING_DATA_SIZE_PER_BUCKET: int = int(os.getenv("LATENCY_MAX_TRAINING_DATA_SIZE_PER_BUCKET", 10000))

settings = Settings()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LatencyPredictor:
    """
    Manages model training, prediction, and data handling.
    """
    def __init__(self):
        self.num_buckets = int(1.0 / 0.05)
        self.bucket_size = settings.MAX_TRAINING_DATA_SIZE_PER_BUCKET 

        # Data buckets for sampling
        self.ttft_data_buckets = {i: deque(maxlen=self.bucket_size) for i in range(self.num_buckets)}
        self.tpot_data_buckets = {i: deque(maxlen=self.bucket_size) for i in range(self.num_buckets)}

        self.ttft_model = None
        self.tpot_model = None
        self.ttft_scaler = None
        self.tpot_scaler = None

        self.lock = threading.Lock()
        self.last_retrain_time = None
        self._shutdown_event = threading.Event()
        self._training_thread: threading.Thread = None

    def shutdown(self):
        """Signal the training thread to exit and join it."""
        self._shutdown_event.set()
        if self._training_thread is not None:
            self._training_thread.join()

    @property
    def is_ready(self) -> bool:
        """Checks if all models and scalers are loaded/trained."""
        return all([self.ttft_model, self.tpot_model, self.ttft_scaler, self.tpot_scaler])

    @is_ready.setter
    def is_ready(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("is_ready must be a boolean value.")
        self._is_ready_override = value

    def _all_samples(self, buckets: dict) -> list:
        samples = []
        for dq in buckets.values():
            samples.extend(dq)
        return samples

    def _train_model_with_scaling(self, features: pd.DataFrame, target: pd.Series) -> Tuple[BayesianRidge, StandardScaler]:
        try:
            if len(features) == 0 or len(target) == 0:
                raise ValueError("Empty training data")
            if features.isnull().any().any() or target.isnull().any():
                raise ValueError("Training data contains NaN values")
            if np.isinf(features.values).any() or np.isinf(target.values).any():
                raise ValueError("Training data contains infinite values")

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
                raise ValueError("Scaling produced invalid values")

            model = BayesianRidge(compute_score=True)
            model.fit(features_scaled, target)
            return model, scaler
        except Exception as e:
            logging.error(f"Error in _train_model_with_scaling: {e}", exc_info=True)
            raise

    def _create_default_model(self, model_type: str) -> Tuple[BayesianRidge, StandardScaler]:
        """Creates and trains a simple default model with initial priors."""
        try:
            logging.info(f"Creating default '{model_type}' model with priors.")
            if model_type == "ttft":
                features = pd.DataFrame({
                    'kv_cache_percentage': [0.0, ],
                    'input_token_length': [1, ],
                    'num_request_waiting': [0, ],
                    'num_request_running': [0, ]
                })
                target = pd.Series([10,])
            else:
                features = pd.DataFrame({
                    'kv_cache_percentage': [0.0],
                    'num_request_waiting': [0, ],
                    'num_request_running': [0, ],
                    'num_tokens_generated': [1,]
                })
                target = pd.Series([10.0])
            return self._train_model_with_scaling(features, target)
        except Exception as e:
            logging.error(f"Error creating default model for {model_type}: {e}", exc_info=True)
            raise

    def train(self):
        try:
            with self.lock:
                ttft_snap = list(self._all_samples(self.ttft_data_buckets))
                tpot_snap = list(self._all_samples(self.tpot_data_buckets))
                total = len(ttft_snap) + len(tpot_snap)
                if total < settings.MIN_SAMPLES_FOR_RETRAIN:
                    logging.info(f"Skipping training: only {total} samples (< {settings.MIN_SAMPLES_FOR_RETRAIN}).")
                    return
                logging.info(f"Initiating training with {total} samples.")

            new_ttft_model = new_ttft_scaler = None
            new_tpot_model = new_tpot_scaler = None

            # Train TTFT
            if ttft_snap:
                df_ttft = pd.DataFrame(ttft_snap).dropna()
                df_ttft = df_ttft[df_ttft['actual_ttft_ms'] > 0]
                if len(df_ttft) >= settings.MIN_SAMPLES_FOR_RETRAIN:
                    X_ttft = df_ttft[['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running']]
                    y_ttft = df_ttft['actual_ttft_ms']
                    try:
                        new_ttft_model, new_ttft_scaler = self._train_model_with_scaling(X_ttft, y_ttft)
                        logging.info(f"TTFT model trained on {len(df_ttft)} samples.")
                    except Exception:
                        logging.error("Error training TTFT model", exc_info=True)
                else:
                    logging.warning("Not enough TTFT samples, skipping TTFT training.")

            # Train TPOT with new feature
            if tpot_snap:
                df_tpot = pd.DataFrame(tpot_snap).dropna()
                df_tpot = df_tpot[df_tpot['actual_tpot_ms'] > 0]
                if len(df_tpot) >= settings.MIN_SAMPLES_FOR_RETRAIN:
                    X_tpot = df_tpot[['kv_cache_percentage', 'num_request_waiting', 'num_request_running', 'num_tokens_generated']]
                    y_tpot = df_tpot['actual_tpot_ms']
                    try:
                        new_tpot_model, new_tpot_scaler = self._train_model_with_scaling(X_tpot, y_tpot)
                        logging.info(f"TPOT model trained on {len(df_tpot)} samples.")
                    except Exception:
                        logging.error("Error training TPOT model", exc_info=True)
                else:
                    logging.warning("Not enough TPOT samples, skipping TPOT training.")

            with self.lock:
                if new_ttft_model and new_ttft_scaler:
                    self.ttft_model, self.ttft_scaler = new_ttft_model, new_ttft_scaler
                if new_tpot_model and new_tpot_scaler:
                    self.tpot_model, self.tpot_scaler = new_tpot_model, new_tpot_scaler
                if self.is_ready:
                    self.last_retrain_time = datetime.now(timezone.utc)
                    try:
                        self._save_models_unlocked()
                    except Exception:
                        logging.error("Error saving models after training.", exc_info=True)
        except Exception as e:
            logging.error(f"Critical error in train(): {e}", exc_info=True)

    def predict(self, features: dict) -> Tuple[float, float, float, float]:
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

                ttft_arr = np.array([[
                    features['kv_cache_percentage'],
                    features['input_token_length'],
                    features['num_request_waiting'],
                    features['num_request_running']
                ]])
                tpot_arr = np.array([[
                    features['kv_cache_percentage'],
                    features['num_request_waiting'],
                    features['num_request_running'],
                    features['num_tokens_generated']
                ]])
                ttft_cols = ['kv_cache_percentage','input_token_length','num_request_waiting','num_request_running']
                tpot_cols = ['kv_cache_percentage','num_request_waiting','num_request_running','num_tokens_generated']
                if np.isnan(ttft_arr).any() or np.isinf(ttft_arr).any():
                    raise ValueError("TTFT features contain invalid values")
                if np.isnan(tpot_arr).any() or np.isinf(tpot_arr).any():
                    raise ValueError("TPOT features contain invalid values")
                
                # turn your feature dict into a single‐row DataFrame
                df_ttft = pd.DataFrame([{col: features[col] for col in ttft_cols}])
                df_tpot = pd.DataFrame([{col: features[col] for col in tpot_cols}])

                # now transform with the names intact
                ttft_scaled = self.ttft_scaler.transform(df_ttft)
                tpot_scaled = self.tpot_scaler.transform(df_tpot)

                ttft_pred, ttft_std = self.ttft_model.predict(ttft_scaled, return_std=True)
                tpot_pred, tpot_std = self.tpot_model.predict(tpot_scaled, return_std=True)
                return ttft_pred[0], tpot_pred[0], ttft_std[0], tpot_std[0]
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
                required = ['kv_cache_percentage', 'actual_ttft_ms', 'actual_tpot_ms', 'num_tokens_generated', 'input_token_length', 'num_request_waiting', 'num_request_running']
                for field in required:
                    if field not in sample or not isinstance(sample[field], (int, float)):
                        logging.warning(f"Invalid sample field: {field}")
                        return
                pct = max(0.0, min(1.0, sample['kv_cache_percentage']))
                idx = min(int(pct * self.num_buckets), self.num_buckets - 1)
                self.ttft_data_buckets[idx].append(sample)
                self.tpot_data_buckets[idx].append(sample)
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

    def _save_models_unlocked(self):
        try:
            if self.ttft_model and self.ttft_scaler:
                os.makedirs(os.path.dirname(settings.TTFT_MODEL_PATH), exist_ok=True)
                joblib.dump(self.ttft_model, settings.TTFT_MODEL_PATH)
                os.makedirs(os.path.dirname(settings.TTFT_SCALER_PATH), exist_ok=True)
                joblib.dump(self.ttft_scaler, settings.TTFT_SCALER_PATH)
                logging.info("TTFT model and scaler saved.")
            if self.tpot_model and self.tpot_scaler:
                os.makedirs(os.path.dirname(settings.TPOT_MODEL_PATH), exist_ok=True)
                joblib.dump(self.tpot_model, settings.TPOT_MODEL_PATH)
                os.makedirs(os.path.dirname(settings.TPOT_SCALER_PATH), exist_ok=True)
                joblib.dump(self.tpot_scaler, settings.TPOT_SCALER_PATH)
                logging.info("TPOT model and scaler saved.")
        except Exception as e:
            logging.error(f"Error saving models: {e}", exc_info=True)

    def load_models(self):
        try:
            with self.lock:
                if os.path.exists(settings.TTFT_MODEL_PATH) and os.path.exists(settings.TTFT_SCALER_PATH):
                    self.ttft_model = joblib.load(settings.TTFT_MODEL_PATH)
                    self.ttft_scaler = joblib.load(settings.TTFT_SCALER_PATH)
                else:
                    self.ttft_model, self.ttft_scaler = self._create_default_model("ttft")
                    self._save_models_unlocked()

                if os.path.exists(settings.TPOT_MODEL_PATH) and os.path.exists(settings.TPOT_SCALER_PATH):
                    self.tpot_model = joblib.load(settings.TPOT_MODEL_PATH)
                    self.tpot_scaler = joblib.load(settings.TPOT_SCALER_PATH)
                else:
                    self.tpot_model, self.tpot_scaler = self._create_default_model("tpot")
                    self._save_models_unlocked()

                if not self.is_ready:
                    raise RuntimeError("Failed to initialize models/scalers")
        except Exception as e:
            logging.error(f"Critical error in load_models: {e}", exc_info=True)
            raise
        
    def get_metrics(self) -> str:
        """Render Prometheus-style metrics: coefficients + bucket counts"""
        try:
            # Quick snapshot without lock to avoid blocking
            models_ready = self.is_ready
            ttft_model = self.ttft_model
            tpot_model = self.tpot_model
            ttft_scaler = self.ttft_scaler
            tpot_scaler = self.tpot_scaler
            
            # Snapshot bucket counts
            bucket_counts = {}
            for i in range(self.num_buckets):
                bucket_counts[f'ttft_{i}'] = len(self.ttft_data_buckets[i])
                bucket_counts[f'tpot_{i}'] = len(self.tpot_data_buckets[i])
            
            lines = []
            
            # Helper function to extract coefficients in original scale
            def add_coeffs(model, scaler, cols, prefix):
                try:
                    if model is None or scaler is None:
                        # Add placeholder metrics if models not available
                        lines.append(f"{prefix}_intercept {{}} 0.0")
                        for name in cols:
                            lines.append(f"{prefix}_coef{{feature=\"{name}\"}} 0.0")
                        return
                        
                    coef_scaled = model.coef_
                    scale = scaler.scale_
                    mean = scaler.mean_
                    w_orig = coef_scaled / scale
                    intercept_scaled = model.intercept_
                    intercept_orig = intercept_scaled - float(np.dot(coef_scaled, mean / scale))
                    
                    # Add intercept metric
                    lines.append(f"{prefix}_intercept {{}} {intercept_orig:.6f}")
                    
                    # Add coefficient metrics
                    for name, w in zip(cols, w_orig):
                        lines.append(f"{prefix}_coef{{feature=\"{name}\"}} {w:.6f}")
                except Exception as e:
                    logging.error(f"Error extracting coefficients for {prefix}: {e}")
                    # Add placeholder metrics if extraction fails
                    lines.append(f"{prefix}_intercept {{}} 0.0")
                    for name in cols:
                        lines.append(f"{prefix}_coef{{feature=\"{name}\"}} 0.0")
            
            # TTFT metrics
            ttft_cols = ['kv_cache_percentage','input_token_length','num_request_waiting','num_request_running']
            add_coeffs(ttft_model, ttft_scaler, ttft_cols, 'ttft')
            
            # TPOT metrics
            tpot_cols = ['kv_cache_percentage','num_request_waiting','num_request_running','num_tokens_generated']
            add_coeffs(tpot_model, tpot_scaler, tpot_cols, 'tpot')
            
            # Bucket counts from snapshot
            for i in range(self.num_buckets):
                lines.append(f"ttft_bucket_count{{bucket=\"{i}\"}} {bucket_counts[f'ttft_{i}']}")
                lines.append(f"tpot_bucket_count{{bucket=\"{i}\"}} {bucket_counts[f'tpot_{i}']}")
            
            return "\n".join(lines)
        except Exception as e:
            logging.error(f"Error generating metrics: {e}", exc_info=True)
            return "# Error generating metrics\n"

# --- FastAPI Application ---
app = FastAPI(
    title="Latency Predictor Service",
    description="A service to predict TTFT and TPOT with continuous training and feature scaling.",
)

predictor = LatencyPredictor()

# --- Pydantic Models for API ---
class TrainingEntry(BaseModel):
    kv_cache_percentage: float = Field(..., ge=0.0, le=1.0)
    input_token_length: int = Field(..., ge=0)
    num_request_waiting: int = Field(..., ge=0)
    num_request_running: int = Field(..., ge=0)
    actual_ttft_ms: float = Field(..., gt=0.0)
    actual_tpot_ms: float = Field(..., gt=0.0)
    num_tokens_generated: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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
        )
    except HTTPException:
        raise
    except Exception:
        logging.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Latency Predictor is running."}

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
    """Prometheus metrics including coefficients and bucket counts."""
    try:
        content = predictor.get_metrics()
        return Response(content, media_type="text/plain; version=0.0.4")
    except Exception as e:
        logging.error(f"Error in metrics endpoint: {e}", exc_info=True)
        return Response("# Error generating metrics\n", media_type="text/plain; version=0.0.4")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)





