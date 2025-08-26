import json
import os
import random
import time
import logging
import threading
from datetime import datetime, timezone
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

from fastapi.responses import Response  # Fixed import
from fastapi.responses import JSONResponse, FileResponse

import joblib
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

import tempfile
import shutil
import os  # Added this import

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Please install with: pip install xgboost")


class ModelType(str, Enum):
    BAYESIAN_RIDGE = "bayesian_ridge"
    XGBOOST = "xgboost"


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
    TTFT_MODEL_PATH: str = os.getenv("LATENCY_TTFT_MODEL_PATH", "/tmp/models/ttft.joblib")
    TPOT_MODEL_PATH: str = os.getenv("LATENCY_TPOT_MODEL_PATH", "/tmp/models/tpot.joblib")
    TTFT_SCALER_PATH: str = os.getenv("LATENCY_TTFT_SCALER_PATH", "/tmp/models/ttft_scaler.joblib")
    TPOT_SCALER_PATH: str = os.getenv("LATENCY_TPOT_SCALER_PATH", "/tmp/models/tpot_scaler.joblib")
    RETRAINING_INTERVAL_SEC: int = int(os.getenv("LATENCY_RETRAINING_INTERVAL_SEC", 1800))
    MIN_SAMPLES_FOR_RETRAIN_FRESH: int = int(os.getenv("LATENCY_MIN_SAMPLES_FOR_RETRAIN_FRESH", 10))
    MIN_SAMPLES_FOR_RETRAIN: int = int(os.getenv("LATENCY_MIN_SAMPLES_FOR_RETRAIN", 1000))
    MAX_TRAINING_DATA_SIZE_PER_BUCKET: int = int(os.getenv("LATENCY_MAX_TRAINING_DATA_SIZE_PER_BUCKET", 10000))
    TEST_TRAIN_RATIO: float = float(os.getenv("LATENCY_TEST_TRAIN_RATIO", "0.1"))  # Default 1:10 (10% test, 90% train)
    MAX_TEST_DATA_SIZE: int = int(os.getenv("LATENCY_MAX_TEST_DATA_SIZE", "1000"))  # Max test samples to keep
    MODEL_TYPE: str = os.getenv("LATENCY_MODEL_TYPE", "xgboost")  # Default to XGBoost

settings = Settings()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add this to your Pydantic models section
class ModelInfoResponse(BaseModel):
    model_type: str
    xgboost_available: bool
    is_ready: bool
    ttft_training_samples: int = Field(default=0, description="Number of TTFT training samples")
    tpot_training_samples: int = Field(default=0, description="Number of TPOT training samples") 
    ttft_test_samples: int = Field(default=0, description="Number of TTFT test samples")
    tpot_test_samples: int = Field(default=0, description="Number of TPOT test samples")
    last_retrain_time: Optional[datetime] = Field(default=None, description="Last retraining timestamp")
    min_samples_for_retrain: int = Field(default=0, description="Minimum samples required for retraining")
    retraining_interval_sec: int = Field(default=0, description="Retraining interval in seconds")

class LatencyPredictor:
    """
    Manages model training, prediction, and data handling.
    """
    def __init__(self, model_type: str = None):
        # Set model type with validation
        if model_type is None:
            model_type = settings.MODEL_TYPE
        
        if model_type not in [ModelType.BAYESIAN_RIDGE, ModelType.XGBOOST]:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(ModelType)}")
        
        if model_type == ModelType.XGBOOST and not XGBOOST_AVAILABLE:
            logging.warning("XGBoost requested but not available. Falling back to Bayesian Ridge.")
            model_type = ModelType.BAYESIAN_RIDGE
        
        self.model_type = ModelType(model_type)
        logging.info(f"Initialized LatencyPredictor with model type: {self.model_type}")
        
        self.num_buckets = int(1.0 / 0.05)
        self.bucket_size = settings.MAX_TRAINING_DATA_SIZE_PER_BUCKET 

        # Data buckets for sampling
        self.ttft_data_buckets = {i: deque(maxlen=self.bucket_size) for i in range(self.num_buckets)}
        self.tpot_data_buckets = {i: deque(maxlen=self.bucket_size) for i in range(self.num_buckets)}
        
        # Test data storage with configurable max size
        self.ttft_test_data = deque(maxlen=settings.MAX_TEST_DATA_SIZE)
        self.tpot_test_data = deque(maxlen=settings.MAX_TEST_DATA_SIZE)
        
        # R² score tracking (store last 5 scores)
        self.ttft_r2_scores = deque(maxlen=5)
        self.tpot_r2_scores = deque(maxlen=5)
        self.ttft_mape_scores = deque(maxlen=5)
        self.tpot_mape_scores = deque(maxlen=5)

        self.ttft_model = None
        self.tpot_model = None
        self.ttft_scaler = None
        self.tpot_scaler = None
        
        self.ttft_coefficients = None  # Will store descaled coefficients as dict
        self.tpot_coefficients = None  # Will store descaled coefficients as dict

        self.lock = threading.Lock()
        self.last_retrain_time = None
        self._shutdown_event = threading.Event()
        self._training_thread: threading.Thread = None
        
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
        else:  # XGBoost
            return all([self.ttft_model, self.tpot_model])

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

    def _train_model_with_scaling(self, features: pd.DataFrame, target: pd.Series) -> Union[Tuple[BayesianRidge, StandardScaler], xgb.XGBRegressor]:
        try:
            if len(features) == 0 or len(target) == 0:
                raise ValueError("Empty training data")
            if features.isnull().any().any() or target.isnull().any():
                raise ValueError("Training data contains NaN values")
            if np.isinf(features.values).any() or np.isinf(target.values).any():
                raise ValueError("Training data contains infinite values")

            if self.model_type == ModelType.BAYESIAN_RIDGE:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
                    raise ValueError("Scaling produced invalid values")

                model = BayesianRidge(compute_score=True)
                model.fit(features_scaled, target)
                return model, scaler
            
            else:  # XGBoost
                model = xgb.XGBRegressor(
                     n_estimators=200,            # Number of trees to build (moderate value for balanced accuracy and speed)
    max_depth=6,                 # Depth of trees; 6 is typically a sweet spot balancing bias/variance
    learning_rate=0.05,          # Smaller learning rate to achieve stable convergence
    subsample=0.8,               # Use 80% of data per tree (adds regularization & reduces overfitting)
    colsample_bytree=0.8,        # Use 80% of features per tree (improves generalization)
    min_child_weight=5,          # Helps control tree splits, reducing overfitting on small datasets
    gamma=0.1,                   # Adds conservative regularization; prevents overfitting
    objective="reg:quantileerror",    # quantile regression
    quantile_alpha=0.9,               # 90th percentile
    tree_method='hist',          # Efficient histogram algorithm; optimal for large datasets
    n_jobs=-1,                   # Utilize all CPU cores for parallel training
    random_state=42,             # Ensures reproducible results
    verbosity=1   
                )
                model.fit(features, target)
                return model
                
        except Exception as e:
            logging.error(f"Error in _train_model_with_scaling: {e}", exc_info=True)
            raise
        
    def _calculate_mape_on_test(self, model, scaler, test_data, feature_cols, target_col):
        """Calculate MAPE (%) on test data"""
        try:
            df = pd.DataFrame(test_data).dropna()
            print(f"df size: {len(df)} with sample data: {df.columns.tolist()}")
            df = df[df[target_col] > 0]
            
            if len(df) < 2:
                return None
            
            X = df[feature_cols]
            if self.model_type == ModelType.BAYESIAN_RIDGE:
                X = scaler.transform(X)
            
            y_true = df[target_col]
            y_pred = model.predict(X)
            return mean_absolute_percentage_error(y_true, y_pred) * 100
        except Exception as e:
            logging.error(f"Error calculating MAPE: {e}", exc_info=True)
        return None
    
    def _calculate_r2_on_test(self, model, scaler, test_data, feature_cols, target_col):
        """Calculate R² score on test data"""
        try:
            if len(test_data) == 0:
                return None
            
            df_test = pd.DataFrame(test_data).dropna()
            df_test = df_test[df_test[target_col] > 0]
            
            if len(df_test) < 2:  # Need at least 2 samples for R²
                return None
                
            X_test = df_test[feature_cols]
            y_test = df_test[target_col]
            
            if self.model_type == ModelType.BAYESIAN_RIDGE:
                X_test = scaler.transform(X_test)
            
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            return r2
        except Exception as e:
            logging.error(f"Error calculating R² score: {e}")
            return None

    def _create_default_model(self, model_type: str) -> Union[Tuple[BayesianRidge, StandardScaler], xgb.XGBRegressor]:
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
                target = pd.Series([10,])
            else:
                features = pd.DataFrame({
                    'kv_cache_percentage': [0.0],
                    'input_token_length': [1],  # Added input_token_length
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
                logging.info(f"Initiating training with {total} samples using {self.model_type}.")

            new_ttft_model = new_ttft_scaler = None
            new_tpot_model = new_tpot_scaler = None

            # Train TTFT
            if ttft_snap:
                df_ttft = pd.DataFrame(ttft_snap).dropna()
                df_ttft = df_ttft[df_ttft['actual_ttft_ms'] > 0]
                print(f"TTFT training data size: {len(df_ttft)} with sample data: {df_ttft.columns.tolist()}")
                if len(df_ttft) >= settings.MIN_SAMPLES_FOR_RETRAIN:
                    # Updated TTFT features to include prefix_cache_score
                    X_ttft = df_ttft[['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'prefix_cache_score']]
                    y_ttft = df_ttft['actual_ttft_ms']
                    try:
                        result = self._train_model_with_scaling(X_ttft, y_ttft)
                        if self.model_type == ModelType.BAYESIAN_RIDGE:
                            new_ttft_model, new_ttft_scaler = result
                        else:
                            new_ttft_model = result
                            new_ttft_scaler = None
                        
                        # Calculate R² on test data
                        ttft_feature_cols = ['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'prefix_cache_score']
                        r2_ttft = self._calculate_r2_on_test(new_ttft_model, new_ttft_scaler, 
                                                           list(self.ttft_test_data), ttft_feature_cols, 'actual_ttft_ms')
                        
                        if r2_ttft is not None:
                            self.ttft_r2_scores.append(r2_ttft)
                            logging.info(f"TTFT model trained on {len(df_ttft)} samples. Test R² = {r2_ttft:.4f}")
                        else:
                            logging.info(f"TTFT model trained on {len(df_ttft)} samples. Test R² = N/A (insufficient test data)")
                        
                        mape_ttft = self._calculate_mape_on_test(
                            new_ttft_model, new_ttft_scaler,
                            list(self.ttft_test_data),
                            ttft_feature_cols, 'actual_ttft_ms')
                        if mape_ttft is not None:
                            self.ttft_mape_scores.append(mape_ttft)
                            logging.info(f"TTFT Test MAPE = {mape_ttft:.2f}%")

                    except Exception:
                        logging.error("Error training TTFT model", exc_info=True)
                else:
                    logging.warning("Not enough TTFT samples, skipping TTFT training.")

            # Train TPOT
            if tpot_snap:
                df_tpot = pd.DataFrame(tpot_snap).dropna()
                df_tpot = df_tpot[df_tpot['actual_tpot_ms'] > 0]
                if len(df_tpot) >= settings.MIN_SAMPLES_FOR_RETRAIN:
                    # TPOT features remain unchanged
                    X_tpot = df_tpot[['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'num_tokens_generated']]
                    y_tpot = df_tpot['actual_tpot_ms']
                    try:
                        result = self._train_model_with_scaling(X_tpot, y_tpot)
                        if self.model_type == ModelType.BAYESIAN_RIDGE:
                            new_tpot_model, new_tpot_scaler = result
                        else:
                            new_tpot_model = result
                            new_tpot_scaler = None
                        
                        # Calculate R² on test data
                        tpot_feature_cols = ['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'num_tokens_generated']
                        r2_tpot = self._calculate_r2_on_test(new_tpot_model, new_tpot_scaler, 
                                                           list(self.tpot_test_data), tpot_feature_cols, 'actual_tpot_ms')
                        if r2_tpot is not None:
                            self.tpot_r2_scores.append(r2_tpot)
                            logging.info(f"TPOT model trained on {len(df_tpot)} samples. Test R² = {r2_tpot:.4f}")
                        else:
                            logging.info(f"TPOT model trained on {len(df_tpot)} samples. Test R² = N/A (insufficient test data)")
                        
                        mape_tpot = self._calculate_mape_on_test(
                            new_tpot_model, new_tpot_scaler,
                            list(self.tpot_test_data),
                            tpot_feature_cols, 'actual_tpot_ms')
                        if mape_tpot is not None:
                            self.tpot_mape_scores.append(mape_tpot)
                            logging.info(f"TPOT Test MAPE = {mape_tpot:.2f}%")
                            
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
                        ttft_features = ['kv_cache_percentage', 'input_token_length', 
                                       'num_request_waiting', 'num_request_running', 'prefix_cache_score']
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
                pct = max(0.0, min(1.0, sample['kv_cache_percentage']))
                idx = min(int(pct * self.num_buckets), self.num_buckets - 1)
            
                if ttft_valid:
                    self.ttft_data_buckets[idx].append(sample)
                if tpot_valid:
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
            if self.ttft_model:
                os.makedirs(os.path.dirname(settings.TTFT_MODEL_PATH), exist_ok=True)
                joblib.dump(self.ttft_model, settings.TTFT_MODEL_PATH)
                logging.info("TTFT model saved.")
            
                # Save XGBoost booster trees as JSON
                if self.model_type == ModelType.XGBOOST:
                    try:
                        booster = self.ttft_model.get_booster()
                        raw_trees = booster.get_dump(dump_format="json")
                        trees = [json.loads(t) for t in raw_trees]
                    
                        # Save to JSON file alongside the model
                        ttft_json_path = settings.TTFT_MODEL_PATH.replace('.joblib', '_trees.json')
                        with open(ttft_json_path, 'w') as f:
                            json.dump(trees, f, indent=2)
                        logging.info(f"TTFT XGBoost trees saved to {ttft_json_path}")
                    except Exception as e:
                        logging.error(f"Error saving TTFT XGBoost trees: {e}", exc_info=True)
            
            if self.ttft_scaler and self.model_type == ModelType.BAYESIAN_RIDGE:
                os.makedirs(os.path.dirname(settings.TTFT_SCALER_PATH), exist_ok=True)
                joblib.dump(self.ttft_scaler, settings.TTFT_SCALER_PATH)
                logging.info("TTFT scaler saved.")
            
            if self.tpot_model:
                os.makedirs(os.path.dirname(settings.TPOT_MODEL_PATH), exist_ok=True)
                joblib.dump(self.tpot_model, settings.TPOT_MODEL_PATH)
                logging.info("TPOT model saved.")
            
                # Save XGBoost booster trees as JSON
                if self.model_type == ModelType.XGBOOST:
                    try:
                        booster = self.tpot_model.get_booster()
                        raw_trees = booster.get_dump(dump_format="json")
                        trees = [json.loads(t) for t in raw_trees]
                    
                        # Save to JSON file alongside the model
                        tpot_json_path = settings.TPOT_MODEL_PATH.replace('.joblib', '_trees.json')
                        with open(tpot_json_path, 'w') as f:
                            json.dump(trees, f, indent=2)
                        logging.info(f"TPOT XGBoost trees saved to {tpot_json_path}")
                    except Exception as e:
                        logging.error(f"Error saving TPOT XGBoost trees: {e}", exc_info=True)
            
            if self.tpot_scaler and self.model_type == ModelType.BAYESIAN_RIDGE:
                os.makedirs(os.path.dirname(settings.TPOT_SCALER_PATH), exist_ok=True)
                joblib.dump(self.tpot_scaler, settings.TPOT_SCALER_PATH)
                logging.info("TPOT scaler saved.")
            
        except Exception as e:
            logging.error(f"Error saving models: {e}", exc_info=True)

    def load_models(self):
        try:
            with self.lock:
                if os.path.exists(settings.TTFT_MODEL_PATH):
                    self.ttft_model = joblib.load(settings.TTFT_MODEL_PATH)
                    if self.model_type == ModelType.BAYESIAN_RIDGE and os.path.exists(settings.TTFT_SCALER_PATH):
                        self.ttft_scaler = joblib.load(settings.TTFT_SCALER_PATH)
                else:
                    result = self._create_default_model("ttft")
                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        self.ttft_model, self.ttft_scaler = result
                    else:
                        self.ttft_model = result
                    settings.MIN_SAMPLES_FOR_RETRAIN = settings.MIN_SAMPLES_FOR_RETRAIN_FRESH
                    self._save_models_unlocked()

                if os.path.exists(settings.TPOT_MODEL_PATH):
                    self.tpot_model = joblib.load(settings.TPOT_MODEL_PATH)
                    if self.model_type == ModelType.BAYESIAN_RIDGE and os.path.exists(settings.TPOT_SCALER_PATH):
                        self.tpot_scaler = joblib.load(settings.TPOT_SCALER_PATH)
                else:
                    result = self._create_default_model("tpot")
                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        self.tpot_model, self.tpot_scaler = result
                    else:
                        self.tpot_model = result
                    settings.MIN_SAMPLES_FOR_RETRAIN = settings.MIN_SAMPLES_FOR_RETRAIN_FRESH
                    self._save_models_unlocked()

                if not self.is_ready:
                    raise RuntimeError("Failed to initialize models/scalers")
        except Exception as e:
            logging.error(f"Critical error in load_models: {e}", exc_info=True)
            raise
        
    def get_metrics(self) -> str:
        """Render Prometheus-style metrics: model, coefficients/importances, bucket counts, R² and MAPE scores."""
        try:
            # Snapshot models & scalers
            ttft_model, tpot_model = self.ttft_model, self.tpot_model
            ttft_scaler, tpot_scaler = self.ttft_scaler, self.tpot_scaler

            lines: List[str] = []
            # 1) Model type
            lines.append(f'model_type{{type="{self.model_type.value}"}} 1')

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
                    # XGBoost importances
                    try:
                        imps = model.feature_importances_
                    except Exception:
                        imps = [0.0]*len(feats)
                    lines.append(f'{prefix}_intercept{{}} 0.0')
                    for f, imp in zip(feats, imps):
                        lines.append(f'{prefix}_importance{{feature="{f}"}} {imp:.6f}')

            # Updated TTFT features to include prefix_cache_score
            ttft_feats = ["kv_cache_percentage","input_token_length","num_request_waiting","num_request_running","prefix_cache_score"]
            tpot_feats = ["kv_cache_percentage","input_token_length","num_request_waiting","num_request_running","num_tokens_generated"]
            emit_metrics(ttft_model, self.ttft_coefficients, ttft_feats, "ttft")
            emit_metrics(tpot_model, self.tpot_coefficients, tpot_feats, "tpot")

            # 3) Bucket counts
            for i in range(self.num_buckets):
                lines.append(f'training_samples_count{{model="ttft",bucket="{i}"}} {len(self.ttft_data_buckets[i])}')
                lines.append(f'training_samples_count{{model="tpot",bucket="{i}"}} {len(self.tpot_data_buckets[i])}')

            # 4) Last up to 5 R² scores
            for idx, score in enumerate(self.ttft_r2_scores):
                lines.append(f'ttft_r2_score{{idx="{idx}"}} {score:.6f}')
            for idx, score in enumerate(self.tpot_r2_scores):
                lines.append(f'tpot_r2_score{{idx="{idx}"}} {score:.6f}')

            # 5) Last up to 5 MAPE scores
            for idx, mape in enumerate(self.ttft_mape_scores):
                lines.append(f'ttft_mape{{idx="{idx}"}} {mape:.6f}')
            for idx, mape in enumerate(self.tpot_mape_scores):
                lines.append(f'tpot_mape{{idx="{idx}"}} {mape:.6f}')

            return "\n".join(lines) + "\n"

        except Exception as e:
            logging.error(f"Error generating metrics: {e}", exc_info=True)
            return "# error_generating_metrics 1\n"

                

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
    ttft_ms: float
    tpot_ms: float
    ttft_uncertainty: float
    tpot_uncertainty: float
    ttft_prediction_bounds: Tuple[float, float]
    tpot_prediction_bounds: Tuple[float, float]
    predicted_at: datetime
    model_type: ModelType = Field(default=predictor.model_type.value, description="Type of model used for prediction")
    
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
            model_type=predictor.model_type.value
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
    """Prometheus metrics including coefficients and bucket counts."""
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
        "model_type": predictor.model_type.value
    }
 
@app.get("/model/download/info")
async def model_download_info():
    """
    Get information about available model downloads and coefficients.
    """
    info = {
        "model_type": predictor.model_type.value,
        "available_endpoints": {}
    }
    
    if predictor.model_type == ModelType.BAYESIAN_RIDGE:
        info["available_endpoints"]["coefficients"] = "/metrics"
        info["coefficients_info"] = {
            "ttft_coefficients_available": predictor.ttft_coefficients is not None,
            "tpot_coefficients_available": predictor.tpot_coefficients is not None,
            "description": "Descaled coefficients available in Prometheus metrics endpoint"
        }
    else:  # XGBoost
        info["available_endpoints"]["trees"] = {
            "ttft_trees": "/model/ttft/xgb/json",
            "tpot_trees": "/model/tpot/xgb/json"
        }
    
    info["model_status"] = {
        "ttft_model_ready": predictor.ttft_model is not None,
        "tpot_model_ready": predictor.tpot_model is not None,
    }
    
    if predictor.model_type == ModelType.BAYESIAN_RIDGE:
        info["model_status"]["ttft_coefficients_ready"] = predictor.ttft_coefficients is not None
        info["model_status"]["tpot_coefficients_ready"] = predictor.tpot_coefficients is not None
    
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



@app.get("/model/{model_name}/info")
async def model_info(model_name: str):
    """Get model file information including last modified time."""
    model_paths = {
        "ttft": settings.TTFT_MODEL_PATH,
        "tpot": settings.TPOT_MODEL_PATH,
        "ttft_scaler": settings.TTFT_SCALER_PATH,
        "tpot_scaler": settings.TPOT_SCALER_PATH
    }
    
    if model_name not in model_paths:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
    
    model_path = model_paths[model_name]
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Get file stats
    stat = os.stat(model_path)
    last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    
    return {
        "model_name": model_name,
        "path": model_path,
        "size_bytes": stat.st_size,
        "last_modified": last_modified.isoformat(),
        "exists": True
    }


@app.get("/model/{model_name}/download")
async def download_model(model_name: str):
    """Download a model file."""
    model_paths = {
        "ttft": settings.TTFT_MODEL_PATH,
        "tpot": settings.TPOT_MODEL_PATH,
        "ttft_scaler": settings.TTFT_SCALER_PATH,
        "tpot_scaler": settings.TPOT_SCALER_PATH
    }
    
    if model_name not in model_paths:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
    
    model_path = model_paths[model_name]
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Return the file
    filename = f"{model_name}.joblib"
    return FileResponse(
        model_path,
        media_type='application/octet-stream',
        filename=filename
    )


@app.get("/models/list")
async def list_models():
    """List all available models with their status."""
    models = {}
    model_paths = {
        "ttft": settings.TTFT_MODEL_PATH,
        "tpot": settings.TPOT_MODEL_PATH,
        "ttft_scaler": settings.TTFT_SCALER_PATH,
        "tpot_scaler": settings.TPOT_SCALER_PATH
    }
    
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            stat = os.stat(model_path)
            models[model_name] = {
                "exists": True,
                "size_bytes": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
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
        "server_time": datetime.now(timezone.utc).isoformat()
    }


