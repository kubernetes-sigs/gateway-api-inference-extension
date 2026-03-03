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
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import requests

# Required dependencies for TreeLite-only inference
import tl2cgen
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import bundle constants for consistent file naming
from common.bundle_constants import (
    TPOT_CONFORMAL_FILENAME,
    TPOT_TREELITE_FILENAME,
    TTFT_CONFORMAL_FILENAME,
    TTFT_TREELITE_FILENAME,
)
from common.conformal_quantile import ConformalQuantilePredictor
from common.feature_encoder import FeatureEncoder
from common.feature_schema import FEATURE_SCHEMA_VERSION


@dataclass(frozen=True)
class ModelBundle:
    """
    Immutable container for TreeLite predictors and conformal calibration.

    This dataclass ensures atomic swapping of models - when we update models,
    we create a new bundle and swap the reference atomically. This allows
    lock-free reads during predictions while ensuring consistency.

    All artifacts in a bundle are version-coupled (trained and compiled together).
    TreeLite is the only inference path.
    """

    # TreeLite predictors (compiled .so models — required for inference)
    ttft_predictor: Any | None
    tpot_predictor: Any | None

    # Conformal predictors (version-coupled to TreeLite models)
    ttft_conformal: Any | None
    tpot_conformal: Any | None

    # Metadata
    model_type: str  # "xgboost" or "lightgbm" (source model type)
    quantile: float
    last_load: datetime | None

    def is_ready(self) -> bool:
        """Check if TreeLite predictors and conformal calibration are loaded."""
        return all([self.ttft_predictor, self.tpot_predictor, self.ttft_conformal, self.tpot_conformal])


class ModelType(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class PredictSettings:
    """Configuration for the prediction server."""

    # Training server URL
    TRAINING_SERVER_URL: str = os.getenv("TRAINING_SERVER_URL", "http://training-service:8000")

    # Bundle cache directory
    BUNDLE_CACHE_DIR: str = os.getenv("BUNDLE_CACHE_DIR", "/local_models/bundles")

    # Sync interval and model type
    MODEL_SYNC_INTERVAL_SEC: int = int(os.getenv("MODEL_SYNC_INTERVAL_SEC", "10"))
    MODEL_TYPE: ModelType = ModelType(os.getenv("LATENCY_MODEL_TYPE", "xgboost"))

    # Quantile configuration (should match training server)
    QUANTILE_ALPHA: float = float(os.getenv("LATENCY_QUANTILE_ALPHA", "0.90"))  # p90 quantile

    # Server host/port
    HOST: str = os.getenv("PREDICT_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PREDICT_PORT", "8001"))

    # HTTP timeout
    HTTP_TIMEOUT: int = int(os.getenv("HTTP_TIMEOUT", "30"))

    # Bootstrap behavior: if True (default), /readyz returns 200 even without models
    ALLOW_BOOTSTRAP_WITHOUT_MODEL: bool = os.getenv("ALLOW_BOOTSTRAP_WITHOUT_MODEL", "true").lower() == "true"


settings = PredictSettings()

# Configure logging level from environment variable (default: INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")


class ModelSyncer:
    """Downloads models from a training server via HTTP using bundle-aware syncing."""

    def __init__(self):
        self._shutdown_event = threading.Event()
        self._sync_thread: threading.Thread | None = None
        self._sync_lock = threading.Lock()

        # Track current bundle ID to avoid re-downloading same bundle
        self.current_bundle_id: str | None = None

        # Cache bundle metadata (training_samples, test_samples, etc.)
        self.current_bundle_info: dict | None = None

        # Track if we've logged the bootstrap waiting message (to avoid spam)
        self._logged_bootstrap_waiting = False

        # Training server lifecycle state (updated on each sync)
        self.training_lifecycle: dict | None = None

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
                # Non-200 status is an error (endpoint broken, network issue, etc.)
                logging.error(f"Failed to fetch bundle info from training server: HTTP {r.status_code}")
                raise Exception(f"Training server bundle endpoint failed with HTTP {r.status_code}")

            bundle_info = r.json()

            # Check status field to distinguish bootstrap from actual bundles
            status = bundle_info.get("status")

            if status == "no_bundles_available":
                # Extract lifecycle state from training server response
                new_lifecycle = bundle_info.get("lifecycle")
                if new_lifecycle:
                    old_state = self.training_lifecycle.get("state") if self.training_lifecycle else None
                    new_state = new_lifecycle.get("state")
                    if old_state != new_state:
                        logging.info(f"Training lifecycle: {old_state} -> {new_state}")
                    self.training_lifecycle = new_lifecycle

                # Bootstrap case: Endpoint exists but no bundles published yet (expected)
                if not self._logged_bootstrap_waiting:
                    # First time - log at info level with helpful message
                    message = bundle_info.get("message", "No bundles available yet")
                    min_samples = bundle_info.get("min_samples_required", 10)
                    logging.info(f"Bootstrap: {message}")
                    logging.info(
                        f"  → Server will become ready once training completes (requires ≥{min_samples} samples)"
                    )
                    self._logged_bootstrap_waiting = True
                else:
                    # Already logged, just debug now
                    logging.debug("Still waiting for first bundle...")
                return False

            if status != "bundle_available":
                # Unexpected status - log warning but don't crash
                logging.warning(f"Unexpected bundle status: {status}. Skipping sync.")
                return False

            # Normal case: bundle is available
            server_bundle_id = bundle_info["bundle_id"]

            if not server_bundle_id:
                # Sanity check: status says available but no bundle_id
                logging.warning("Bundle status is 'bundle_available' but bundle_id is null")
                return False

            # 2. Check if we already have this bundle
            if server_bundle_id == self.current_bundle_id:
                logging.debug(f"Bundle {server_bundle_id[:8]} already synced")
                return False

            logging.info(
                f"New bundle available: {server_bundle_id[:8]} (current: {self.current_bundle_id[:8] if self.current_bundle_id else 'none'})"
            )

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
                        # Write to a unique temp file, then atomic rename.
                        # Using a unique name avoids collisions when multiple
                        # prediction pods share the same volume.
                        fd, temp_dest = tempfile.mkstemp(dir=bundle_dir, suffix=f".{file_name}.tmp")
                        try:
                            with os.fdopen(fd, "wb") as f:
                                for chunk in file_r.iter_content(8192):
                                    if chunk:
                                        f.write(chunk)

                            # Atomic rename
                            os.replace(temp_dest, file_dest)
                            files_downloaded += 1
                            logging.debug(f"  ✓ Downloaded {file_name} ({os.path.getsize(file_dest)} bytes)")
                        except BaseException:
                            # Clean up temp file on any failure
                            try:
                                os.unlink(temp_dest)
                            except OSError:
                                pass
                            raise
                    else:
                        logging.warning(f"  ✗ Failed to download {file_name}: HTTP {file_r.status_code}")
                except Exception as e:
                    logging.error(f"  ✗ Error downloading {file_name}: {e}")

            # 5. Update current bundle ID, cache bundle info, and extract lifecycle
            with self._sync_lock:
                self.current_bundle_id = server_bundle_id
                self.current_bundle_info = bundle_info  # Cache full bundle info including training_samples
                new_lifecycle = bundle_info.get("lifecycle")
                if new_lifecycle:
                    old_state = self.training_lifecycle.get("state") if self.training_lifecycle else None
                    new_state = new_lifecycle.get("state")
                    if old_state != new_state:
                        logging.info(f"Training lifecycle: {old_state} -> {new_state}")
                    self.training_lifecycle = new_lifecycle

            # Reset bootstrap flag if this was the first successful sync
            if self._logged_bootstrap_waiting:
                logging.info(
                    f"🎉 First bundle synced! Bundle {server_bundle_id[:8]} available ({files_downloaded} files downloaded)"
                )
            else:
                logging.info(
                    f"✓ Bundle {server_bundle_id[:8]} synced ({files_downloaded} files downloaded, {len(bundle_info['files']) - files_downloaded} cached)"
                )
            return True

        except requests.RequestException as e:
            logging.debug(f"Bundle sync failed (network): {e}")
            return False
        except Exception as e:
            logging.error(f"Bundle sync failed: {e}", exc_info=True)
            return False

    def get_bundle_path(self, bundle_id: str | None = None) -> str | None:
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
        self.model_type = settings.MODEL_TYPE
        self.quantile = settings.QUANTILE_ALPHA
        self.prefix_buckets = 4

        # Initialize feature encoder for categorical encoding (default until bundle loads)
        self.feature_encoder = FeatureEncoder.from_default_schema()

        # Initialize with empty ModelBundle (will be populated by load_models)
        self.models = ModelBundle(
            ttft_predictor=None,
            tpot_predictor=None,
            ttft_conformal=None,
            tpot_conformal=None,
            model_type=self.model_type.value,
            quantile=self.quantile,
            last_load=None,
        )

        # Lock only for model swapping (not for predictions!)
        self.lock = threading.RLock()
        logging.info(f"Predictor type: {self.model_type}, quantile: {self.quantile}")

    @property
    def is_ready(self) -> bool:
        # No lock needed - atomic read of bundle reference
        return self.models.is_ready()

    def load_models(self) -> bool:
        """
        Load TreeLite predictors and conformal calibration from a bundle atomically.

        All artifacts are version-coupled: TreeLite .so files and conformal .json
        must all come from the same bundle. If any required artifact is missing,
        the load fails (no partial loads, no fallbacks).
        """
        try:
            bundle_path = model_syncer.get_bundle_path()
            if not bundle_path:
                logging.error("No bundle available - cannot load models")
                return False

            logging.debug(f"Loading models from bundle: {bundle_path}")

            ttft_treelite_path = os.path.join(bundle_path, TTFT_TREELITE_FILENAME)
            tpot_treelite_path = os.path.join(bundle_path, TPOT_TREELITE_FILENAME)
            ttft_conformal_path = os.path.join(bundle_path, TTFT_CONFORMAL_FILENAME)
            tpot_conformal_path = os.path.join(bundle_path, TPOT_CONFORMAL_FILENAME)

            # Fail fast if any required artifact is missing
            missing = []
            for path, name in [
                (ttft_treelite_path, "TTFT TreeLite .so"),
                (tpot_treelite_path, "TPOT TreeLite .so"),
                (ttft_conformal_path, "TTFT conformal .json"),
                (tpot_conformal_path, "TPOT conformal .json"),
            ]:
                if not os.path.exists(path):
                    missing.append(name)

            if missing:
                logging.error(f"Bundle incomplete, missing: {', '.join(missing)}")
                return False

            # Load all artifacts WITHOUT holding lock (I/O can be slow)
            new_ttft_predictor = tl2cgen.Predictor(ttft_treelite_path, nthread=1)
            logging.debug(f"TTFT TreeLite loaded: {ttft_treelite_path}")

            new_tpot_predictor = tl2cgen.Predictor(tpot_treelite_path, nthread=1)
            logging.debug(f"TPOT TreeLite loaded: {tpot_treelite_path}")

            with open(ttft_conformal_path) as f:
                ttft_conf_state = json.load(f)
            new_ttft_conformal = ConformalQuantilePredictor.from_state(ttft_conf_state)
            logging.debug(
                f"TTFT conformal loaded ({new_ttft_conformal.calibration_sample_count} samples, "
                f"calibrated_alpha={new_ttft_conformal._calibrated_alpha})"
            )

            with open(tpot_conformal_path) as f:
                tpot_conf_state = json.load(f)
            new_tpot_conformal = ConformalQuantilePredictor.from_state(tpot_conf_state)
            logging.debug(
                f"TPOT conformal loaded ({new_tpot_conformal.calibration_sample_count} samples, "
                f"calibrated_alpha={new_tpot_conformal._calibrated_alpha})"
            )

            # Build new bundle — all artifacts are version-coupled
            new_bundle = ModelBundle(
                ttft_predictor=new_ttft_predictor,
                tpot_predictor=new_tpot_predictor,
                ttft_conformal=new_ttft_conformal,
                tpot_conformal=new_tpot_conformal,
                model_type=self.model_type.value,
                quantile=self.quantile,
                last_load=datetime.now(UTC),
            )

            # Load feature encoder from bundle schema (if available)
            bundle_info = model_syncer.current_bundle_info
            remote_schema = bundle_info.get("feature_schema") if bundle_info else None
            remote_version = bundle_info.get("feature_schema_version") if bundle_info else None

            if remote_version is not None and remote_schema is not None:
                if remote_version != FEATURE_SCHEMA_VERSION:
                    logging.error(
                        f"Feature schema version mismatch: bundle={remote_version}, "
                        f"server={FEATURE_SCHEMA_VERSION}. Refusing to load."
                    )
                    return False
                new_encoder = FeatureEncoder.from_dict(remote_schema)
                logging.info(f"Feature encoder loaded from bundle schema v{remote_version}")
            else:
                logging.warning("Bundle has no feature_schema — using built-in default")
                new_encoder = FeatureEncoder.from_default_schema()

            # Atomic swap
            with self.lock:
                self.models = new_bundle
                self.feature_encoder = new_encoder

            bundle_id_short = model_syncer.current_bundle_id[:8] if model_syncer.current_bundle_id else "unknown"
            logging.info(f"Bundle {bundle_id_short} loaded: TreeLite + conformal (ready)")
            return True

        except Exception as e:
            logging.error(f"Bundle load failed: {e}", exc_info=True)
            return False

    def predict(self, features: dict) -> tuple[float, float]:
        """
        Make quantile predictions using TreeLite + conformal calibration.

        Lock-free: atomically reads the model bundle, uses that consistent
        snapshot for the entire prediction.
        """
        try:
            models = self.models

            if not models.is_ready():
                raise HTTPException(status_code=503, detail="Models not ready")

            # Validation
            required = [
                "kv_cache_percentage",
                "input_token_length",
                "num_request_waiting",
                "num_request_running",
                "num_tokens_generated",
                "prefix_cache_score",
            ]
            for f in required:
                if f not in features:
                    raise ValueError(f"Missing required feature: {f}")
                if not isinstance(features[f], (int, float)):
                    raise ValueError(f"Invalid type for feature {f}: expected number")

            # Extract and encode pod_type
            pod_type = features.get("pod_type", "")
            pod_type_code = self.feature_encoder.encode_value("pod_type", pod_type)

            # TTFT features
            effective_input_tokens = (1 - features["prefix_cache_score"]) * features["input_token_length"]
            prefill_score_bucket = self.feature_encoder.encode_value(
                "prefix_cache_score", features["prefix_cache_score"]
            )

            ttft_features = np.array(
                [
                    [
                        features["kv_cache_percentage"],
                        features["input_token_length"],
                        features["num_request_waiting"],
                        features["num_request_running"],
                        features["prefix_cache_score"],
                        effective_input_tokens,
                        prefill_score_bucket,
                        pod_type_code,
                    ]
                ],
                dtype=np.float32,
            )

            # TPOT features
            tpot_features = np.array(
                [
                    [
                        features["kv_cache_percentage"],
                        features["input_token_length"],
                        features["num_request_waiting"],
                        features["num_request_running"],
                        features["num_tokens_generated"],
                        pod_type_code,
                    ]
                ],
                dtype=np.float32,
            )

            # TreeLite prediction
            ttft_mean = float(np.ravel(models.ttft_predictor.predict(tl2cgen.DMatrix(ttft_features)))[0])
            tpot_mean = float(np.ravel(models.tpot_predictor.predict(tl2cgen.DMatrix(tpot_features)))[0])

            # Conformal correction (always applied — is_ready() guarantees both are loaded)
            ttft_pred = models.ttft_conformal.conformalize(ttft_mean)
            tpot_pred = models.tpot_conformal.conformalize(tpot_mean)

            return ttft_pred, tpot_pred

        except ValueError as ve:
            logging.warning(f"Client error in predict(): {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except HTTPException:
            raise
        except Exception:
            logging.error("Error in predict():", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during prediction")

    def predict_batch(self, features_list: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """
        Make batch quantile predictions using TreeLite + conformal calibration.

        Lock-free: atomically reads the model bundle, uses that consistent
        snapshot for the entire batch.
        """
        try:
            models = self.models

            if not models.is_ready():
                raise HTTPException(status_code=503, detail="Models not ready")

            # Validation
            required = [
                "kv_cache_percentage",
                "input_token_length",
                "num_request_waiting",
                "num_request_running",
                "num_tokens_generated",
                "prefix_cache_score",
            ]
            for i, features in enumerate(features_list):
                for f in required:
                    if f not in features:
                        raise ValueError(f"Missing required feature '{f}' in request {i}")
                    if not isinstance(features[f], (int, float)):
                        raise ValueError(f"Invalid type for feature '{f}' in request {i}: expected number")

            # Extract pod_type codes for all requests
            pod_type_values = [features.get("pod_type", "") for features in features_list]
            pod_type_codes = self.feature_encoder.encode_batch("pod_type", np.array(pod_type_values)).astype(np.float32)

            # TTFT features: vectorized construction
            kv = np.array([f["kv_cache_percentage"] for f in features_list], dtype=np.float32)
            inp_len = np.array([f["input_token_length"] for f in features_list], dtype=np.float32)
            waiting = np.array([f["num_request_waiting"] for f in features_list], dtype=np.float32)
            running = np.array([f["num_request_running"] for f in features_list], dtype=np.float32)
            prefix = np.array([f["prefix_cache_score"] for f in features_list], dtype=np.float32)
            tokens = np.array([f["num_tokens_generated"] for f in features_list], dtype=np.float32)

            effective_input_tokens = (1.0 - prefix) * inp_len
            prefill_score_bucket = self.feature_encoder.encode_batch("prefix_cache_score", prefix)

            ttft_features = np.column_stack(
                [
                    kv,
                    inp_len,
                    waiting,
                    running,
                    prefix,
                    effective_input_tokens,
                    prefill_score_bucket,
                    pod_type_codes,
                ]
            )

            tpot_features = np.column_stack(
                [
                    kv,
                    inp_len,
                    waiting,
                    running,
                    tokens,
                    pod_type_codes,
                ]
            )

            # TreeLite batch prediction
            ttft_mean_preds = np.ravel(models.ttft_predictor.predict(tl2cgen.DMatrix(ttft_features)))
            tpot_mean_preds = np.ravel(models.tpot_predictor.predict(tl2cgen.DMatrix(tpot_features)))

            # Conformal correction (always applied — is_ready() guarantees both are loaded)
            ttft_pred = models.ttft_conformal.conformalize_batch(ttft_mean_preds)
            tpot_pred = models.tpot_conformal.conformalize_batch(tpot_mean_preds)

            return ttft_pred, tpot_pred

        except ValueError as ve:
            logging.warning(f"Client error in predict_batch(): {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except HTTPException:
            raise
        except Exception:
            logging.error("Error in predict_batch():", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during batch prediction")


# Instantiate
model_syncer = ModelSyncer()
predictor = LightweightPredictor()

# FastAPI app
app = FastAPI(
    title="HTTP-based Quantile Latency Predictor",
    description="A prediction service that downloads quantile regression models from training server via HTTP.",
    version="1.0.0",
)


# Pydantic models
class PredictionRequest(BaseModel):
    kv_cache_percentage: float = Field(..., ge=0.0, le=1.0)
    input_token_length: int = Field(..., ge=0)
    num_request_waiting: int = Field(..., ge=0)
    num_request_running: int = Field(..., ge=0)
    num_tokens_generated: int = Field(..., ge=0)
    prefix_cache_score: float = Field(..., ge=0.0, le=1.0, description="Prefix cache hit ratio score (0.0 to 1.0)")
    pod_type: str | None = Field(default="", description="Pod type: 'prefill', 'decode', or '' for monolithic")


class PredictionResponse(BaseModel):
    ttft_ms: float = Field(..., description=f"Predicted {settings.QUANTILE_ALPHA:.0%} quantile TTFT in milliseconds")
    tpot_ms: float = Field(..., description=f"Predicted {settings.QUANTILE_ALPHA:.0%} quantile TPOT in milliseconds")
    predicted_at: datetime
    model_type: str = Field(..., description="Type of model used for prediction")
    quantile: float = Field(..., description="Quantile being predicted")
    last_model_load: datetime | None


class StatusResponse(BaseModel):
    is_ready: bool
    model_type: str
    quantile: float = Field(..., description="Quantile being predicted")
    last_model_load: datetime | None
    training_server_url: str
    models_exist: dict
    bundle_info: dict | None = Field(
        None, description="Current bundle metadata (training_samples, test_samples, bundle_id, created_at)"
    )
    training_lifecycle: dict | None = Field(
        None,
        description="Training server lifecycle state (ABSENT, WAITING_FOR_SAMPLES, TRAINING, COMPILING, READY, ERROR)",
    )


class BulkPredictionRequest(BaseModel):
    requests: list[PredictionRequest] = Field(
        ..., min_items=1, max_items=10000, description="List of prediction requests (max 10000)"
    )


class BulkPredictionResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(..., description="List of prediction responses")
    total_requests: int = Field(..., description="Total number of requests processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class BulkPredictionError(BaseModel):
    index: int = Field(..., description="Index of the failed request in the original batch")
    error: str = Field(..., description="Error message")
    request: PredictionRequest = Field(..., description="The original request that failed")


class BulkPredictionResponseWithErrors(BaseModel):
    predictions: list[PredictionResponse | None] = Field(
        ..., description="List of prediction responses (None for failed predictions)"
    )
    errors: list[BulkPredictionError] = Field(..., description="List of errors for failed predictions")
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

    # Check if TreeLite artifacts exist in current bundle
    bundle_path = model_syncer.get_bundle_path()
    models_exist = {}

    if bundle_path:
        models_exist = {
            "ttft_treelite": os.path.exists(os.path.join(bundle_path, TTFT_TREELITE_FILENAME)),
            "tpot_treelite": os.path.exists(os.path.join(bundle_path, TPOT_TREELITE_FILENAME)),
            "ttft_conformal": os.path.exists(os.path.join(bundle_path, TTFT_CONFORMAL_FILENAME)),
            "tpot_conformal": os.path.exists(os.path.join(bundle_path, TPOT_CONFORMAL_FILENAME)),
        }
    else:
        models_exist = {"error": "No bundle loaded"}

    # Build bundle_info from cached bundle metadata
    bundle_info = None
    if model_syncer.current_bundle_info:
        bundle_info = {
            "bundle_id": model_syncer.current_bundle_info.get("bundle_id", "unknown")[:8],
            "training_samples": model_syncer.current_bundle_info.get("training_samples", {}),
            "test_samples": model_syncer.current_bundle_info.get("test_samples", {}),
            "created_at": model_syncer.current_bundle_info.get("created_at", "unknown"),
        }

    return StatusResponse(
        is_ready=predictor.is_ready,
        model_type=predictor.model_type.value,
        quantile=predictor.quantile,
        last_model_load=models.last_load,
        training_server_url=settings.TRAINING_SERVER_URL,
        models_exist=models_exist,
        bundle_info=bundle_info,
        training_lifecycle=model_syncer.training_lifecycle,
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
            predicted_at=datetime.now(UTC),
            model_type=predictor.model_type.value,
            quantile=predictor.quantile,
            last_model_load=models.last_load,
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
        current_time = datetime.now(UTC)

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
                last_model_load=models.last_load,
            )

            predictions.append(prediction_response)

        processing_time_ms = (time.time() - start_time) * 1000

        return BulkPredictionResponse(
            predictions=predictions,
            total_requests=len(request.requests),
            successful_predictions=len(predictions),
            failed_predictions=0,
            processing_time_ms=processing_time_ms,
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
    required = [
        "kv_cache_percentage",
        "input_token_length",
        "num_request_waiting",
        "num_request_running",
        "num_tokens_generated",
        "prefix_cache_score",
    ]

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
            errors.append(BulkPredictionError(index=i, error=str(e), request=pred_request))

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
            current_time = datetime.now(UTC)

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
                    last_model_load=models.last_load,
                )

                predictions[original_idx] = prediction_response

        except Exception as e:
            # If batch prediction fails, mark all valid requests as failed
            for original_idx in valid_indices:
                errors.append(
                    BulkPredictionError(
                        index=original_idx,
                        error=f"Batch prediction error: {e!s}",
                        request=request.requests[original_idx],
                    )
                )
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
        processing_time_ms=processing_time_ms,
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
            "last_load_time": predictor.models.last_load,
        }
    except Exception as e:
        logging.error(f"Error reloading models: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading models: {e!s}")


@app.get("/calibration/stats")
async def get_calibration_stats():
    """Get conformal calibration statistics."""

    # Atomic read of model bundle
    models = predictor.models

    stats = {
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile,
        "ttft_conformal": None,
        "tpot_conformal": None,
    }

    if models.ttft_conformal:
        # Ensure cache is updated before reading
        if models.ttft_conformal._cache_dirty:
            models.ttft_conformal._update_quantile_cache()

        stats["ttft_conformal"] = {
            "calibration_samples": models.ttft_conformal.calibration_sample_count,
            "quantile_adjustment_ms": float(models.ttft_conformal._cached_quantile_value)
            if models.ttft_conformal._cached_quantile_value is not None
            else None,
            "target_quantile": models.ttft_conformal.quantile,
            "max_calibration_samples": models.ttft_conformal.max_calibration_samples,
            "calibrated_alpha": models.ttft_conformal._calibrated_alpha,
        }
    else:
        stats["ttft_conformal"] = {"error": "TTFT conformal calibration not loaded"}

    if models.tpot_conformal:
        # Ensure cache is updated before reading
        if models.tpot_conformal._cache_dirty:
            models.tpot_conformal._update_quantile_cache()

        stats["tpot_conformal"] = {
            "calibration_samples": models.tpot_conformal.calibration_sample_count,
            "quantile_adjustment_ms": float(models.tpot_conformal._cached_quantile_value)
            if models.tpot_conformal._cached_quantile_value is not None
            else None,
            "target_quantile": models.tpot_conformal.quantile,
            "max_calibration_samples": models.tpot_conformal.max_calibration_samples,
            "calibrated_alpha": models.tpot_conformal._calibrated_alpha,
        }
    else:
        stats["tpot_conformal"] = {"error": "TPOT conformal calibration not loaded"}

    return stats


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "http-based-quantile-latency-predictor"}


@app.get("/readyz")
async def readiness_check():
    """
    Readiness check endpoint.

    If ALLOW_BOOTSTRAP_WITHOUT_MODEL=true (default): always returns 200.
    If ALLOW_BOOTSTRAP_WITHOUT_MODEL=false: returns 503 until models are loaded.

    Individual /predict requests will return 503 until models are loaded regardless.
    """
    ready = predictor.is_ready

    if not settings.ALLOW_BOOTSTRAP_WITHOUT_MODEL and not ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "models_loaded": False,
                "model_type": predictor.model_type.value,
                "training_lifecycle": model_syncer.training_lifecycle,
            },
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ready",
            "models_loaded": ready,
            "model_type": predictor.model_type.value,
            "quantile": predictor.quantile,
            "bundle_id": model_syncer.current_bundle_id[:8] if model_syncer.current_bundle_id else None,
            "bootstrap": not ready,
            "training_lifecycle": model_syncer.training_lifecycle,
        },
    )


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
        "training_server": settings.TRAINING_SERVER_URL,
    }


@app.on_event("startup")
async def startup():
    logging.info("Starting prediction server...")
    logging.info(f"  Model type: {predictor.model_type.value}")
    logging.info(f"  Quantile: {predictor.quantile}")
    logging.info(f"  Training server: {settings.TRAINING_SERVER_URL}")
    logging.info(f"  Sync interval: {settings.MODEL_SYNC_INTERVAL_SEC}s")

    # Initial bundle sync & load (may be no bundles yet during bootstrap)
    try:
        bundle_synced = model_syncer.sync_bundle()
        if bundle_synced:
            predictor.load_models()
            logging.info("✓ Initial bundle loaded successfully")
        else:
            # Could be network error or bootstrap - check which
            if model_syncer._logged_bootstrap_waiting:
                # Bootstrap case - we got a response but no bundles
                logging.info("⚠ No bundles available yet - waiting for training server to publish first bundle")
                logging.info("  (Server will become ready once training completes and bundle is published)")
            else:
                # Network error or other failure - training server not reachable yet
                logging.info("⚠ Training server not reachable yet - will retry in background")
                logging.info("  (This is normal during startup if training server is still starting)")
    except Exception as e:
        logging.warning(f"Initial bundle sync failed: {e}")
        logging.info("  Background sync will continue retrying...")

    # Start background sync loop
    model_syncer.start()
    logging.info("✓ Prediction server startup complete")


@app.on_event("shutdown")
async def shutdown():
    logging.info("Shutting down...")
    model_syncer.shutdown()


if __name__ == "__main__":
    uvicorn.run("__main__:app", host=settings.HOST, port=settings.PORT, reload=True)
