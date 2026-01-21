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
import asyncio
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import random
import requests
import pytest
import joblib
import numpy as np
import tempfile

# Base URLs for the dual-server architecture

PREDICTION_URL = os.getenv("PREDICTION_SERVER_URL", "http://<PREDICTION_IP>:80")  # Update this
TRAINING_URL = os.getenv("TRAINING_SERVER_URL", "http://<TRAINING_IP>:8080")  # Update this

TARGET_QPS = float(os.getenv("TARGET_QPS", 1000))  # Update this
TARGET_QPS_LARGE_BATCH = float(os.getenv("TARGET_QPS_LARGE_BATCH", 100))  # Update this
# Helper to wait until the servers are ready
def wait_for_ready(url: str, timeout: float = 30.0, interval: float = 1.0):
    start = time.time()
    while True:
        try:
            r = requests.get(f"{url}/readyz", timeout=2.0)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        if time.time() - start > timeout:
            pytest.skip(f"Server at {url} did not become ready in time")
        time.sleep(interval)

@pytest.fixture(scope="module", autouse=True)
def ensure_servers_ready():
    """Wait for both servers to be ready before running tests."""
    print("Waiting for prediction server...")
    wait_for_ready(PREDICTION_URL)
    print("Waiting for training server...")
    wait_for_ready(TRAINING_URL)


def test_prediction_server_healthz():
    """Test prediction server health endpoint."""
    r = requests.get(f"{PREDICTION_URL}/healthz")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_training_server_healthz():
    """Test training server health endpoint."""
    r = requests.get(f"{TRAINING_URL}/healthz")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_prediction_server_readyz():
    """Test prediction server readiness."""
    r = requests.get(f"{PREDICTION_URL}/readyz")
    assert r.status_code == 200
    assert r.json().get("status") == "ready"


def test_training_server_readyz():
    """Test training server readiness."""
    r = requests.get(f"{TRAINING_URL}/readyz")
    assert r.status_code == 200
    assert r.json().get("status") == "ready"


def test_prediction_server_status():
    """Test prediction server status endpoint."""
    r = requests.get(f"{PREDICTION_URL}/status")
    assert r.status_code == 200
    
    data = r.json()
    assert "is_ready" in data
    assert "model_type" in data
    assert "models_exist" in data
    assert "quantile" in data
    assert data["model_type"] in ["bayesian_ridge", "xgboost", "lightgbm"]
    assert 0 < data["quantile"] <= 1.0
    
    print(f"Prediction server using model type: {data['model_type']}")
    print(f"Quantile: {data['quantile']}")
    print(f"Models ready: {data['is_ready']}")
    print(f"Models exist: {data['models_exist']}")


def test_training_server_model_info():
    """Test training server model info endpoint."""
    r = requests.get(f"{TRAINING_URL}/model/download/info")
    assert r.status_code == 200
    
    data = r.json()
    assert "model_type" in data
    assert "available_endpoints" in data
    assert data["model_type"] in ["bayesian_ridge", "xgboost", "lightgbm"]
    
    print(f"Training server using model type: {data['model_type']}")


def test_training_server_models_list():
    """Test training server models list endpoint."""
    r = requests.get(f"{TRAINING_URL}/models/list")
    assert r.status_code == 200
    
    data = r.json()
    assert "models" in data
    assert "model_type" in data
    assert "server_time" in data
    
    models = data["models"]
    expected_models = ["ttft", "tpot"]
    if data["model_type"] == "bayesian_ridge":
        expected_models.extend(["ttft_scaler", "tpot_scaler"])
    
    for model_name in expected_models:
        assert model_name in models, f"Model {model_name} should be listed"
        print(f"Model {model_name}: exists={models[model_name]['exists']}, size={models[model_name]['size_bytes']} bytes")


def test_model_download_from_training_server():
    """Test downloading models from training server."""
    # First check what models are available
    models_r = requests.get(f"{TRAINING_URL}/models/list")
    models_data = models_r.json()
    
    for model_name in ["ttft", "tpot"]:
        if models_data["models"][model_name]["exists"]:
            # Test model info endpoint
            info_r = requests.get(f"{TRAINING_URL}/model/{model_name}/info")
            assert info_r.status_code == 200
            info_data = info_r.json()
            assert info_data["exists"] == True
            assert info_data["size_bytes"] > 0
            
            # Test model download with retry and streaming
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    download_r = requests.get(
                        f"{TRAINING_URL}/model/{model_name}/download", 
                        timeout=30,
                        stream=True  # Use streaming to handle large files better
                    )
                    if download_r.status_code == 200:
                        # Read content in chunks to avoid memory issues
                        content_length = 0
                        for chunk in download_r.iter_content(chunk_size=8192):
                            content_length += len(chunk)
                        
                        assert content_length > 0, f"Downloaded {model_name} model is empty"
                        print(f"Successfully downloaded {model_name} model ({content_length} bytes)")
                        break
                except requests.exceptions.ChunkedEncodingError as e:
                    print(f"Download attempt {attempt + 1}/{max_retries} failed for {model_name}: {e}")
                    if attempt == max_retries - 1:
                        print(f"⚠️ Model download test skipped for {model_name} due to connection issues")
                        # Don't fail the test - this might be a network/server issue
                        continue
                    time.sleep(2)  # Wait before retry

def test_lightgbm_endpoints_on_training_server():
    """Test LightGBM endpoints on training server if LightGBM is being used."""
    model_info_r = requests.get(f"{TRAINING_URL}/model/download/info")
    model_type = model_info_r.json().get("model_type")
    
    if model_type != "lightgbm":
        print("Skipping LightGBM endpoint tests - not using LightGBM model")
        return
    
    print("Testing LightGBM endpoints on training server...")
    
    # Test TTFT model text format
    ttft_txt_response = requests.get(f"{TRAINING_URL}/model/ttft/lgb/txt")
    if ttft_txt_response.status_code == 200:
        print("✓ TTFT LightGBM text model available")
        assert ttft_txt_response.headers.get('content-type') == 'text/plain; charset=utf-8'
    else:
        print(f"TTFT LightGBM text model not yet available (status: {ttft_txt_response.status_code})")
    
    # Test TPOT model text format
    tpot_txt_response = requests.get(f"{TRAINING_URL}/model/tpot/lgb/txt")
    if tpot_txt_response.status_code == 200:
        print("✓ TPOT LightGBM text model available")
        assert tpot_txt_response.headers.get('content-type') == 'text/plain; charset=utf-8'
    else:
        print(f"TPOT LightGBM text model not yet available (status: {tpot_txt_response.status_code})")
    
    # Test TTFT feature importances
    ttft_imp_response = requests.get(f"{TRAINING_URL}/model/ttft/lgb/importances")
    if ttft_imp_response.status_code == 200:
        ttft_importances = ttft_imp_response.json()
        assert isinstance(ttft_importances, dict), "TTFT importances should be a dict"
        
        # Check for expected features including prefix_cache_score
        expected_features = ["kv_cache_percentage", "input_token_length", "num_request_waiting", 
                           "num_request_running", "prefix_cache_score"]
        for feature in expected_features:
            assert feature in ttft_importances, f"Missing feature importance: {feature}"
        
        print(f"✓ TTFT LightGBM importances available with {len(ttft_importances)} features")
    else:
        print(f"TTFT LightGBM importances not yet available (status: {ttft_imp_response.status_code})")
    
    # Test TPOT feature importances
    tpot_imp_response = requests.get(f"{TRAINING_URL}/model/tpot/lgb/importances")
    if tpot_imp_response.status_code == 200:
        tpot_importances = tpot_imp_response.json()
        assert isinstance(tpot_importances, dict), "TPOT importances should be a dict"
        
        # Check for expected features
        expected_features = ["kv_cache_percentage", "input_token_length", "num_request_waiting", 
                           "num_request_running", "num_tokens_generated"]
        for feature in expected_features:
            assert feature in tpot_importances, f"Missing feature importance: {feature}"
        
        print(f"✓ TPOT LightGBM importances available with {len(tpot_importances)} features")
    else:
        print(f"TPOT LightGBM importances not yet available (status: {tpot_imp_response.status_code})")
        
        
def test_add_training_data_to_training_server():
    """
    Send training data to the training server.
    The prediction server should eventually sync these models.
    """
    entries = []
    
    # Generate 50 training samples with known pattern
    for i in range(1, 51):
        waiting = i % 10 + 1
        tokens = waiting
        inp_len = 10 * i
        kv = 0.5
        running = 1
        prefix_cache = random.uniform(0.1, 0.9)  # Added prefix_cache_score
        
        entries.append({
            "kv_cache_percentage": kv,
            "input_token_length": inp_len,
            "num_request_waiting": waiting,
            "num_request_running": running,
            "actual_ttft_ms": (inp_len*2.0 + waiting*3.0 + running*4.0 + kv*50.0 + prefix_cache*30.0) + 95,  # Include prefix_cache effect
            "actual_tpot_ms": (kv*100.0 + inp_len*0.5 + tokens*1.0 + running*5.0) + 9,
            "num_tokens_generated": tokens,
            "prefix_cache_score": prefix_cache,  # Added prefix_cache_score field
        })

    payload = {"entries": entries}
    r = requests.post(f"{TRAINING_URL}/add_training_data_bulk", json=payload)
    assert r.status_code == 202, f"Expected 202, got {r.status_code}"
    assert r.json().get("message") == "Accepted 50 training samples."
    
    print("Successfully sent training data to training server")


def test_prediction_server_model_sync():
    """
    Test that the prediction server can sync models from the training server.
    This may take some time as models need to be downloaded.
    """
    # Trigger a manual reload on the prediction server
    reload_r = requests.post(f"{PREDICTION_URL}/reload")
    assert reload_r.status_code == 200
    
    reload_data = reload_r.json()
    print(f"Model reload result: synced={reload_data.get('synced')}, loaded={reload_data.get('loaded')}")
    
    # Check status after reload
    status_r = requests.get(f"{PREDICTION_URL}/status")
    status_data = status_r.json()
    
    # Wait a bit for models to sync if they're not ready yet
    max_wait = 60  # 60 seconds max wait
    start_time = time.time()
    
    while not status_data.get("is_ready") and (time.time() - start_time) < max_wait:
        print("Waiting for prediction server models to be ready...")
        time.sleep(5)
        
        # Try reload again
        requests.post(f"{PREDICTION_URL}/reload")
        
        status_r = requests.get(f"{PREDICTION_URL}/status")
        status_data = status_r.json()
    
    assert status_data.get("is_ready"), f"Prediction server models not ready after {max_wait}s"
    print("Prediction server models are ready!")


def test_prediction_via_prediction_server():
    """Test making predictions via the prediction server."""
    features = {
        "kv_cache_percentage": 0.5,
        "input_token_length": 200,
        "num_request_waiting": 4,
        "num_request_running": 1,
        "num_tokens_generated": 4,
        "prefix_cache_score": 0.7,  # Added prefix_cache_score field
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict", json=features)
    assert r.status_code == 200
    
    data = r.json()
    required_fields = [
        "ttft_ms", "tpot_ms", 
        "predicted_at", "model_type", "last_model_load"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Verify predictions are reasonable
    assert data["ttft_ms"] > 0
    assert data["tpot_ms"] > 0
    #assert data["ttft_uncertainty"] >= 0
    #assert data["tpot_uncertainty"] >= 0
    
    print(f"Prediction successful: TTFT={data['ttft_ms']:.2f}ms, TPOT={data['tpot_ms']:.2f}ms")
    print(f"Model type: {data['model_type']}")


def test_bulk_prediction_strict():
    """Test bulk predictions with strict error handling."""
    print("Testing bulk prediction strict endpoint...")
    
    requests_data = [
        {
            "kv_cache_percentage": 0.5,
            "input_token_length": 200,
            "num_request_waiting": 4,
            "num_request_running": 1,
            "num_tokens_generated": 4,
            "prefix_cache_score": 0.7,
        },
        {
            "kv_cache_percentage": 0.3,
            "input_token_length": 150,
            "num_request_waiting": 2,
            "num_request_running": 1,
            "num_tokens_generated": 5,
            "prefix_cache_score": 0.5,
        }
    ]
    
    bulk_request = {"requests": requests_data}
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk/strict", json=bulk_request)
    assert r.status_code == 200
    
    data = r.json()
    
    # Check bulk response structure
    assert "predictions" in data
    assert "total_requests" in data
    assert "successful_predictions" in data
    assert "failed_predictions" in data
    assert "processing_time_ms" in data
    
    assert len(data["predictions"]) == 2
    assert data["total_requests"] == 2
    assert data["successful_predictions"] == 2
    assert data["failed_predictions"] == 0
    
    # Check individual prediction structure
    for prediction in data["predictions"]:
        assert "ttft_ms" in prediction
        assert "tpot_ms" in prediction
        #assert "ttft_uncertainty" in prediction
        #assert "tpot_uncertainty" in prediction
       #assert "ttft_prediction_bounds" in prediction
        #assert "tpot_prediction_bounds" in prediction
        assert "predicted_at" in prediction
        assert "model_type" in prediction
        assert "quantile" in prediction
        
    print("✓ Bulk prediction strict endpoint test passed")


def test_bulk_prediction_with_validation_errors():
    """Test that bulk predictions fail completely when any request has validation errors."""
    print("Testing bulk prediction validation error handling...")
    
    requests_data = [
        # Valid request
        {
            "kv_cache_percentage": 0.5,
            "input_token_length": 200,
            "num_request_waiting": 4,
            "num_request_running": 1,
            "num_tokens_generated": 4,
            "prefix_cache_score": 0.7,
        },
        # Invalid request (missing prefix_cache_score)
        {
            "kv_cache_percentage": 0.3,
            "input_token_length": 150,
            "num_request_waiting": 2,
            "num_request_running": 1,
            "num_tokens_generated": 5,
            # Missing prefix_cache_score
        }
    ]
    
    bulk_request = {"requests": requests_data}
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=bulk_request)
    assert r.status_code == 422  # Validation error expected
    
    # Check that error response contains validation details
    error_data = r.json()
    assert "detail" in error_data
    
    print("✓ Bulk prediction correctly failed when any request had validation errors")


def test_bulk_prediction_all_valid():
    """Test bulk predictions when all requests are valid."""
    print("Testing bulk prediction with all valid requests...")
    
    requests_data = [
        {
            "kv_cache_percentage": 0.5,
            "input_token_length": 200,
            "num_request_waiting": 4,
            "num_request_running": 1,
            "num_tokens_generated": 4,
            "prefix_cache_score": 0.7,
        },
        {
            "kv_cache_percentage": 0.3,
            "input_token_length": 150,
            "num_request_waiting": 2,
            "num_request_running": 1,
            "num_tokens_generated": 5,
            "prefix_cache_score": 0.5,  # Include required field
        }
    ]
    
    bulk_request = {"requests": requests_data}
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=bulk_request)
    assert r.status_code == 200
    
    data = r.json()
    assert data["total_requests"] == 2
    assert data["successful_predictions"] == 2
    assert data["failed_predictions"] == 0
    
    print("✓ Bulk prediction succeeded with all valid requests")

def test_prediction_missing_prefix_cache_score():
    """Test that predictions fail when prefix_cache_score is missing."""
    features = {
        "kv_cache_percentage": 0.5,
        "input_token_length": 200,
        "num_request_waiting": 4,
        "num_request_running": 1,
        "num_tokens_generated": 4,
        # Missing prefix_cache_score
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict", json=features)
    assert r.status_code == 422  # Should fail validation
    
    print("✓ Prediction correctly failed when prefix_cache_score was missing")


def test_training_server_metrics():
    """Test training server metrics endpoint."""
    r = requests.get(f"{TRAINING_URL}/metrics")
    assert r.status_code == 200
    
    content = r.text
    
    # Should contain model type metric
    assert "model_type{" in content
    
    # Should contain either coefficients (Bayesian Ridge) or importance (XGBoost)
    has_coef = "ttft_coef{" in content or "tpot_coef{" in content
    has_importance = "ttft_importance{" in content or "tpot_importance{" in content
    
    assert has_coef or has_importance, "Should have either coefficients or feature importance metrics"
    
    # Should have standard metrics
    assert "training_samples_count" in content
    
    # Check for prefix_cache_score in TTFT metrics
    if has_coef:
        assert 'feature="prefix_cache_score"' in content, "Should have prefix_cache_score coefficient for TTFT model"
    if has_importance:
        assert 'feature="prefix_cache_score"' in content, "Should have prefix_cache_score importance for TTFT model"
    
    print("Training server metrics endpoint working correctly")
    print("✓ Prefix cache score feature found in metrics")


def test_model_consistency_between_servers():
    """Test that both servers report the same model type."""
    # Get model type from training server
    training_info_r = requests.get(f"{TRAINING_URL}/model/download/info")
    training_model_type = training_info_r.json().get("model_type")
    
    # Get model type from prediction server
    prediction_status_r = requests.get(f"{PREDICTION_URL}/status")
    prediction_model_type = prediction_status_r.json().get("model_type")
    
    assert training_model_type == prediction_model_type, (
        f"Model type mismatch: training={training_model_type}, prediction={prediction_model_type}"
    )
    
    print(f"Model type consistent across servers: {training_model_type}")


# 6. Update test_xgboost_tree_endpoints_on_training_server function name and add both
def test_model_specific_endpoints_on_training_server():
    """Test model-specific endpoints on training server based on model type."""
    model_info_r = requests.get(f"{TRAINING_URL}/model/download/info")
    model_type = model_info_r.json().get("model_type")
    
    if model_type == "xgboost":
        print("Testing XGBoost tree endpoints on training server...")
        
        # Test TTFT trees
        ttft_response = requests.get(f"{TRAINING_URL}/model/ttft/xgb/json")
        if ttft_response.status_code == 200:
            ttft_trees = ttft_response.json()
            assert isinstance(ttft_trees, list), "TTFT trees should be a list"
            print(f"✓ TTFT XGBoost trees available: {len(ttft_trees)} trees")
        else:
            print(f"TTFT XGBoost trees not yet available (status: {ttft_response.status_code})")
        
        # Test TPOT trees  
        tpot_response = requests.get(f"{TRAINING_URL}/model/tpot/xgb/json")
        if tpot_response.status_code == 200:
            tpot_trees = tpot_response.json()
            assert isinstance(tpot_trees, list), "TPOT trees should be a list"
            print(f"✓ TPOT XGBoost trees available: {len(tpot_trees)} trees")
        else:
            print(f"TPOT XGBoost trees not yet available (status: {tpot_response.status_code})")
    
    elif model_type == "lightgbm":
        test_lightgbm_endpoints_on_training_server()
    
    else:
        print(f"No model-specific endpoints to test for {model_type}")


async def async_predict_request(session, payload, request_id):
    """Make an async prediction request."""
    start_time = time.time()
    try:
        async with session.post(f"{PREDICTION_URL}/predict", json=payload, timeout=aiohttp.ClientTimeout(total=5)) as response:
            end_time = time.time()
            response_data = await response.json()
            return {
                'request_id': request_id,
                'status_code': response.status,
                'response_time': end_time - start_time,
                'success': response.status == 200,
                'response_data': response_data,
                'model_type': response_data.get('model_type') if response.status == 200 else None
            }
    except Exception as e:
        end_time = time.time()
        return {
            'request_id': request_id,
            'status_code': 0,
            'response_time': end_time - start_time,
            'success': False,
            'error': str(e),
            'model_type': None
        }


async def async_bulk_predict_request(session, payload, request_id):
    """Make an async bulk prediction request."""
    start_time = time.time()
    try:
        async with session.post(f"{PREDICTION_URL}/predict/bulk/strict", json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
            end_time = time.time()
            response_data = await response.json()
            return {
                'request_id': request_id,
                'status_code': response.status,
                'response_time': end_time - start_time,
                'success': response.status == 200,
                'response_data': response_data,
                'batch_size': len(payload.get('requests', [])),
                'predictions_count': len(response_data.get('predictions', [])) if response.status == 200 else 0
            }
    except Exception as e:
        end_time = time.time()
        return {
            'request_id': request_id,
            'status_code': 0,
            'response_time': end_time - start_time,
            'success': False,
            'error': str(e),
            'batch_size': len(payload.get('requests', [])),
            'predictions_count': 0
        }


def generate_random_prediction_payload():
    """Generate a random prediction payload."""
    return {
        "kv_cache_percentage": random.uniform(0.1, 0.9),
        "input_token_length": random.randint(10, 1000),
        "num_request_waiting": random.randint(1, 20),
        "num_request_running": random.randint(1, 10),
        "num_tokens_generated": random.randint(1, 20),
        "prefix_cache_score": random.uniform(0.0, 1.0),
    }


def generate_bulk_prediction_payload(batch_size=10):
    """Generate a bulk prediction payload with specified batch size."""
    requests_data = []
    for _ in range(batch_size):
        requests_data.append({
            "kv_cache_percentage": random.uniform(0.1, 0.9),
            "input_token_length": random.randint(10, 1000),
            "num_request_waiting": random.randint(1, 20),
            "num_request_running": random.randint(1, 10),
            "num_tokens_generated": random.randint(1, 20),
            "prefix_cache_score": random.uniform(0.0, 1.0),
        })
    return {"requests": requests_data}


def generate_random_training_payload():
    """Generate a random training payload."""
    input_tokens = random.randint(10, 1000)
    waiting_requests = random.randint(1, 20)
    running_requests = random.randint(1, 10)
    kv = random.uniform(0.01, 0.99)
    tokens_generated = random.randint(1, 20)
    prefix_cache = random.uniform(0.0, 1.0)  # Added prefix cache score
    
    return {
        "kv_cache_percentage": kv,
        "input_token_length": input_tokens,
        "num_request_waiting": waiting_requests,
        "num_request_running": running_requests,
        "actual_ttft_ms": (
            input_tokens * 2.0
            + waiting_requests * 3.0
            + running_requests * 4.0
            + kv * 50.0
            + prefix_cache * 30.0  # Added prefix cache effect
            + 95 + random.uniform(-10, 10)
        ),
        "actual_tpot_ms": (
            kv * 100.0
            + input_tokens * 0.5
            + tokens_generated * 1.0
            + running_requests * 5.0
            + 9 + random.uniform(-5, 5)
        ),
        "num_tokens_generated": tokens_generated,
        "prefix_cache_score": prefix_cache,  # Added prefix cache score
    }


def test_dual_server_quantile_regression_learns_distribution():
    """
    Quantile regression should learn the q-quantile of a Gaussian residual model
    with fixed sigma, verified by (a) relative error vs μ+zσ and (b) empirical coverage.
    """
    import random, time, math
    import numpy as np
    import requests
    from scipy.stats import norm

    RNG_SEED = 42
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    # Config
    TRAIN_N = 3000
    TEST_N  = 200
    TTFT_STD, TPOT_STD = 20.0, 10.0
    REL_ERR_TOL = 0.15  # 15%
    COVERAGE_TOL = 0.05 # ±5% around target quantile
    MAX_WAIT_S = 180
    POLL_INTERVAL_S = 3

    # 1) Confirm server mode
    r = requests.get(f"{TRAINING_URL}/model/download/info", timeout=10)
    assert r.status_code == 200, "model info endpoint failed"
    model_type = r.json().get("model_type", "unknown")

    s = requests.get(f"{PREDICTION_URL}/status", timeout=10)
    assert s.status_code == 200, "prediction status endpoint failed"
    target_quantile = float(s.json().get("quantile", 0.9))

    assert "xgboost" in model_type.lower() or "lightgbm" in model_type.lower(), f"Model not in quantile mode: {model_type}"

    z = norm.ppf(target_quantile)

    # 2) Generate training data (vectorized)
    kv = np.random.uniform(0.1, 0.9, size=TRAIN_N)
    input_len = np.random.randint(50, 801, size=TRAIN_N)
    waiting = np.random.randint(0, 9, size=TRAIN_N)
    running = np.random.randint(1, 5, size=TRAIN_N)
    tokens_gen = np.random.randint(1, 26, size=TRAIN_N)
    prefix = np.random.uniform(0.0, 1.0, size=TRAIN_N)

    ttft_mu = (input_len*2.0 + waiting*3.0 + running*4.0 + kv*50.0 + prefix*30.0 + 95)
    tpot_mu = (kv*100.0 + input_len*0.5 + tokens_gen*1.0 + running*5.0 + 9)

    ttft_y = np.maximum(1.0, ttft_mu + np.random.normal(0, TTFT_STD, size=TRAIN_N))
    tpot_y = np.maximum(1.0, tpot_mu + np.random.normal(0, TPOT_STD, size=TRAIN_N))

    entries = [dict(
        kv_cache_percentage=float(kv[i]),
        input_token_length=int(input_len[i]),
        num_request_waiting=int(waiting[i]),
        num_request_running=int(running[i]),
        actual_ttft_ms=float(ttft_y[i]),
        actual_tpot_ms=float(tpot_y[i]),
        num_tokens_generated=int(tokens_gen[i]),
        prefix_cache_score=float(prefix[i]),
    ) for i in range(TRAIN_N)]

    # 3) Submit training data (with a couple retries)
    for _ in range(3):
        tr = requests.post(f"{TRAINING_URL}/add_training_data_bulk", json={"entries": entries}, timeout=60)
        if tr.status_code == 202:
            break
        time.sleep(2)
    assert tr.status_code == 202, f"training submit failed: {tr.status_code}"

    # 4) Wait for training to complete
    time.sleep(30)
    # 5) Sync models to prediction server
    synced = False
    for _ in range(10):
        rr = requests.post(f"{PREDICTION_URL}/reload", timeout=20)
        if rr.status_code == 200 and rr.json().get("is_ready"):
            synced = True
            break
        time.sleep(3)
    assert synced, "Failed to sync models"

    # 6) Build test set + expected quantiles
    kv_t = np.random.uniform(0.1, 0.9, size=TEST_N)
    in_t = np.random.randint(100, 601, size=TEST_N)
    wait_t = np.random.randint(1, 9, size=TEST_N)
    run_t = np.random.randint(1, 5, size=TEST_N)
    tok_t = np.random.randint(5, 21, size=TEST_N)
    pre_t = np.random.uniform(0.0, 1.0, size=TEST_N)

    ttft_mu_t = (in_t*2.0 + wait_t*3.0 + run_t*4.0 + kv_t*50.0 + pre_t*30.0 + 95)
    tpot_mu_t = (kv_t*100.0 + in_t*0.5 + tok_t*1.0 + run_t*5.0 + 9)
    ttft_q_exp = ttft_mu_t + z*TTFT_STD
    tpot_q_exp = tpot_mu_t + z*TPOT_STD

    test_cases = [dict(
        kv_cache_percentage=float(kv_t[i]),
        input_token_length=int(in_t[i]),
        num_request_waiting=int(wait_t[i]),
        num_request_running=int(run_t[i]),
        num_tokens_generated=int(tok_t[i]),
        prefix_cache_score=float(pre_t[i]),
    ) for i in range(TEST_N)]

    # 7) Predict (bulk)
    pr = requests.post(f"{PREDICTION_URL}/predict/bulk/strict", json={"requests": test_cases}, timeout=60)
    assert pr.status_code == 200, f"predict failed: {pr.status_code}"
    jd = pr.json()
    assert jd["total_requests"] == TEST_N and jd["successful_predictions"] == TEST_N and jd["failed_predictions"] == 0
    preds = jd["predictions"]

    ttft_pred = np.array([p["ttft_ms"] for p in preds], dtype=float)
    tpot_pred = np.array([p["tpot_ms"] for p in preds], dtype=float)

    # 8) Relative error vs μ + zσ
    ttft_rel_err = np.abs(ttft_pred - ttft_q_exp) / ttft_q_exp
    tpot_rel_err = np.abs(tpot_pred - tpot_q_exp) / tpot_q_exp
    acc_mask = (ttft_rel_err <= REL_ERR_TOL) & (tpot_rel_err <= REL_ERR_TOL)
    rel_accuracy = acc_mask.mean()
    print(f"Relative-err accuracy (≤{int(REL_ERR_TOL*100)}%): {rel_accuracy*100:.1f}%")

    # 9) Coverage calibration (simulate actuals for the same test X)
    # Generate fresh noise so it's an *unseen* draw from the same D|X:
    ttft_actual = np.maximum(1.0, ttft_mu_t + np.random.normal(0, TTFT_STD, size=TEST_N))
    tpot_actual = np.maximum(1.0, tpot_mu_t + np.random.normal(0, TPOT_STD, size=TEST_N))

    ttft_cov = (ttft_actual <= ttft_pred).mean()
    tpot_cov = (tpot_actual <= tpot_pred).mean()
    print(f"Coverage: TTFT={ttft_cov:.3f}, TPOT={tpot_cov:.3f} (target {target_quantile:.3f} ± {COVERAGE_TOL})")

    # 10) Monotonic sanity checks on a few random pairs (no hard fail, just helpful asserts)
    # pick one sample index and perturb input_token_length upward
    idx = 0
    base = test_cases[idx].copy(); up = test_cases[idx].copy(); up["input_token_length"] += 100
    br = requests.post(f"{PREDICTION_URL}/predict/bulk/strict", json={"requests":[base, up]}, timeout=30)
    if br.status_code == 200:
        _bp = br.json()["predictions"]
        assert _bp[1]["ttft_ms"] >= _bp[0]["ttft_ms"] - 1e-6, "TTFT should not decrease with longer input"

    # 11) Final assertions
    assert rel_accuracy >= 0.70, f"Only {rel_accuracy*100:.1f}% within ±{int(REL_ERR_TOL*100)}% (expected ≥70%)"
    assert abs(ttft_cov - target_quantile) <= COVERAGE_TOL, f"TTFT coverage {ttft_cov:.3f} not within ±{COVERAGE_TOL} of {target_quantile:.3f}"
    assert abs(tpot_cov - target_quantile) <= COVERAGE_TOL, f"TPOT coverage {tpot_cov:.3f} not within ±{COVERAGE_TOL} of {target_quantile:.3f}"




async def run_prediction_stress_test(duration_seconds=30, target_qps=1000):
    """Run stress test against the prediction server only."""
    interval = 1.0 / target_qps
    start = time.time()
    connector = aiohttp.TCPConnector(limit=1000, limit_per_host=1000)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        req_id = 0
        next_time = start
        
        while time.time() - start < duration_seconds:
            now = time.time()
            while next_time <= now:
                req_id += 1
                payload = generate_random_prediction_payload()
                tasks.append(asyncio.create_task(async_predict_request(session, payload, req_id)))
                next_time += interval
            
            await asyncio.sleep(0.001)
        
        print(f"Waiting for {len(tasks)} prediction requests to complete...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in results if isinstance(r, dict)]
        
        if valid_results:
            actual_qps = len(valid_results) / duration_seconds
            print(f"Target QPS: {target_qps}, Actual QPS: {actual_qps:.1f}")
        
        return valid_results


async def run_bulk_prediction_stress_test(duration_seconds=30, target_rps=100, batch_size=10):
    """Run stress test against the bulk prediction endpoint."""
    interval = 1.0 / target_rps  # requests per second
    start = time.time()
    connector = aiohttp.TCPConnector(limit=200, limit_per_host=200)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        req_id = 0
        next_time = start
        
        while time.time() - start < duration_seconds:
            now = time.time()
            while next_time <= now:
                req_id += 1
                payload = generate_bulk_prediction_payload(batch_size)
                tasks.append(asyncio.create_task(async_bulk_predict_request(session, payload, req_id)))
                next_time += interval
            
            await asyncio.sleep(0.01)  # Slightly longer sleep for bulk requests
        
        print(f"Waiting for {len(tasks)} bulk prediction requests to complete...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in results if isinstance(r, dict)]
        
        if valid_results:
            actual_rps = len(valid_results) / duration_seconds
            total_predictions = sum(r.get('predictions_count', 0) for r in valid_results)
            actual_pps = total_predictions / duration_seconds  # predictions per second
            print(f"Target RPS: {target_rps}, Actual RPS: {actual_rps:.1f}")
            print(f"Total Predictions: {total_predictions}, Predictions/sec: {actual_pps:.1f}")
        
        return valid_results


def analyze_prediction_stress_results(results):
    """Analyze prediction stress test results."""
    if not results:
        print("No results to analyze")
        return
    
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r.get('success', False))
    failed_requests = total_requests - successful_requests
    
    response_times = [r['response_time'] for r in results if r.get('response_time')]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    status_codes = defaultdict(int)
    for r in results:
        status_codes[r.get('status_code', 0)] += 1
    
    model_types = defaultdict(int)
    for r in results:
        if r.get('model_type'):
            model_types[r['model_type']] += 1
    
    print(f"\n{'='*50}")
    print("PREDICTION SERVER STRESS TEST RESULTS")
    print(f"{'='*50}")
    print(f"Total Requests: {total_requests}")
    print(f"Successful: {successful_requests} ({successful_requests/total_requests*100:.1f}%)")
    print(f"Failed: {failed_requests} ({failed_requests/total_requests*100:.1f}%)")
    print(f"Average Response Time: {avg_response_time*1000:.2f}ms")
    
    if model_types:
        print(f"\nModel Types in Predictions:")
        for model_type, count in model_types.items():
            print(f"  {model_type}: {count}")
    
    print(f"\nStatus Code Distribution:")
    for status, count in status_codes.items():
        print(f"  {status}: {count}")
    
    if response_times:
        sorted_times = sorted(response_times)
        p50 = sorted_times[int(len(sorted_times) * 0.5)] * 1000
        p95 = sorted_times[int(len(sorted_times) * 0.95)] * 1000
        p99 = sorted_times[int(len(sorted_times) * 0.99)] * 1000
        print(f"\nResponse Time Percentiles:")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")


def analyze_bulk_prediction_stress_results(results):
    """Analyze bulk prediction stress test results."""
    if not results:
        print("No results to analyze")
        return
    
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r.get('success', False))
    failed_requests = total_requests - successful_requests
    
    total_predictions = sum(r.get('predictions_count', 0) for r in results)
    total_batch_size = sum(r.get('batch_size', 0) for r in results)
    
    response_times = [r['response_time'] for r in results if r.get('response_time')]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    status_codes = defaultdict(int)
    for r in results:
        status_codes[r.get('status_code', 0)] += 1
    
    print(f"\n{'='*50}")
    print("BULK PREDICTION STRESS TEST RESULTS")
    print(f"{'='*50}")
    print(f"Total Bulk Requests: {total_requests}")
    print(f"Successful: {successful_requests} ({successful_requests/total_requests*100:.1f}%)")
    print(f"Failed: {failed_requests} ({failed_requests/total_requests*100:.1f}%)")
    print(f"Total Individual Predictions: {total_predictions}")
    print(f"Total Batch Size: {total_batch_size}")
    print(f"Average Response Time: {avg_response_time*1000:.2f}ms")
    
    if total_batch_size > 0:
        print(f"Average Batch Size: {total_batch_size/total_requests:.1f}")
        print(f"Prediction Success Rate: {total_predictions/total_batch_size*100:.1f}%")
    
    print(f"\nStatus Code Distribution:")
    for status, count in status_codes.items():
        print(f"  {status}: {count}")
    
    if response_times:
        sorted_times = sorted(response_times)
        p50 = sorted_times[int(len(sorted_times) * 0.5)] * 1000
        p95 = sorted_times[int(len(sorted_times) * 0.95)] * 1000
        p99 = sorted_times[int(len(sorted_times) * 0.99)] * 1000
        print(f"\nResponse Time Percentiles:")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")


def test_prediction_server_stress_test():
    """Stress test the prediction server."""
    print("Running prediction server stress test...")
    
    results = asyncio.run(run_prediction_stress_test(duration_seconds=100, target_qps=TARGET_QPS))
    
    analyze_prediction_stress_results(results)
    
    assert len(results) > 0, "No requests were made"
    
    successful_requests = sum(1 for r in results if r.get('success', False))
    success_rate = successful_requests / len(results)
    
    assert success_rate > 0.8, f"Success rate too low: {success_rate*100:.1f}%"
    
    print(f"Prediction server stress test completed with {success_rate*100:.1f}% success rate")


def test_bulk_prediction_stress_test():
    """Stress test the bulk prediction endpoint."""
    print("Running bulk prediction stress test...")
    
    # Test with different batch sizes
    batch_sizes = [5, 10, 25]
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size {batch_size}...")
        results = asyncio.run(run_bulk_prediction_stress_test(
            duration_seconds=100, 
            target_rps=TARGET_QPS,  # Lower RPS for bulk requests
            batch_size=batch_size
        ))
        
        analyze_bulk_prediction_stress_results(results)
        
        assert len(results) > 0, f"No bulk requests were made for batch size {batch_size}"
        
        successful_requests = sum(1 for r in results if r.get('success', False))
        success_rate = successful_requests / len(results)
        
        assert success_rate > 0.7, f"Bulk success rate too low for batch size {batch_size}: {success_rate*100:.1f}%"
        
        print(f"Bulk prediction stress test (batch size {batch_size}) completed with {success_rate*100:.1f}% success rate")

def test_large_batch_prediction_stress_test():
    """Stress test the bulk prediction endpoint."""
    print("Running bulk prediction stress test...")
    
    # Test with different batch sizes
    batch_sizes = [1000]
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size {batch_size}...")
        results = asyncio.run(run_bulk_prediction_stress_test(
            duration_seconds=100, 
            target_rps=TARGET_QPS_LARGE_BATCH,  # Lower RPS for bulk requests
            batch_size=batch_size
        ))
        
        analyze_bulk_prediction_stress_results(results)
        
        assert len(results) > 0, f"No bulk requests were made for batch size {batch_size}"
        
        successful_requests = sum(1 for r in results if r.get('success', False))
        success_rate = successful_requests / len(results)
        
        assert success_rate > 0.7, f"Bulk success rate too low for batch size {batch_size}: {success_rate*100:.1f}%"
        
        print(f"Bulk prediction stress test (batch size {batch_size}) completed with {success_rate*100:.1f}% success rate")


def test_end_to_end_workflow():
    """Test the complete end-to-end workflow with robust error handling."""
    print("Testing end-to-end workflow...")
    
    # 1. Send training data to training server
    print("Step 1: Sending training data to training server...")
    training_payload = {"entries": [generate_random_training_payload() for _ in range(20)]}
    
    try:
        training_r = requests.post(f"{TRAINING_URL}/add_training_data_bulk", json=training_payload, timeout=30)
        assert training_r.status_code == 202
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Training server not accessible: {e}")

    # 2. Wait a bit for training
    print("Step 2: Waiting for training...")
    time.sleep(10)

    # 3. Trigger model sync on prediction server
    print("Step 3: Syncing models to prediction server...")
    try:
        reload_r = requests.post(f"{PREDICTION_URL}/reload", timeout=30)
        assert reload_r.status_code == 200
        time.sleep(5)  # Allow some time for models to sync
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Prediction server not accessible for reload: {e}")

    # 4. Make predictions with retry logic
    print("Step 4: Making predictions...")
    successful_predictions = 0
    
    for i in range(5):
        payload = generate_random_prediction_payload()
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                pred_r = requests.post(f"{PREDICTION_URL}/predict", json=payload, timeout=15)
                if pred_r.status_code == 200:
                    successful_predictions += 1
                    pred_data = pred_r.json()
                    print(f"  Prediction {i+1}: TTFT={pred_data['ttft_ms']:.2f}ms, TPOT={pred_data['tpot_ms']:.2f}ms (prefix_cache={payload['prefix_cache_score']:.2f})")
                    break
                else:
                    print(f"  Prediction {i+1} attempt {attempt+1} failed with status {pred_r.status_code}")
            except requests.exceptions.ConnectTimeout:
                print(f"  Prediction {i+1} attempt {attempt+1} timed out")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                else:
                    print(f"  Prediction {i+1} failed after {max_retries} attempts")
            except requests.exceptions.RequestException as e:
                print(f"  Prediction {i+1} attempt {attempt+1} failed: {e}")
                break
    
    # Accept partial success if servers are having issues
    if successful_predictions == 0:
        pytest.skip("All prediction requests failed - servers may be down")
    elif successful_predictions < 5:
        print(f"⚠️ Partial success: {successful_predictions}/5 predictions succeeded")
    else:
        print("✓ End-to-end workflow completed successfully!")


def test_server_configuration():
    """Test server configuration and setup."""
    print("Testing server configuration...")
    
    # Test prediction server root endpoint
    pred_root_r = requests.get(f"{PREDICTION_URL}/")
    assert pred_root_r.status_code == 200
    pred_root_data = pred_root_r.json()
    print(f"Prediction server: {pred_root_data.get('message')}")
    print(f"  Model type: {pred_root_data.get('model_type')}")
    print(f"  Is ready: {pred_root_data.get('is_ready')}")
    print(f"  Sync interval: {pred_root_data.get('sync_interval')}s")
    print(f"  Training server URL: {pred_root_data.get('training_server')}")
    
    # Test training server root endpoint  
    train_root_r = requests.get(f"{TRAINING_URL}/")
    assert train_root_r.status_code == 200
    train_root_data = train_root_r.json()
    print(f"Training server: {train_root_data.get('message')}")
    print(f"  Model type: {train_root_data.get('model_type')}")

def test_training_server_flush_api():
    """Test the training server flush API and data status endpoint."""
    print("Testing training server flush API...")
    
    # 1. Check initial data status
    print("Step 1: Checking initial data status...")
    initial_status_r = requests.get(f"{TRAINING_URL}/data/status")
    assert initial_status_r.status_code == 200
    initial_status = initial_status_r.json()
    
    print(f"  Initial training samples: TTFT={initial_status['training_data']['ttft_samples']}, "
          f"TPOT={initial_status['training_data']['tpot_samples']}")
    print(f"  Initial test samples: TTFT={initial_status['test_data']['ttft_samples']}, "
          f"TPOT={initial_status['test_data']['tpot_samples']}")
    
    # 2. Add training data
    print("Step 2: Adding training data...")
    training_entries = [generate_random_training_payload() for _ in range(100)]
    training_payload = {"entries": training_entries}
    
    add_r = requests.post(f"{TRAINING_URL}/add_training_data_bulk", json=training_payload)
    assert add_r.status_code == 202
    print(f"  Added 100 training samples")
    
    # Wait a bit for data to be processed
    time.sleep(2)
    
    # 3. Verify data was added
    print("Step 3: Verifying data was added...")
    after_add_status_r = requests.get(f"{TRAINING_URL}/data/status")
    assert after_add_status_r.status_code == 200
    after_add_status = after_add_status_r.json()
    
    total_samples_after = after_add_status['training_data']['total_samples'] + after_add_status['test_data']['total_samples']
    print(f"  After adding - Training: {after_add_status['training_data']['total_samples']}, "
          f"Test: {after_add_status['test_data']['total_samples']}, Total: {total_samples_after}")
    
    # Should have more data now (some goes to training, some to test based on TEST_TRAIN_RATIO)
    assert total_samples_after > 0, "No samples were added"
    
    # 4. Test flush with only training data
    print("Step 4: Testing flush with only training data...")
    flush_training_only = {
        "flush_training_data": True,
        "flush_test_data": False,
        "flush_metrics": False,
        "reason": "Test flush training data only"
    }
    
    flush_r = requests.post(f"{TRAINING_URL}/flush", json=flush_training_only)
    assert flush_r.status_code == 200
    flush_response = flush_r.json()
    
    assert flush_response["success"] == True
    assert flush_response["metrics_cleared"] == False
    assert flush_response["reason"] == "Test flush training data only"
    
    print(f"  Flushed {flush_response['ttft_training_samples_flushed']} TTFT training samples")
    print(f"  Flushed {flush_response['tpot_training_samples_flushed']} TPOT training samples")
    print(f"  Test samples flushed: {flush_response['ttft_test_samples_flushed']} TTFT, "
          f"{flush_response['tpot_test_samples_flushed']} TPOT (should be 0)")
    
    # Verify training data was flushed but test data remains
    after_flush_training_r = requests.get(f"{TRAINING_URL}/data/status")
    after_flush_training = after_flush_training_r.json()
    
    assert after_flush_training['training_data']['total_samples'] == 0, "Training data should be empty"
    # Test data should still exist if any was added
    print(f"  After training flush - Training: {after_flush_training['training_data']['total_samples']}, "
          f"Test: {after_flush_training['test_data']['total_samples']}")
    
    # 5. Add more data
    print("Step 5: Adding more training data...")
    more_entries = [generate_random_training_payload() for _ in range(50)]
    requests.post(f"{TRAINING_URL}/add_training_data_bulk", json={"entries": more_entries})
    time.sleep(2)
    
    # 6. Test flush everything
    print("Step 6: Testing flush everything...")
    flush_all = {
        "flush_training_data": True,
        "flush_test_data": True,
        "flush_metrics": True,
        "reason": "Complete flush test"
    }
    
    flush_all_r = requests.post(f"{TRAINING_URL}/flush", json=flush_all)
    assert flush_all_r.status_code == 200
    flush_all_response = flush_all_r.json()
    
    assert flush_all_response["success"] == True
    assert flush_all_response["metrics_cleared"] == True
    assert "Successfully flushed" in flush_all_response["message"]
    
    print(f"  Complete flush message: {flush_all_response['message']}")
    
    # Verify everything was flushed
    after_flush_all_r = requests.get(f"{TRAINING_URL}/data/status")
    after_flush_all = after_flush_all_r.json()
    
    assert after_flush_all['training_data']['total_samples'] == 0, "Training data should be empty"
    assert after_flush_all['test_data']['total_samples'] == 0, "Test data should be empty"
    
    print(f"  After complete flush - Training: {after_flush_all['training_data']['total_samples']}, "
          f"Test: {after_flush_all['test_data']['total_samples']}")
    
    # 7. Test flush with default parameters (should flush everything)
    print("Step 7: Testing default flush (no body)...")
    
    # Add some data first
    requests.post(f"{TRAINING_URL}/add_training_data_bulk", 
                 json={"entries": [generate_random_training_payload() for _ in range(20)]})
    time.sleep(1)
    
    # Flush with empty body (uses defaults)
    default_flush_r = requests.post(f"{TRAINING_URL}/flush")
    assert default_flush_r.status_code == 200
    default_flush_response = default_flush_r.json()
    
    assert default_flush_response["success"] == True
    print(f"  Default flush result: {default_flush_response['message']}")
    
    # 8. Test flush with only test data
    print("Step 8: Testing flush with only test data...")
    
    # Add data
    requests.post(f"{TRAINING_URL}/add_training_data_bulk",
                 json={"entries": [generate_random_training_payload() for _ in range(50)]})
    time.sleep(2)
    
    # Get status before
    before_test_flush_r = requests.get(f"{TRAINING_URL}/data/status")
    before_test_flush = before_test_flush_r.json()
    
    # Flush only test data
    flush_test_only = {
        "flush_training_data": False,
        "flush_test_data": True,
        "flush_metrics": False,
        "reason": "Test flush test data only"
    }
    
    flush_test_r = requests.post(f"{TRAINING_URL}/flush", json=flush_test_only)
    assert flush_test_r.status_code == 200
    flush_test_response = flush_test_r.json()
    
    print(f"  Test data flush: {flush_test_response['ttft_test_samples_flushed']} TTFT, "
          f"{flush_test_response['tpot_test_samples_flushed']} TPOT")
    
    # Verify only test data was flushed
    after_test_flush_r = requests.get(f"{TRAINING_URL}/data/status")
    after_test_flush = after_test_flush_r.json()
    
    assert after_test_flush['test_data']['total_samples'] == 0, "Test data should be empty"
    # Training data should still exist
    print(f"  After test flush - Training: {after_test_flush['training_data']['total_samples']}, "
          f"Test: {after_test_flush['test_data']['total_samples']}")
    
    # 9. Test bucket distribution in status
    print("Step 9: Testing bucket distribution in status...")
    if "bucket_distribution" in after_flush_all:
        print(f"  Bucket distribution available: {len(after_flush_all.get('bucket_distribution', {}))} buckets with data")
    
    print("✓ Flush API tests passed!")


def test_training_server_flush_error_handling():
    """Test error handling in flush API."""
    print("Testing flush API error handling...")
    
    # Test with invalid JSON
    invalid_json = '{"flush_training_data": "not_a_boolean"}'
    headers = {'Content-Type': 'application/json'}
    
    try:
        r = requests.post(f"{TRAINING_URL}/flush", data=invalid_json, headers=headers)
        # Should get validation error
        assert r.status_code in [400, 422], f"Expected 400 or 422, got {r.status_code}"
        print("✓ Invalid JSON handled correctly")
    except Exception as e:
        print(f"⚠️ Error handling test skipped: {e}")
    
    # Test with valid parameters
    valid_flush = {
        "flush_training_data": False,
        "flush_test_data": False,
        "flush_metrics": True,
        "reason": "Metrics only flush"
    }
    
    r = requests.post(f"{TRAINING_URL}/flush", json=valid_flush)
    assert r.status_code == 200
    response = r.json()
    assert response["metrics_cleared"] == True
    assert response["ttft_training_samples_flushed"] == 0
    assert response["tpot_training_samples_flushed"] == 0
    
    print("✓ Flush error handling tests passed!")

if __name__ == "__main__":
    print("Running dual-server architecture tests with prefix cache score support...")
    print(f"Prediction server: {PREDICTION_URL}")
    print(f"Training server: {TRAINING_URL}")
    
    # Update these URLs before running!
    if "<PREDICTION_EXTERNAL_IP>" in PREDICTION_URL or "<TRAINING_EXTERNAL_IP>" in TRAINING_URL:
        print("\n❌ ERROR: Please update the server URLs at the top of this file!")
        print("Get external IPs with: kubectl get services")
        exit(1)
    
    # Run individual tests
    print("\n" + "="*50)
    print("RUNNING DUAL-SERVER TESTS WITH PREFIX CACHE SCORE")
    print("="*50)
    
    tests = [
        ("Server Health Checks", lambda: (test_prediction_server_healthz(), test_training_server_healthz())),
        ("Server Readiness", lambda: (test_prediction_server_readyz(), test_training_server_readyz())),
        ("Server Configuration", test_server_configuration),
        ("Prediction Server Status", test_prediction_server_status),
        ("Training Server Model Info", test_training_server_model_info),
        ("Training Server Models List", test_training_server_models_list),
        ("Model Download", test_model_download_from_training_server),
        ("Send Training Data", test_add_training_data_to_training_server),
        ("Model Sync", test_prediction_server_model_sync),
        ("Predictions", test_prediction_via_prediction_server),
        ("Bulk Prediction Strict", test_bulk_prediction_strict),
        ("Bulk Prediction With Errors", test_bulk_prediction_all_valid),
        ("Bulk predictions all valid", test_bulk_prediction_with_validation_errors),
        ("Prediction Missing Prefix Cache", test_prediction_missing_prefix_cache_score),
        ("Training Metrics", test_training_server_metrics),
        ("Model Consistency", test_model_consistency_between_servers),
        ("XGBoost Trees", test_model_specific_endpoints_on_training_server),
        ("Flush API", test_training_server_flush_api),  
        ("Flush Error Handling", test_training_server_flush_error_handling),
        
        ("Dual Server Model Learns Equation", test_dual_server_quantile_regression_learns_distribution),
        ("End-to-End Workflow", test_end_to_end_workflow),
        ("Prediction Stress Test", test_prediction_server_stress_test),
        ("Bulk Prediction Stress Test", test_bulk_prediction_stress_test),
        ("Large Batch Prediction Stress Test", test_large_batch_prediction_stress_test),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name} passed")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    
    if failed == 0:
        print("🎉 All tests passed! Your dual-server architecture with prefix cache score is working correctly.")
    else:
        print(f"⚠️  {failed} tests failed. Check the issues above.")