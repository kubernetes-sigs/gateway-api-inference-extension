import os
import time
import asyncio
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import random

import pytest
import requests

import joblib
import numpy as np
import tempfile
import xgboost

# Base URLs for the dual-server architecture
PREDICTION_URL = os.getenv("PREDICTION_SERVER_URL", "http://<PREDICTION_EXTERNAL_IP>")  # Update this
TRAINING_URL = os.getenv("TRAINING_SERVER_URL", "http://<TRAINING_EXTERNAL_IP>:8080")  # Update this

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
    assert data["model_type"] in ["bayesian_ridge", "xgboost"]
    
    print(f"Prediction server using model type: {data['model_type']}")
    print(f"Models ready: {data['is_ready']}")
    print(f"Models exist: {data['models_exist']}")


def test_training_server_model_info():
    """Test training server model info endpoint."""
    r = requests.get(f"{TRAINING_URL}/model/download/info")
    assert r.status_code == 200
    
    data = r.json()
    assert "model_type" in data
    assert "available_endpoints" in data
    assert data["model_type"] in ["bayesian_ridge", "xgboost"]
    
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
            
            # Test model download
            download_r = requests.get(f"{TRAINING_URL}/model/{model_name}/download")
            assert download_r.status_code == 200
            assert len(download_r.content) > 0
            print(f"Successfully downloaded {model_name} model ({len(download_r.content)} bytes)")


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
        
        entries.append({
            "kv_cache_percentage": kv,
            "input_token_length": inp_len,
            "num_request_waiting": waiting,
            "num_request_running": running,
            "actual_ttft_ms": (inp_len*2.0 + waiting*3.0 + running*4.0 + kv*50.0) + 95,
            "actual_tpot_ms": (kv*100.0 + inp_len*0.5 + tokens*1.0 + running*5.0) + 9,
            "num_tokens_generated": tokens,
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
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict", json=features)
    assert r.status_code == 200
    
    data = r.json()
    required_fields = [
        "ttft_ms", "tpot_ms", "ttft_uncertainty", "tpot_uncertainty",
        "ttft_prediction_bounds", "tpot_prediction_bounds", 
        "predicted_at", "model_type", "last_model_load"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Verify predictions are reasonable
    assert data["ttft_ms"] > 0
    assert data["tpot_ms"] > 0
    assert data["ttft_uncertainty"] >= 0
    assert data["tpot_uncertainty"] >= 0
    
    print(f"Prediction successful: TTFT={data['ttft_ms']:.2f}ms, TPOT={data['tpot_ms']:.2f}ms")
    print(f"Model type: {data['model_type']}")


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
    
    print("Training server metrics endpoint working correctly")


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


def test_xgboost_tree_endpoints_on_training_server():
    """Test XGBoost tree endpoints on training server if XGBoost is being used."""
    model_info_r = requests.get(f"{TRAINING_URL}/model/download/info")
    model_type = model_info_r.json().get("model_type")
    
    if model_type != "xgboost":
        print("Skipping XGBoost tree tests - not using XGBoost model")
        return
    
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

def test_dual_server_model_learns_equation():
    """
    Test that the dual-server architecture can learn equations end-to-end:
    1. Send training data to training server with known linear pattern
    2. Wait for training server to retrain models
    3. Trigger prediction server to sync new models
    4. Verify predictions match the known equation within tolerance
    
    Equations being learned:
    TTFT = 2*input_token_length + 3*num_request_waiting + 4*num_request_running + 50*kv_cache_percentage + 95
    TPOT = 100*kv_cache_percentage + 0.5*input_token_length + 1*num_tokens_generated + 5*num_request_running + 9
    """
    print("Testing dual-server end-to-end learning...")
    
    # Step 1: Get current model type from training server
    model_info_r = requests.get(f"{TRAINING_URL}/model/download/info")
    assert model_info_r.status_code == 200
    model_type = model_info_r.json().get("model_type", "unknown")
    print(f"Training server model type: {model_type}")
    
    # Step 2: Generate training data with known linear pattern
    print("Step 1: Generating training data with known pattern...")
    entries = []
    
    # Generate 200 training samples to ensure model learns well
    for i in range(1, 501):
        kv = random.uniform(0.1, 0.9)  # Vary KV cache
        input_len = random.randint(50, 2000)  # Vary input length
        waiting = random.randint(0, 15)  # Vary waiting requests
        running = random.randint(1, 8)  # Vary running requests  
        tokens_gen = random.randint(1, 50)  # Vary generated tokens
        
        # Apply the exact linear equations with small noise
        noise_ttft = random.uniform(-5, 5)  # Small noise
        noise_tpot = random.uniform(-3, 3)
        
        actual_ttft = (
            input_len * 2.0 
            + waiting * 3.0 
            + running * 4.0 
            + kv * 50.0 
            + 95
        ) + noise_ttft
        
        actual_tpot = (
            kv * 100.0 
            + input_len * 0.5 
            + tokens_gen * 1.0 
            + running * 5.0 
            + 9
        ) + noise_tpot
        
        entries.append({
            "kv_cache_percentage": kv,
            "input_token_length": input_len,
            "num_request_waiting": waiting,
            "num_request_running": running,
            "actual_ttft_ms": max(1.0, actual_ttft),  # Ensure positive
            "actual_tpot_ms": max(1.0, actual_tpot),  # Ensure positive
            "num_tokens_generated": tokens_gen,
        })
    
    # Step 3: Send training data to training server
    print(f"Step 2: Sending {len(entries)} training samples to training server...")
    payload = {"entries": entries}
    training_r = requests.post(f"{TRAINING_URL}/add_training_data_bulk", json=payload, timeout=30)
    assert training_r.status_code == 202, f"Training data rejected: {training_r.status_code}"
    print(f"✓ Training server accepted {len(entries)} samples")
    
    # Step 4: Wait for training to complete
    print("Step 3: Waiting for training server to retrain models...")
    training_deadline = time.time() + 120  # 2 minutes max wait for training
    
    while time.time() < training_deadline:
        # Check training server metrics to see if training happened
        try:
            metrics_r = requests.get(f"{TRAINING_URL}/metrics", timeout=10)
            if metrics_r.status_code == 200:
                metrics = metrics_r.text
                # Look for R² scores indicating training completed
                if "ttft_r2_score" in metrics and "tpot_r2_score" in metrics:
                    print("✓ Training server has R² metrics - training likely completed")
                    break
        except:
            pass
        
        print("  Waiting for training to complete...")
        time.sleep(10)
    
    # Step 5: Trigger prediction server to sync models
    print("Step 4: Syncing models to prediction server...")
    sync_deadline = time.time() + 60  # 1 minute max for model sync
    models_synced = False
    
    while time.time() < sync_deadline and not models_synced:
        try:
            # Trigger manual reload
            reload_r = requests.post(f"{PREDICTION_URL}/reload", timeout=15)
            if reload_r.status_code == 200:
                reload_data = reload_r.json()
                if reload_data.get("synced") and reload_data.get("loaded") and reload_data.get("is_ready"):
                    print("✓ Prediction server successfully synced and loaded models")
                    models_synced = True
                    break
                elif reload_data.get("is_ready"):
                    print("✓ Prediction server models are ready")
                    models_synced = True
                    break
        except Exception as e:
            print(f"  Sync attempt failed: {e}")
        
        if not models_synced:
            print("  Waiting for model sync...")
            time.sleep(5)
    
    assert models_synced, "Prediction server failed to sync models within timeout"
    
    # Step 6: Test predictions match the learned equations
    print("Step 5: Testing that predictions match learned equations...")
    
    # Define test cases with known expected outputs
    test_cases = [
        {
            "kv_cache_percentage": 0.5,
            "input_token_length": 200,
            "num_request_waiting": 4,
            "num_request_running": 2,
            "num_tokens_generated": 10,
        },
        {
            "kv_cache_percentage": 0.3,
            "input_token_length": 500,
            "num_request_waiting": 8,
            "num_request_running": 1,
            "num_tokens_generated": 25,
        },
        {
            "kv_cache_percentage": 0.8,
            "input_token_length": 100,
            "num_request_waiting": 2,
            "num_request_running": 3,
            "num_tokens_generated": 5,
        }
    ]
    
    # Calculate expected values for each test case
    tolerance = 0.15 if model_type == "xgboost" else 0.10  # XGBoost may be less precise
    all_predictions_correct = True
    
    for i, test_case in enumerate(test_cases):
        # Calculate expected values using the linear equations
        expected_ttft = (
            test_case["input_token_length"] * 2.0
            + test_case["num_request_waiting"] * 3.0
            + test_case["num_request_running"] * 4.0
            + test_case["kv_cache_percentage"] * 50.0
            + 95
        )
        
        expected_tpot = (
            test_case["kv_cache_percentage"] * 100.0
            + test_case["input_token_length"] * 0.5
            + test_case["num_tokens_generated"] * 1.0
            + test_case["num_request_running"] * 5.0
            + 9
        )
        
        # Make prediction via prediction server
        pred_r = requests.post(f"{PREDICTION_URL}/predict", json=test_case, timeout=10)
        assert pred_r.status_code == 200, f"Prediction failed for test case {i+1}"
        
        pred_data = pred_r.json()
        actual_ttft = pred_data["ttft_ms"]
        actual_tpot = pred_data["tpot_ms"]
        
        # Check if predictions are within tolerance
        ttft_error = abs(actual_ttft - expected_ttft) / expected_ttft
        tpot_error = abs(actual_tpot - expected_tpot) / expected_tpot
        
        ttft_ok = ttft_error <= tolerance
        tpot_ok = tpot_error <= tolerance
        
        print(f"  Test case {i+1}:")
        print(f"    TTFT: expected={expected_ttft:.1f}, actual={actual_ttft:.1f}, error={ttft_error*100:.1f}% {'✓' if ttft_ok else '✗'}")
        print(f"    TPOT: expected={expected_tpot:.1f}, actual={actual_tpot:.1f}, error={tpot_error*100:.1f}% {'✓' if tpot_ok else '✗'}")
        
        if not (ttft_ok and tpot_ok):
            all_predictions_correct = False
    
    # Final assertions
    if all_predictions_correct:
        print(f"🎉 SUCCESS: Dual-server architecture learned equations correctly!")
        print(f"   Model type: {model_type}")
        print(f"   Tolerance: ±{tolerance*100:.0f}%")
        print(f"   All {len(test_cases)} test cases passed")
    else:
        # Print detailed failure info
        print(f"❌ FAILURE: Model did not learn equations within {tolerance*100:.0f}% tolerance")
        
        # Get additional debug info
        try:
            status_r = requests.get(f"{PREDICTION_URL}/status")
            if status_r.status_code == 200:
                status_data = status_r.json()
                print(f"   Prediction server status: {status_data}")
        except:
            pass
        
        try:
            metrics_r = requests.get(f"{TRAINING_URL}/metrics")
            if metrics_r.status_code == 200:
                metrics = metrics_r.text
                # Extract R² scores if available
                r2_lines = [line for line in metrics.split('\n') if 'r2_score' in line]
                if r2_lines:
                    print(f"   Training server R² scores:")
                    for line in r2_lines[:4]:  # Show first few R² scores
                        print(f"     {line}")
        except:
            pass
    
    assert all_predictions_correct, f"Model learning failed - predictions not within ±{tolerance*100:.0f}% tolerance"


def test_dual_server_model_convergence_over_time():
    """
    Test that the dual-server architecture improves predictions over time
    as more training data is added.
    """
    print("Testing model convergence over multiple training iterations...")
    
    # Test features for consistent testing
    test_features = {
        "kv_cache_percentage": 0.6,
        "input_token_length": 300,
        "num_request_waiting": 5,
        "num_request_running": 2,
        "num_tokens_generated": 15,
    }
    
    # Expected values
    expected_ttft = (300 * 2.0 + 5 * 3.0 + 2 * 4.0 + 0.6 * 50.0 + 95)
    expected_tpot = (0.6 * 100.0 + 300 * 0.5 + 15 * 1.0 + 2 * 5.0 + 9)
    
    predictions_over_time = []
    
    # Send training data in batches and test convergence
    for iteration in range(1, 4):  # 3 iterations
        print(f"\nIteration {iteration}: Adding more training data...")
        
        # Generate batch of training data
        batch_entries = []
        for _ in range(50):  # 50 samples per batch
            kv = random.uniform(0.1, 0.9)
            input_len = random.randint(50, 1000)
            waiting = random.randint(0, 10)
            running = random.randint(1, 5)
            tokens_gen = random.randint(1, 30)
            
            # Add small amount of noise
            noise_ttft = random.uniform(-3, 3)
            noise_tpot = random.uniform(-2, 2)
            
            actual_ttft = (input_len * 2.0 + waiting * 3.0 + running * 4.0 + kv * 50.0 + 95) + noise_ttft
            actual_tpot = (kv * 100.0 + input_len * 0.5 + tokens_gen * 1.0 + running * 5.0 + 9) + noise_tpot
            
            batch_entries.append({
                "kv_cache_percentage": kv,
                "input_token_length": input_len,
                "num_request_waiting": waiting,
                "num_request_running": running,
                "actual_ttft_ms": max(1.0, actual_ttft),
                "actual_tpot_ms": max(1.0, actual_tpot),
                "num_tokens_generated": tokens_gen,
            })
        
        # Send to training server
        training_r = requests.post(f"{TRAINING_URL}/add_training_data_bulk", 
                                 json={"entries": batch_entries}, timeout=20)
        assert training_r.status_code == 202
        
        # Wait for training
        time.sleep(15)
        
        # Sync models to prediction server
        for attempt in range(3):  # Try up to 3 times
            reload_r = requests.post(f"{PREDICTION_URL}/reload", timeout=15)
            if reload_r.status_code == 200 and reload_r.json().get("is_ready"):
                break
            time.sleep(5)
        
        # Make prediction
        pred_r = requests.post(f"{PREDICTION_URL}/predict", json=test_features, timeout=10)
        assert pred_r.status_code == 200
        
        pred_data = pred_r.json()
        ttft_error = abs(pred_data["ttft_ms"] - expected_ttft) / expected_ttft
        tpot_error = abs(pred_data["tpot_ms"] - expected_tpot) / expected_tpot
        
        predictions_over_time.append({
            "iteration": iteration,
            "training_samples": iteration * 50,
            "ttft_prediction": pred_data["ttft_ms"],
            "tpot_prediction": pred_data["tpot_ms"],
            "ttft_error": ttft_error,
            "tpot_error": tpot_error,
        })
        
        print(f"  After {iteration * 50} samples:")
        print(f"    TTFT error: {ttft_error*100:.1f}%")
        print(f"    TPOT error: {tpot_error*100:.1f}%")
    
    # Verify that errors generally decrease over time (convergence)
    print(f"\nConvergence Analysis:")
    for pred in predictions_over_time:
        print(f"  {pred['training_samples']} samples: TTFT={pred['ttft_error']*100:.1f}%, TPOT={pred['tpot_error']*100:.1f}%")
    
    # Check that final iteration has reasonable accuracy
    final_prediction = predictions_over_time[-1]
    assert final_prediction["ttft_error"] < 0.2, f"TTFT error too high after convergence: {final_prediction['ttft_error']*100:.1f}%"
    assert final_prediction["tpot_error"] < 0.2, f"TPOT error too high after convergence: {final_prediction['tpot_error']*100:.1f}%"
    
    print(f"✓ Model convergence test passed - final errors: TTFT={final_prediction['ttft_error']*100:.1f}%, TPOT={final_prediction['tpot_error']*100:.1f}%")


def test_dual_server_model_persistence():
    """
    Test that models persist correctly across prediction server restarts
    (simulated by reloading models).
    """
    print("Testing model persistence across prediction server 'restarts'...")
    
    # Make initial prediction
    test_features = {
        "kv_cache_percentage": 0.4,
        "input_token_length": 150,
        "num_request_waiting": 3,
        "num_request_running": 1,
        "num_tokens_generated": 8,
    }
    
    pred1_r = requests.post(f"{PREDICTION_URL}/predict", json=test_features, timeout=10)
    assert pred1_r.status_code == 200
    pred1_data = pred1_r.json()
    
    print(f"Initial prediction: TTFT={pred1_data['ttft_ms']:.2f}, TPOT={pred1_data['tpot_ms']:.2f}")
    
    # Simulate "restart" by manually reloading models
    print("Simulating prediction server restart by reloading models...")
    reload_r = requests.post(f"{PREDICTION_URL}/reload", timeout=15)
    assert reload_r.status_code == 200
    assert reload_r.json().get("is_ready"), "Models should be ready after reload"
    
    # Make same prediction again
    pred2_r = requests.post(f"{PREDICTION_URL}/predict", json=test_features, timeout=10)
    assert pred2_r.status_code == 200
    pred2_data = pred2_r.json()
    
    print(f"Post-restart prediction: TTFT={pred2_data['ttft_ms']:.2f}, TPOT={pred2_data['tpot_ms']:.2f}")
    
    # Predictions should be identical (deterministic models)
    ttft_diff = abs(pred1_data["ttft_ms"] - pred2_data["ttft_ms"])
    tpot_diff = abs(pred1_data["tpot_ms"] - pred2_data["tpot_ms"])
    
    # Allow tiny differences due to floating point precision
    assert ttft_diff < 0.01, f"TTFT predictions should be identical: {ttft_diff}"
    assert tpot_diff < 0.01, f"TPOT predictions should be identical: {tpot_diff}"
    
    print("✓ Model persistence test passed - predictions identical after reload")



            
async def run_prediction_stress_test(duration_seconds=30, target_qps=2000):
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


def generate_random_prediction_payload():
    """Generate a random prediction payload."""
    return {
        "kv_cache_percentage": random.uniform(0.1, 0.9),
        "input_token_length": random.randint(10, 1000),
        "num_request_waiting": random.randint(1, 20),
        "num_request_running": random.randint(1, 10),
        "num_tokens_generated": random.randint(1, 20),
    }


def generate_random_training_payload():
    """Generate a random training payload."""
    input_tokens = random.randint(10, 1000)
    waiting_requests = random.randint(1, 20)
    running_requests = random.randint(1, 10)
    kv = random.uniform(0.01, 0.99)
    tokens_generated = random.randint(1, 20)
    
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
    }


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


def test_prediction_server_stress_test():
    """Stress test the prediction server."""
    print("Running prediction server stress test...")
    
    results = asyncio.run(run_prediction_stress_test(duration_seconds=60, target_qps=2000))
    
    analyze_prediction_stress_results(results)
    
    assert len(results) > 0, "No requests were made"
    
    successful_requests = sum(1 for r in results if r.get('success', False))
    success_rate = successful_requests / len(results)
    
    assert success_rate > 0.8, f"Success rate too low: {success_rate*100:.1f}%"
    
    print(f"Prediction server stress test completed with {success_rate*100:.1f}% success rate")


def test_end_to_end_workflow():
    """Test the complete end-to-end workflow."""
    print("Testing end-to-end workflow...")
    
    # 1. Send training data to training server
    print("Step 1: Sending training data to training server...")
    training_payload = {"entries": [generate_random_training_payload() for _ in range(20)]}
    training_r = requests.post(f"{TRAINING_URL}/add_training_data_bulk", json=training_payload)
    assert training_r.status_code == 202
    
    # 2. Wait a bit for training
    print("Step 2: Waiting for training...")
    time.sleep(10)
    
    # 3. Trigger model sync on prediction server
    #print("Step 3: Syncing models to prediction server...")
    reload_r = requests.post(f"{PREDICTION_URL}/reload")
    assert reload_r.status_code == 200
    time.sleep(5)  # Allow some time for models to sync
    # 4. Make predictions
    print("Step 4: Making predictions...")
    for i in range(5):
        payload = generate_random_prediction_payload()
        pred_r = requests.post(f"{PREDICTION_URL}/predict", json=payload)
        assert pred_r.status_code == 200
        pred_data = pred_r.json()
        print(f"  Prediction {i+1}: TTFT={pred_data['ttft_ms']:.2f}ms, TPOT={pred_data['tpot_ms']:.2f}ms")
    
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


if __name__ == "__main__":
    print("Running dual-server architecture tests...")
    print(f"Prediction server: {PREDICTION_URL}")
    print(f"Training server: {TRAINING_URL}")
    
    # Update these URLs before running!
    if "<PREDICTION_EXTERNAL_IP>" in PREDICTION_URL or "<TRAINING_EXTERNAL_IP>" in TRAINING_URL:
        print("\n❌ ERROR: Please update the server URLs at the top of this file!")
        print("Get external IPs with: kubectl get services")
        exit(1)
    
    # Run individual tests
    print("\n" + "="*50)
    print("RUNNING DUAL-SERVER TESTS")
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
        ("Training Metrics", test_training_server_metrics),
        ("Model Consistency", test_model_consistency_between_servers),
        ("XGBoost Trees", test_xgboost_tree_endpoints_on_training_server),
        ("Dual Server Model Learns Equation", test_dual_server_model_learns_equation),
        ("Dual Server Model Convergence", test_dual_server_model_convergence_over_time),
        ("Model Persistence", test_dual_server_model_persistence),
        ("End-to-End Workflow", test_end_to_end_workflow),
        ("Prediction Stress Test", test_prediction_server_stress_test),
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
        print("🎉 All tests passed! Your dual-server architecture is working correctly.")
    else:
        print(f"⚠️  {failed} tests failed. Check the issues above.")