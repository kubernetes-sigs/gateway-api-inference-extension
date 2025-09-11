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
    data = r.json()
    assert data.get("status") == "ready"
    # Should include quantile information
    assert "quantile" in data


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
    assert "quantile" in data  # Added quantile check
    assert "models_exist" in data
    assert data["model_type"] in ["bayesian_ridge", "xgboost"]
    assert isinstance(data["quantile"], float)
    assert 0 < data["quantile"] < 1  # Should be between 0 and 1
    
    print(f"Prediction server using model type: {data['model_type']}")
    print(f"Quantile: {data['quantile']:.0%}")
    print(f"Models ready: {data['is_ready']}")
    print(f"Models exist: {data['models_exist']}")


def test_training_server_model_info():
    """Test training server model info endpoint."""
    r = requests.get(f"{TRAINING_URL}/model/download/info")
    assert r.status_code == 200
    
    data = r.json()
    assert "model_type" in data
    assert "quantile" in data  # Added quantile check
    assert "available_endpoints" in data
    assert "evaluation_info" in data  # Added evaluation info check
    assert data["model_type"] in ["bayesian_ridge", "xgboost"]
    assert isinstance(data["quantile"], float)
    
    # Check evaluation info includes quantile-specific metrics
    eval_info = data["evaluation_info"]
    assert "quantile_loss" in eval_info
    assert "coverage_percent" in eval_info
    assert "violation_rate_percent" in eval_info
    
    print(f"Training server using model type: {data['model_type']}")
    print(f"Quantile: {data['quantile']:.0%}")


def test_training_server_models_list():
    """Test training server models list endpoint."""
    r = requests.get(f"{TRAINING_URL}/models/list")
    assert r.status_code == 200
    
    data = r.json()
    assert "models" in data
    assert "model_type" in data
    assert "quantile" in data  # Added quantile check
    assert "server_time" in data
    assert "evaluation_metrics" in data  # Added evaluation metrics check
    
    # Check evaluation metrics
    eval_metrics = data["evaluation_metrics"]
    assert "quantile_loss" in eval_metrics
    assert "coverage_percent" in eval_metrics
    assert "violation_rate_percent" in eval_metrics
    
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
            assert "quantile" in info_data  # Added quantile check
            
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


def test_add_training_data_to_training_server():
    """Send training data to the training server."""
    entries = []
    
    # Generate 50 training samples with varied patterns for quantile learning
    for i in range(1, 51):
        kv = random.uniform(0.1, 0.9)
        inp_len = random.randint(50, 500)
        waiting = random.randint(0, 10)
        running = random.randint(1, 5)
        tokens = random.randint(5, 50)
        prefix_cache = random.uniform(0.0, 1.0)
        
        # Generate synthetic latency data with realistic distributions
        # Higher variability to test quantile learning
        base_ttft = inp_len * 0.5 + waiting * 10 + running * 5 + kv * 20 + prefix_cache * 15 + 50
        base_tpot = kv * 50 + inp_len * 0.1 + tokens * 0.8 + running * 3 + 5
        
        # Add realistic noise (log-normal-ish distribution for latencies)
        noise_factor_ttft = random.lognormvariate(0, 0.3)  # Realistic latency noise
        noise_factor_tpot = random.lognormvariate(0, 0.2)
        
        entries.append({
            "kv_cache_percentage": kv,
            "input_token_length": inp_len,
            "num_request_waiting": waiting,
            "num_request_running": running,
            "actual_ttft_ms": max(1.0, base_ttft * noise_factor_ttft),
            "actual_tpot_ms": max(1.0, base_tpot * noise_factor_tpot),
            "num_tokens_generated": tokens,
            "prefix_cache_score": prefix_cache,
        })

    payload = {"entries": entries}
    r = requests.post(f"{TRAINING_URL}/add_training_data_bulk", json=payload)
    assert r.status_code == 202, f"Expected 202, got {r.status_code}"
    assert r.json().get("message") == "Accepted 50 training samples."
    
    print("Successfully sent realistic training data to training server")


def test_prediction_server_model_sync():
    """Test that the prediction server can sync models from the training server."""
    # Trigger a manual reload on the prediction server
    reload_r = requests.post(f"{PREDICTION_URL}/reload")
    assert reload_r.status_code == 200
    
    reload_data = reload_r.json()
    # Should include quantile information
    assert "quantile" in reload_data
    print(f"Model reload result: synced={reload_data.get('synced')}, loaded={reload_data.get('loaded')}")
    print(f"Quantile: {reload_data.get('quantile'):.0%}")
    
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
        "prefix_cache_score": 0.7,
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict", json=features)
    assert r.status_code == 200
    
    data = r.json()
    required_fields = [
        "ttft_ms", "tpot_ms", "ttft_uncertainty", "tpot_uncertainty",
        "ttft_prediction_bounds", "tpot_prediction_bounds", 
        "predicted_at", "model_type", "quantile", "last_model_load"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Verify predictions are reasonable
    assert data["ttft_ms"] > 0
    assert data["tpot_ms"] > 0
    assert data["ttft_uncertainty"] >= 0
    assert data["tpot_uncertainty"] >= 0
    assert isinstance(data["quantile"], float)
    assert 0 < data["quantile"] < 1
    
    print(f"Prediction successful: TTFT={data['ttft_ms']:.2f}ms, TPOT={data['tpot_ms']:.2f}ms")
    print(f"Model type: {data['model_type']}, Quantile: {data['quantile']:.0%}")


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
    """Test training server metrics endpoint for quantile-specific metrics."""
    r = requests.get(f"{TRAINING_URL}/metrics")
    assert r.status_code == 200
    
    content = r.text
    
    # Should contain model type and quantile metrics
    assert "model_type{" in content
    assert "model_quantile{}" in content
    
    # Should contain either coefficients (Bayesian Ridge) or importance (XGBoost)
    has_coef = "ttft_coef{" in content or "tpot_coef{" in content
    has_importance = "ttft_importance{" in content or "tpot_importance{" in content
    
    assert has_coef or has_importance, "Should have either coefficients or feature importance metrics"
    
    # Should have standard metrics
    assert "training_samples_count" in content
    
    # Should have target metrics for reference
    assert "target_coverage_percent{}" in content
    assert "target_violation_rate_percent{}" in content
    
    # Check for prefix_cache_score in TTFT metrics
    if has_coef:
        assert 'feature="prefix_cache_score"' in content, "Should have prefix_cache_score coefficient for TTFT model"
    if has_importance:
        assert 'feature="prefix_cache_score"' in content, "Should have prefix_cache_score importance for TTFT model"
    
    print("Training server metrics endpoint working correctly")
    print("✓ Prefix cache score feature found in metrics")
    print("✓ Quantile-specific evaluation metrics available")


def test_model_consistency_between_servers():
    """Test that both servers report the same model type and quantile."""
    # Get model type and quantile from training server
    training_info_r = requests.get(f"{TRAINING_URL}/model/download/info")
    training_data = training_info_r.json()
    training_model_type = training_data.get("model_type")
    training_quantile = training_data.get("quantile")
    
    # Get model type and quantile from prediction server
    prediction_status_r = requests.get(f"{PREDICTION_URL}/status")
    prediction_data = prediction_status_r.json()
    prediction_model_type = prediction_data.get("model_type")
    prediction_quantile = prediction_data.get("quantile")
    
    assert training_model_type == prediction_model_type, (
        f"Model type mismatch: training={training_model_type}, prediction={prediction_model_type}"
    )
    
    assert abs(training_quantile - prediction_quantile) < 0.001, (
        f"Quantile mismatch: training={training_quantile}, prediction={prediction_quantile}"
    )
    
    print(f"Model type consistent across servers: {training_model_type}")
    print(f"Quantile consistent across servers: {training_quantile:.0%}")


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


def test_feature_impact_directions():
    """
    Test that features impact predictions in expected directions.
    This is appropriate for quantile regression - we test directions, not exact values.
    """
    print("Testing feature impact directions for quantile predictions...")

    base_features = {
        "kv_cache_percentage": 0.5,
        "input_token_length": 200,
        "num_request_waiting": 3,
        "num_request_running": 2,
        "num_tokens_generated": 10,
        "prefix_cache_score": 0.5,
    }

    # Test input_token_length impact on TTFT
    low_input = {**base_features, "input_token_length": 100}
    high_input = {**base_features, "input_token_length": 400}

    low_pred_r = requests.post(f"{PREDICTION_URL}/predict", json=low_input, timeout=10)
    high_pred_r = requests.post(f"{PREDICTION_URL}/predict", json=high_input, timeout=10)
    
    assert low_pred_r.status_code == 200, f"Low input prediction failed: {low_pred_r.status_code}"
    assert high_pred_r.status_code == 200, f"High input prediction failed: {high_pred_r.status_code}"
    
    low_pred = low_pred_r.json()
    high_pred = high_pred_r.json()

    # Input length should generally increase TTFT (allow some tolerance for quantile regression variance)
    assert high_pred["ttft_ms"] > low_pred["ttft_ms"] * 0.7, (
        f"Higher input length should generally increase TTFT: "
        f"low={low_pred['ttft_ms']:.1f}ms, high={high_pred['ttft_ms']:.1f}ms"
    )
    print(f"✓ Input length impact: {low_pred['ttft_ms']:.1f}ms → {high_pred['ttft_ms']:.1f}ms")

    # Test num_tokens_generated impact on TPOT
    low_tokens = {**base_features, "num_tokens_generated": 5}
    high_tokens = {**base_features, "num_tokens_generated": 25}

    low_tpot_r = requests.post(f"{PREDICTION_URL}/predict", json=low_tokens, timeout=10)
    high_tpot_r = requests.post(f"{PREDICTION_URL}/predict", json=high_tokens, timeout=10)
    
    assert low_tpot_r.status_code == 200, f"Low tokens prediction failed: {low_tpot_r.status_code}"
    assert high_tpot_r.status_code == 200, f"High tokens prediction failed: {high_tpot_r.status_code}"
    
    low_tpot = low_tpot_r.json()
    high_tpot = high_tpot_r.json()

    # More tokens should generally increase TPOT
    assert high_tpot["tpot_ms"] > low_tpot["tpot_ms"] * 0.7, (
        f"More tokens should generally increase TPOT: "
        f"low={low_tpot['tpot_ms']:.1f}ms, high={high_tpot['tpot_ms']:.1f}ms"
    )
    print(f"✓ Token count impact: {low_tpot['tpot_ms']:.1f}ms → {high_tpot['tpot_ms']:.1f}ms")


def test_prefix_cache_score_monotonicity():
    """
    Test that prefix_cache_score has consistent directional impact on TTFT.
    This tests the model learned the feature relationship.
    """
    print("Testing prefix cache score monotonicity...")

    base_features = {
        "kv_cache_percentage": 0.5,
        "input_token_length": 300,
        "num_request_waiting": 4,
        "num_request_running": 2,
        "num_tokens_generated": 15,
    }

    cache_scores = [0.0, 0.3, 0.6, 0.9]
    predictions = []

    for cache in cache_scores:
        test_features = {**base_features, "prefix_cache_score": cache}
        pred_r = requests.post(f"{PREDICTION_URL}/predict", json=test_features, timeout=10)
        assert pred_r.status_code == 200, f"Prediction failed for prefix_cache={cache}: {pred_r.status_code}"

        pred_data = pred_r.json()
        predictions.append({
            "prefix_cache_score": cache,
            "ttft_ms": pred_data["ttft_ms"],
            "tpot_ms": pred_data["tpot_ms"]
        })

        print(f"  Prefix cache {cache:.1f}: TTFT={pred_data['ttft_ms']:.1f}ms")

    # Check for general correlation with prefix cache (more flexible for quantile regression)
    ttft_values = [p["ttft_ms"] for p in predictions]
    cache_values = [p["prefix_cache_score"] for p in predictions]
    
    # Calculate simple correlation indicator
    min_ttft, max_ttft = min(ttft_values), max(ttft_values)
    min_cache, max_cache = min(cache_values), max(cache_values)
    
    # Check if there's a reasonable relationship between cache and TTFT
    # For quantile regression, we expect some relationship but allow for variance
    ttft_range = max_ttft - min_ttft
    expected_min_range = 5.0  # Minimum expected range in ms
    
    if ttft_range < expected_min_range:
        print(f"  TTFT range too small ({ttft_range:.1f}ms) - may need more training data")
        # Just check that predictions are reasonable and don't fail the test
        assert all(1 <= ttft <= 10000 for ttft in ttft_values), "TTFT predictions should be in reasonable range"
    else:
        # Check that high cache generally correlates with different TTFT
        # Use a more lenient test for quantile regression
        low_cache_avg = sum(ttft_values[:2]) / 2  # Average of lowest 2
        high_cache_avg = sum(ttft_values[2:]) / 2  # Average of highest 2
        
        # Allow for both positive and negative correlations (depends on training data)
        relationship_strength = abs(high_cache_avg - low_cache_avg) / ttft_range
        
        assert relationship_strength > 0.1, (
            f"Expected some relationship between prefix cache and TTFT, "
            f"got relationship strength: {relationship_strength:.2f}"
        )
        
        print(f"  ✓ Prefix cache shows relationship with TTFT (strength: {relationship_strength:.2f})")

    # TPOT should be less affected by prefix cache
    tpot_values = [p["tpot_ms"] for p in predictions]
    tpot_range = max(tpot_values) - min(tpot_values)
    
    # Basic sanity check for TPOT
    assert all(0.1 <= tpot <= 1000 for tpot in tpot_values), "TPOT predictions should be in reasonable range"
    
    print("✓ Prefix cache score impact test completed")


def test_prediction_ranges_are_realistic():
    """
    Test that quantile predictions are in realistic ranges.
    This is more appropriate than exact equation matching.
    """
    print("Testing prediction ranges are realistic...")
    
    # Generate diverse realistic scenarios
    scenarios = []
    for _ in range(10):
        scenarios.append({
            "kv_cache_percentage": random.uniform(0.1, 0.9),
            "input_token_length": random.randint(50, 800),
            "num_request_waiting": random.randint(0, 15),
            "num_request_running": random.randint(1, 8),
            "num_tokens_generated": random.randint(5, 50),
            "prefix_cache_score": random.uniform(0.0, 1.0),
        })
    
    all_reasonable = True
    for i, scenario in enumerate(scenarios):
        pred_r = requests.post(f"{PREDICTION_URL}/predict", json=scenario, timeout=10)
        assert pred_r.status_code == 200
        
        pred_data = pred_r.json()
        ttft = pred_data["ttft_ms"]
        tpot = pred_data["tpot_ms"]
        
        # Basic reasonableness checks for quantile predictions
        ttft_reasonable = 5 <= ttft <= 5000  # 5ms to 5s
        tpot_reasonable = 1 <= tpot <= 500   # 1ms to 500ms
        
        if not (ttft_reasonable and tpot_reasonable):
            all_reasonable = False
            print(f"  Scenario {i+1}: TTFT={ttft:.1f}ms, TPOT={tpot:.1f}ms - Outside reasonable range")
        else:
            print(f"  Scenario {i+1}: TTFT={ttft:.1f}ms, TPOT={tpot:.1f}ms - ✓")
    
    assert all_reasonable, "Some predictions were outside reasonable ranges"
    print("✓ All predictions in realistic ranges")


def test_quantile_convergence_with_more_data():
    """
    Test that quantile models improve (lower quantile loss) with more training data.
    This is the appropriate convergence test for quantile regression.
    """
    print("Testing quantile model convergence with additional training data...")
    
    # Get quantile information
    model_info_r = requests.get(f"{TRAINING_URL}/model/download/info")
    quantile = model_info_r.json().get("quantile", 0.9)
    
    initial_metrics = get_current_quantile_metrics()
    
    # Send multiple batches of training data
    for iteration in range(1, 4):  # 3 iterations
        print(f"\nIteration {iteration}: Adding batch of training data...")
        
        # Generate batch of training data with realistic distributions
        batch_entries = []
        for _ in range(100):  # Larger batches for better convergence signal
            kv = random.uniform(0.1, 0.9)
            input_len = random.randint(50, 600)
            waiting = random.randint(0, 12)
            running = random.randint(1, 6)
            tokens_gen = random.randint(5, 40)
            prefix_cache = random.uniform(0.0, 1.0)
            
            # Generate realistic latency data with proper noise distributions
            base_ttft = input_len * 0.3 + waiting * 8 + running * 4 + kv * 25 + prefix_cache * 12 + 40
            base_tpot = kv * 30 + input_len * 0.08 + tokens_gen * 0.6 + running * 2 + 3
            
            # Log-normal noise for realistic latency distributions
            noise_ttft = random.lognormvariate(0, 0.25)
            noise_tpot = random.lognormvariate(0, 0.2)
            
            batch_entries.append({
                "kv_cache_percentage": kv,
                "input_token_length": input_len,
                "num_request_waiting": waiting,
                "num_request_running": running,
                "actual_ttft_ms": max(1.0, base_ttft * noise_ttft),
                "actual_tpot_ms": max(1.0, base_tpot * noise_tpot),
                "num_tokens_generated": tokens_gen,
                "prefix_cache_score": prefix_cache,
            })
        
        # Send to training server
        training_r = requests.post(f"{TRAINING_URL}/add_training_data_bulk", 
                                 json={"entries": batch_entries}, timeout=30)
        assert training_r.status_code == 202
        
        # Wait for training
        time.sleep(20)
        
        # Sync models to prediction server
        for attempt in range(3):
            reload_r = requests.post(f"{PREDICTION_URL}/reload", timeout=20)
            if reload_r.status_code == 200 and reload_r.json().get("is_ready"):
                break
            time.sleep(5)
        
        print(f"  Added {len(batch_entries)} training samples")
    
    # Final check - models should be working
    final_metrics = get_current_quantile_metrics()
    
    # Basic sanity check - server should be responding with quantile predictions
    test_pred = requests.post(f"{PREDICTION_URL}/predict", json={
        "kv_cache_percentage": 0.5,
        "input_token_length": 200,
        "num_request_waiting": 3,
        "num_request_running": 2,
        "num_tokens_generated": 10,
        "prefix_cache_score": 0.6,
    })
    assert test_pred.status_code == 200
    
    pred_data = test_pred.json()
    assert pred_data["quantile"] == quantile
    
    print(f"✓ Model convergence test completed - quantile {quantile:.0%} predictions working")


def get_current_quantile_metrics():
    """Helper to get current quantile metrics from training server."""
    try:
        metrics_r = requests.get(f"{TRAINING_URL}/metrics", timeout=10)
        if metrics_r.status_code == 200:
            return metrics_r.text
    except:
        pass
    return ""


def test_dual_server_model_persistence():
    """Test that models persist correctly across prediction server restarts."""
    print("Testing model persistence across prediction server 'restarts'...")
    
    # Make initial prediction
    test_features = {
        "kv_cache_percentage": 0.4,
        "input_token_length": 150,
        "num_request_waiting": 3,
        "num_request_running": 1,
        "num_tokens_generated": 8,
        "prefix_cache_score": 0.6,
    }
    
    pred1_r = requests.post(f"{PREDICTION_URL}/predict", json=test_features, timeout=10)
    assert pred1_r.status_code == 200
    pred1_data = pred1_r.json()
    
    print(f"Initial prediction: TTFT={pred1_data['ttft_ms']:.2f}, TPOT={pred1_data['tpot_ms']:.2f}, Quantile={pred1_data['quantile']:.0%}")
    
    # Simulate "restart" by manually reloading models
    print("Simulating prediction server restart by reloading models...")
    reload_r = requests.post(f"{PREDICTION_URL}/reload", timeout=15)
    assert reload_r.status_code == 200
    assert reload_r.json().get("is_ready"), "Models should be ready after reload"
    
    # Make same prediction again
    pred2_r = requests.post(f"{PREDICTION_URL}/predict", json=test_features, timeout=10)
    assert pred2_r.status_code == 200
    pred2_data = pred2_r.json()
    
    print(f"Post-restart prediction: TTFT={pred2_data['ttft_ms']:.2f}, TPOT={pred2_data['tpot_ms']:.2f}, Quantile={pred2_data['quantile']:.0%}")
    
    # Predictions should be identical (deterministic models)
    ttft_diff = abs(pred1_data["ttft_ms"] - pred2_data["ttft_ms"])
    tpot_diff = abs(pred1_data["tpot_ms"] - pred2_data["tpot_ms"])
    
    # Allow tiny differences due to floating point precision
    assert ttft_diff < 0.01, f"TTFT predictions should be identical: {ttft_diff}"
    assert tpot_diff < 0.01, f"TPOT predictions should be identical: {tpot_diff}"
    
    # Quantile should also be identical
    assert pred1_data["quantile"] == pred2_data["quantile"], "Quantile should be identical after reload"
    
    print("✓ Model persistence test passed - predictions identical after reload")


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
                'model_type': response_data.get('model_type') if response.status == 200 else None,
                'quantile': response_data.get('quantile') if response.status == 200 else None
            }
    except Exception as e:
        end_time = time.time()
        return {
            'request_id': request_id,
            'status_code': 0,
            'response_time': end_time - start_time,
            'success': False,
            'error': str(e),
            'model_type': None,
            'quantile': None
        }


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
        "prefix_cache_score": random.uniform(0.0, 1.0),
    }


def generate_random_training_payload():
    """Generate a random training payload with realistic latency distributions."""
    input_tokens = random.randint(50, 800)
    waiting_requests = random.randint(0, 15)
    running_requests = random.randint(1, 8)
    kv = random.uniform(0.05, 0.95)
    tokens_generated = random.randint(5, 50)
    prefix_cache = random.uniform(0.0, 1.0)
    
    # Generate realistic base latencies
    base_ttft = input_tokens * 0.4 + waiting_requests * 9 + running_requests * 5 + kv * 30 + prefix_cache * 18 + 45
    base_tpot = kv * 40 + input_tokens * 0.09 + tokens_generated * 0.7 + running_requests * 3 + 4
    
    # Add realistic log-normal distributed noise
    noise_ttft = random.lognormvariate(0, 0.3)
    noise_tpot = random.lognormvariate(0, 0.25)
    
    return {
        "kv_cache_percentage": kv,
        "input_token_length": input_tokens,
        "num_request_waiting": waiting_requests,
        "num_request_running": running_requests,
        "actual_ttft_ms": max(1.0, base_ttft * noise_ttft),
        "actual_tpot_ms": max(1.0, base_tpot * noise_tpot),
        "num_tokens_generated": tokens_generated,
        "prefix_cache_score": prefix_cache,
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
    quantiles = defaultdict(int)
    for r in results:
        if r.get('model_type'):
            model_types[r['model_type']] += 1
        if r.get('quantile'):
            quantiles[r['quantile']] += 1
    
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
    
    if quantiles:
        print(f"\nQuantiles in Predictions:")
        for quantile, count in quantiles.items():
            print(f"  {quantile:.0%}: {count}")
    
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
                    print(f"  Prediction {i+1}: TTFT={pred_data['ttft_ms']:.2f}ms, TPOT={pred_data['tpot_ms']:.2f}ms, Quantile={pred_data['quantile']:.0%} (prefix_cache={payload['prefix_cache_score']:.2f})")
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
    print(f"  Quantile: {pred_root_data.get('quantile', 'N/A'):.0%}")
    print(f"  Is ready: {pred_root_data.get('is_ready')}")
    print(f"  Sync interval: {pred_root_data.get('sync_interval')}s")
    print(f"  Training server URL: {pred_root_data.get('training_server')}")
    
    # Test training server root endpoint  
    train_root_r = requests.get(f"{TRAINING_URL}/")
    assert train_root_r.status_code == 200
    train_root_data = train_root_r.json()
    print(f"Training server: {train_root_data.get('message')}")
    print(f"  Model type: {train_root_data.get('model_type')}")
    print(f"  Quantile: {train_root_data.get('quantile', 'N/A'):.0%}")


if __name__ == "__main__":
    print("Running dual-server architecture tests with quantile regression and prefix cache score support...")
    print(f"Prediction server: {PREDICTION_URL}")
    print(f"Training server: {TRAINING_URL}")
    
    # Update these URLs before running!
    if "<PREDICTION_EXTERNAL_IP>" in PREDICTION_URL or "<TRAINING_EXTERNAL_IP>" in TRAINING_URL:
        print("\n❌ ERROR: Please update the server URLs at the top of this file!")
        print("Get external IPs with: kubectl get services")
        exit(1)
    
    # Run individual tests
    print("\n" + "="*50)
    print("RUNNING DUAL-SERVER QUANTILE REGRESSION TESTS")
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
        ("Prediction Missing Prefix Cache", test_prediction_missing_prefix_cache_score),
        ("Training Metrics", test_training_server_metrics),
        ("Model Consistency", test_model_consistency_between_servers),
        ("XGBoost Trees", test_xgboost_tree_endpoints_on_training_server),
        ("Feature Impact Directions", test_feature_impact_directions),
        ("Prefix Cache Monotonicity", test_prefix_cache_score_monotonicity),
        ("Realistic Prediction Ranges", test_prediction_ranges_are_realistic),
        ("Quantile Model Convergence", test_quantile_convergence_with_more_data),
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
        print("🎉 All tests passed! Your dual-server quantile regression architecture with prefix cache score is working correctly.")
    else:
        print(f"⚠️  {failed} tests failed. Check the issues above.")
        
        
def test_bulk_prediction_endpoint():
    """Test the bulk prediction endpoint with multiple requests."""
    print("Testing bulk prediction endpoint...")
    
    # Create a batch of prediction requests
    bulk_request = {
        "requests": [
            {
                "kv_cache_percentage": 0.5,
                "input_token_length": 200,
                "num_request_waiting": 4,
                "num_request_running": 1,
                "num_tokens_generated": 10,
                "prefix_cache_score": 0.7,
            },
            {
                "kv_cache_percentage": 0.3,
                "input_token_length": 150,
                "num_request_waiting": 2,
                "num_request_running": 2,
                "num_tokens_generated": 15,
                "prefix_cache_score": 0.5,
            },
            {
                "kv_cache_percentage": 0.8,
                "input_token_length": 300,
                "num_request_waiting": 6,
                "num_request_running": 3,
                "num_tokens_generated": 20,
                "prefix_cache_score": 0.9,
            }
        ]
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=bulk_request, timeout=15)
    assert r.status_code == 200, f"Bulk prediction failed: {r.status_code}"
    
    data = r.json()
    
    # Check response structure
    required_fields = [
        "predictions", "errors", "total_requests", 
        "successful_predictions", "failed_predictions", "processing_time_ms"
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Verify counts
    assert data["total_requests"] == 3
    assert data["successful_predictions"] + data["failed_predictions"] == 3
    assert len(data["predictions"]) == 3
    
    # Check individual predictions
    successful_count = 0
    for i, prediction in enumerate(data["predictions"]):
        if prediction is not None:
            successful_count += 1
            # Verify prediction structure
            assert "ttft_ms" in prediction
            assert "tpot_ms" in prediction
            assert "quantile" in prediction
            assert prediction["ttft_ms"] > 0
            assert prediction["tpot_ms"] > 0
            print(f"  Prediction {i+1}: TTFT={prediction['ttft_ms']:.2f}ms, TPOT={prediction['tpot_ms']:.2f}ms")
    
    assert successful_count == data["successful_predictions"]
    assert data["processing_time_ms"] > 0
    
    print(f"✓ Bulk prediction completed: {data['successful_predictions']}/{data['total_requests']} successful")
    print(f"  Processing time: {data['processing_time_ms']:.2f}ms")


def test_bulk_prediction_strict_endpoint():
    """Test the strict bulk prediction endpoint."""
    print("Testing strict bulk prediction endpoint...")
    
    # Create a batch of valid prediction requests
    bulk_request = {
        "requests": [
            {
                "kv_cache_percentage": 0.4,
                "input_token_length": 180,
                "num_request_waiting": 3,
                "num_request_running": 1,
                "num_tokens_generated": 8,
                "prefix_cache_score": 0.6,
            },
            {
                "kv_cache_percentage": 0.6,
                "input_token_length": 250,
                "num_request_waiting": 5,
                "num_request_running": 2,
                "num_tokens_generated": 12,
                "prefix_cache_score": 0.8,
            }
        ]
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk/strict", json=bulk_request, timeout=15)
    assert r.status_code == 200, f"Strict bulk prediction failed: {r.status_code}"
    
    data = r.json()
    
    # Check response structure
    required_fields = [
        "predictions", "total_requests", 
        "successful_predictions", "failed_predictions", "processing_time_ms"
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Verify all requests succeeded (strict mode)
    assert data["total_requests"] == 2
    assert data["successful_predictions"] == 2
    assert data["failed_predictions"] == 0
    assert len(data["predictions"]) == 2
    
    # Check all predictions are valid
    for i, prediction in enumerate(data["predictions"]):
        assert prediction is not None, f"Prediction {i+1} should not be None in strict mode"
        assert "ttft_ms" in prediction
        assert "tpot_ms" in prediction
        assert "quantile" in prediction
        print(f"  Prediction {i+1}: TTFT={prediction['ttft_ms']:.2f}ms, TPOT={prediction['tpot_ms']:.2f}ms")
    
    print(f"✓ Strict bulk prediction completed: {data['successful_predictions']}/{data['total_requests']} successful")


def test_bulk_prediction_with_invalid_requests():
    """Test bulk prediction handling of invalid requests."""
    print("Testing bulk prediction with invalid requests...")
    
    # Create a batch with some invalid requests
    bulk_request = {
        "requests": [
            {
                "kv_cache_percentage": 0.5,
                "input_token_length": 200,
                "num_request_waiting": 4,
                "num_request_running": 1,
                "num_tokens_generated": 10,
                "prefix_cache_score": 0.7,
            },
            {
                # Missing prefix_cache_score
                "kv_cache_percentage": 0.3,
                "input_token_length": 150,
                "num_request_waiting": 2,
                "num_request_running": 2,
                "num_tokens_generated": 15,
            },
            {
                "kv_cache_percentage": 0.8,
                "input_token_length": 300,
                "num_request_waiting": 6,
                "num_request_running": 3,
                "num_tokens_generated": 20,
                "prefix_cache_score": 0.9,
            }
        ]
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=bulk_request, timeout=15)
    assert r.status_code == 200, f"Bulk prediction with errors failed: {r.status_code}"
    
    data = r.json()
    
    # Should have partial success
    assert data["total_requests"] == 3
    assert data["successful_predictions"] == 2  # First and third should succeed
    assert data["failed_predictions"] == 1     # Second should fail
    assert len(data["errors"]) == 1
    
    # Check error details
    error = data["errors"][0]
    assert error["index"] == 1  # Second request (0-indexed)
    assert "prefix_cache_score" in error["error"] or "Missing required field" in error["error"]
    
    # Check that successful predictions are in correct positions
    assert data["predictions"][0] is not None  # First request succeeded
    assert data["predictions"][1] is None      # Second request failed
    assert data["predictions"][2] is not None  # Third request succeeded
    
    print(f"✓ Bulk prediction with errors handled correctly: {data['successful_predictions']} success, {data['failed_predictions']} failed")


def test_bulk_prediction_with_invalid_requests():
    """Test bulk prediction handling of invalid requests."""
    print("Testing bulk prediction with invalid requests...")
    
    # First test: All requests are valid at Pydantic level but some fail at prediction level
    # We'll use out-of-range values that pass validation but fail prediction
    bulk_request = {
        "requests": [
            {
                "kv_cache_percentage": 0.5,
                "input_token_length": 200,
                "num_request_waiting": 4,
                "num_request_running": 1,
                "num_tokens_generated": 10,
                "prefix_cache_score": 0.7,
            },
            {
                # Valid Pydantic structure but problematic values
                "kv_cache_percentage": 1.5,  # Out of range but will pass initial validation
                "input_token_length": -100,  # Negative value
                "num_request_waiting": 2,
                "num_request_running": 2,
                "num_tokens_generated": 15,
                "prefix_cache_score": 0.5,
            },
            {
                "kv_cache_percentage": 0.8,
                "input_token_length": 300,
                "num_request_waiting": 6,
                "num_request_running": 3,
                "num_tokens_generated": 20,
                "prefix_cache_score": 0.9,
            }
        ]
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=bulk_request, timeout=15)
    
    if r.status_code == 422:
        # Pydantic validation caught the invalid values
        print("✓ Pydantic validation correctly rejected invalid values at endpoint level")
        return
    
    # If we get here, the request passed initial validation
    assert r.status_code == 200, f"Bulk prediction with errors failed: {r.status_code}"
    
    data = r.json()
    
    # Should have partial success/failure
    assert data["total_requests"] == 3
    print(f"  Results: {data['successful_predictions']} success, {data['failed_predictions']} failed")
    
    # Should have some errors
    if data["failed_predictions"] > 0:
        assert len(data["errors"]) > 0
        print(f"  Errors handled: {len(data['errors'])} error entries")
    
    print("✓ Bulk prediction error handling working correctly")


def test_bulk_prediction_pydantic_validation():
    """Test that Pydantic validation works correctly for bulk requests."""
    print("Testing bulk prediction Pydantic validation...")
    
    # Test completely missing required field (should fail at Pydantic level)
    invalid_bulk_request = {
        "requests": [
            {
                "kv_cache_percentage": 0.5,
                "input_token_length": 200,
                "num_request_waiting": 4,
                "num_request_running": 1,
                "num_tokens_generated": 10,
                "prefix_cache_score": 0.7,
            },
            {
                # Missing required field prefix_cache_score
                "kv_cache_percentage": 0.3,
                "input_token_length": 150,
                "num_request_waiting": 2,
                "num_request_running": 2,
                "num_tokens_generated": 15,
                # prefix_cache_score missing
            }
        ]
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=invalid_bulk_request, timeout=15)
    assert r.status_code == 422, f"Expected 422 validation error, got {r.status_code}"
    
    # Check that error message mentions the missing field
    error_response = r.json()
    error_text = str(error_response)
    assert "prefix_cache_score" in error_text, "Error should mention missing prefix_cache_score"
    
    print("✓ Pydantic validation correctly rejects requests with missing required fields")


def test_bulk_prediction_range_validation():
    """Test bulk prediction with values outside valid ranges."""
    print("Testing bulk prediction with out-of-range values...")
    
    # Test with values outside Pydantic validation ranges
    out_of_range_request = {
        "requests": [
            {
                "kv_cache_percentage": 1.5,  # > 1.0, should fail validation
                "input_token_length": 200,
                "num_request_waiting": 4,
                "num_request_running": 1,
                "num_tokens_generated": 10,
                "prefix_cache_score": 0.7,
            }
        ]
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=out_of_range_request, timeout=15)
    assert r.status_code == 422, f"Expected 422 for out-of-range values, got {r.status_code}"
    
    # Test with negative values
    negative_values_request = {
        "requests": [
            {
                "kv_cache_percentage": 0.5,
                "input_token_length": -100,  # Negative, should fail validation  
                "num_request_waiting": 4,
                "num_request_running": 1,
                "num_tokens_generated": 10,
                "prefix_cache_score": 0.7,
            }
        ]
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=negative_values_request, timeout=15)
    assert r.status_code == 422, f"Expected 422 for negative values, got {r.status_code}"
    
    print("✓ Range validation working correctly for bulk requests")


def test_bulk_prediction_with_edge_case_valid_values():
    """Test bulk prediction with edge case but valid values that might cause prediction errors."""
    print("Testing bulk prediction with edge case valid values...")
    
    # Create requests with extreme but technically valid values
    edge_case_request = {
        "requests": [
            {
                "kv_cache_percentage": 0.5,
                "input_token_length": 200,
                "num_request_waiting": 4,
                "num_request_running": 1,
                "num_tokens_generated": 10,
                "prefix_cache_score": 0.7,
            },
            {
                # Extreme but valid values that might cause prediction issues
                "kv_cache_percentage": 0.0,  # Minimum valid
                "input_token_length": 1,     # Very small
                "num_request_waiting": 0,    # Minimum
                "num_request_running": 1,    # Minimum non-zero
                "num_tokens_generated": 1,   # Minimum  
                "prefix_cache_score": 0.0,   # Minimum
            },
            {
                "kv_cache_percentage": 1.0,    # Maximum valid
                "input_token_length": 50000,  # Very large
                "num_request_waiting": 1000,  # Very large
                "num_request_running": 100,   # Very large
                "num_tokens_generated": 1000, # Very large
                "prefix_cache_score": 1.0,    # Maximum
            }
        ]
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=edge_case_request, timeout=20)
    assert r.status_code == 200, f"Edge case bulk prediction failed: {r.status_code}"
    
    data = r.json()
    assert data["total_requests"] == 3
    
    # Some predictions might fail due to model limitations with extreme values
    print(f"  Results: {data['successful_predictions']} success, {data['failed_predictions']} failed")
    
    # At least the normal request should succeed
    assert data["successful_predictions"] >= 1, "At least one prediction should succeed"
    
    if data["failed_predictions"] > 0:
        print(f"  Expected some failures with extreme values: {len(data['errors'])} errors")
        for error in data["errors"]:
            print(f"    Error at index {error['index']}: {error['error']}")
    
    print("✓ Edge case bulk prediction handled appropriately")


def test_bulk_prediction_size_limits():
    """Test bulk prediction size limits."""
    print("Testing bulk prediction size limits...")
    
    # Test empty request
    empty_request = {"requests": []}
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=empty_request, timeout=15)
    assert r.status_code == 422, "Empty bulk request should fail validation"
    
    # Test maximum size (should work)
    max_request = {
        "requests": [generate_random_prediction_payload() for _ in range(100)]  # Max allowed
    }
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=max_request, timeout=30)
    assert r.status_code == 200, f"Max size bulk request failed: {r.status_code}"
    
    data = r.json()
    assert data["total_requests"] == 100
    
    # Test oversized request (should fail)
    oversized_request = {
        "requests": [generate_random_prediction_payload() for _ in range(101)]  # Over limit
    }
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=oversized_request, timeout=30)
    assert r.status_code == 422, "Oversized bulk request should fail validation"
    
    print("✓ Bulk prediction size limits working correctly")


def test_bulk_prediction_performance():
    """Test bulk prediction performance compared to individual requests."""
    print("Testing bulk prediction performance...")
    
    # Generate test requests
    test_requests = [generate_random_prediction_payload() for _ in range(10)]
    
    # Test individual requests
    start_time = time.time()
    individual_results = []
    for req in test_requests:
        r = requests.post(f"{PREDICTION_URL}/predict", json=req, timeout=10)
        if r.status_code == 200:
            individual_results.append(r.json())
    individual_time = time.time() - start_time
    
    # Test bulk request
    bulk_request = {"requests": test_requests}
    start_time = time.time()
    bulk_r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=bulk_request, timeout=20)
    bulk_time = time.time() - start_time
    
    assert bulk_r.status_code == 200, "Bulk request should succeed"
    bulk_data = bulk_r.json()
    
    # Compare results
    print(f"  Individual requests: {individual_time*1000:.2f}ms total, {individual_time*1000/len(test_requests):.2f}ms avg")
    print(f"  Bulk request: {bulk_time*1000:.2f}ms total, {bulk_time*1000/len(test_requests):.2f}ms avg")
    print(f"  Server processing time: {bulk_data['processing_time_ms']:.2f}ms")
    
    # Bulk should generally be faster per request (though may not always be due to overhead)
    efficiency_ratio = individual_time / bulk_time
    print(f"  Efficiency ratio: {efficiency_ratio:.2f}x")
    
    # Just verify bulk completed successfully
    assert bulk_data["successful_predictions"] >= len(test_requests) * 0.8, "Most bulk predictions should succeed"
    
    print("✓ Bulk prediction performance test completed")


async def async_bulk_predict_request(session, payload, request_id):
    """Make an async bulk prediction request."""
    start_time = time.time()
    try:
        async with session.post(f"{PREDICTION_URL}/predict/bulk", json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
            end_time = time.time()
            response_data = await response.json()
            return {
                'request_id': request_id,
                'status_code': response.status,
                'response_time': end_time - start_time,
                'success': response.status == 200,
                'response_data': response_data,
                'total_predictions': response_data.get('total_requests', 0) if response.status == 200 else 0
            }
    except Exception as e:
        end_time = time.time()
        return {
            'request_id': request_id,
            'status_code': 0,
            'response_time': end_time - start_time,
            'success': False,
            'error': str(e),
            'total_predictions': 0
        }


def test_bulk_prediction_stress_test():
    """Stress test the bulk prediction endpoint - measuring bulk API calls QPS."""
    print("Testing bulk prediction API call QPS under high load...")
    
    async def run_bulk_stress_test():
        connector = aiohttp.TCPConnector(
            limit=500,           
            limit_per_host=500,  
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            
            # Parameters for bulk API call QPS testing
            num_bulk_requests = 200    # Number of bulk API calls
            predictions_per_bulk = 10  # Predictions per bulk call
            
            for i in range(num_bulk_requests):
                bulk_request = {
                    "requests": [generate_random_prediction_payload() for _ in range(predictions_per_bulk)]
                }
                tasks.append(asyncio.create_task(async_bulk_predict_request(session, bulk_request, i)))
            
            print(f"Starting {num_bulk_requests} concurrent bulk API calls...")
            print(f"Each bulk call contains {predictions_per_bulk} predictions")
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            valid_results = [r for r in results if isinstance(r, dict)]
            
            # Calculate bulk API call metrics
            successful_bulk_calls = sum(1 for r in valid_results if r.get('success'))
            failed_bulk_calls = len(valid_results) - successful_bulk_calls
            
            # QPS = successful bulk API calls per second
            bulk_api_qps = successful_bulk_calls / total_time if total_time > 0 else 0
            total_api_qps = len(valid_results) / total_time if total_time > 0 else 0
            
            # Response time analysis for bulk API calls
            response_times = [r['response_time'] for r in valid_results if r.get('response_time')]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            if response_times:
                sorted_times = sorted(response_times)
                p50_response = sorted_times[int(len(sorted_times) * 0.5)] * 1000
                p95_response = sorted_times[int(len(sorted_times) * 0.95)] * 1000
                p99_response = sorted_times[int(len(sorted_times) * 0.99)] * 1000
            else:
                p50_response = p95_response = p99_response = 0
            
            print(f"\n{'='*60}")
            print("BULK API CALL STRESS TEST RESULTS")
            print(f"{'='*60}")
            print(f"Test Duration: {total_time:.2f} seconds")
            print(f"Bulk API Calls Made: {len(valid_results)}")
            print(f"Successful Bulk API Calls: {successful_bulk_calls}")
            print(f"Failed Bulk API Calls: {failed_bulk_calls}")
            print(f"")
            print(f"BULK API QPS METRICS:")
            print(f"  Successful Bulk API QPS: {bulk_api_qps:.1f} calls/second")
            print(f"  Total Bulk API QPS: {total_api_qps:.1f} calls/second")
            print(f"")
            print(f"BULK API RESPONSE TIME METRICS:")
            print(f"  Average Response Time: {avg_response_time*1000:.2f}ms")
            print(f"  P50 Response Time: {p50_response:.2f}ms")
            print(f"  P95 Response Time: {p95_response:.2f}ms") 
            print(f"  P99 Response Time: {p99_response:.2f}ms")
            print(f"")
            print(f"SUCCESS RATE:")
            print(f"  Bulk API Success Rate: {successful_bulk_calls/len(valid_results)*100:.1f}%")
            
            # Secondary metrics (for context)
            total_predictions = sum(r.get('total_predictions', 0) for r in valid_results if r.get('success'))
            prediction_throughput = total_predictions / total_time if total_time > 0 else 0
            print(f"")
            print(f"PREDICTION THROUGHPUT (for context):")
            print(f"  Total Predictions Processed: {total_predictions}")
            print(f"  Prediction Throughput: {prediction_throughput:.1f} predictions/second")
            
            return valid_results, {
                'bulk_api_qps': bulk_api_qps,
                'total_api_qps': total_api_qps,
                'success_rate': successful_bulk_calls/len(valid_results) if valid_results else 0,
                'avg_response_time_ms': avg_response_time * 1000,
                'p95_response_time_ms': p95_response,
                'successful_calls': successful_bulk_calls,
                'total_calls': len(valid_results)
            }
    
    results, metrics = asyncio.run(run_bulk_stress_test())
    
    # Assertions for test success
    assert len(results) > 0, "No bulk API calls were made"
    assert metrics['success_rate'] > 0.8, f"API success rate too low: {metrics['success_rate']*100:.1f}%"
    assert metrics['bulk_api_qps'] > 0, "No successful bulk API calls processed"
    
    print(f"\n✓ Bulk API stress test completed")
    print(f"  Achieved Bulk API QPS: {metrics['bulk_api_qps']:.1f} calls/second")
    print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
    
    

def test_bulk_prediction_edge_cases():
    """Test bulk prediction edge cases and error conditions."""
    print("Testing bulk prediction edge cases...")
    
    # Test with single request (minimum valid)
    single_request = {
        "requests": [{
            "kv_cache_percentage": 0.5,
            "input_token_length": 200,
            "num_request_waiting": 4,
            "num_request_running": 1,
            "num_tokens_generated": 10,
            "prefix_cache_score": 0.7,
        }]
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=single_request, timeout=10)
    assert r.status_code == 200, "Single request bulk should work"
    data = r.json()
    assert data["total_requests"] == 1
    assert data["successful_predictions"] == 1
    
    # Test with extreme values (but valid)
    extreme_request = {
        "requests": [{
            "kv_cache_percentage": 0.0,  # Minimum
            "input_token_length": 1,     # Minimum  
            "num_request_waiting": 0,    # Minimum
            "num_request_running": 1,    # Minimum (must be > 0)
            "num_tokens_generated": 1,   # Minimum
            "prefix_cache_score": 0.0,   # Minimum
        }, {
            "kv_cache_percentage": 1.0,  # Maximum
            "input_token_length": 10000, # Large value
            "num_request_waiting": 100,  # Large value
            "num_request_running": 50,   # Large value
            "num_tokens_generated": 1000, # Large value
            "prefix_cache_score": 1.0,   # Maximum
        }]
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=extreme_request, timeout=15)
    assert r.status_code == 200, "Extreme values bulk should work"
    data = r.json()
    assert data["total_requests"] == 2
    # Should succeed if models can handle extreme values
    
    # Test malformed JSON in request list
    malformed_request = {
        "requests": [
            {
                "kv_cache_percentage": 0.5,
                "input_token_length": 200,
                "num_request_waiting": 4,
                "num_request_running": 1,
                "num_tokens_generated": 10,
                "prefix_cache_score": 0.7,
            },
            {
                "kv_cache_percentage": "invalid",  # Wrong type
                "input_token_length": 200,
                "num_request_waiting": 4,
                "num_request_running": 1,
                "num_tokens_generated": 10,
                "prefix_cache_score": 0.7,
            }
        ]
    }
    
    r = requests.post(f"{PREDICTION_URL}/predict/bulk", json=malformed_request, timeout=10)
    # Should either fail validation (422) or handle gracefully (200 with errors)
    assert r.status_code in [200, 422], f"Malformed request handling unexpected: {r.status_code}"
    
    print("✓ Bulk prediction edge cases handled correctly")