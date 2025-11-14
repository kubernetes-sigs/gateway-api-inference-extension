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

import pytest
import requests

import joblib
import numpy as np
import tempfile
import xgboost

# Base URL of your running FastAPI server
BASE_URL = os.getenv("TRAINING_SERVER_URL", "http://34.143.221.122:80")

# Helper to wait until the server is ready
def wait_for_ready(timeout: float = 30.0, interval: float = 1.0):
    start = time.time()
    while True:
        try:
            r = requests.get(f"{BASE_URL}/readyz", timeout=2.0)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        if time.time() - start > timeout:
            pytest.skip("Server did not become ready in time")
        time.sleep(interval)

@pytest.fixture(scope="module", autouse=True)
def ensure_server_ready():
    """Wait for the /readyz endpoint before running tests."""
    wait_for_ready()


def test_healthz():
    r = requests.get(f"{BASE_URL}/healthz")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_readyz():
    r = requests.get(f"{BASE_URL}/readyz")
    assert r.status_code == 200
    assert r.json().get("status") == "ready"


def test_model_info():
    """Test the simplified /model/download/info endpoint."""
    r = requests.get(f"{BASE_URL}/model/download/info")
    assert r.status_code == 200
    
    data = r.json()
    assert "model_type" in data
    assert "model_status" in data
    assert "available_endpoints" in data
    assert data["model_type"] in ["bayesian_ridge", "xgboost"]
    assert isinstance(data["model_status"], dict)
    
    print(f"Server using model type: {data['model_type']}")
    
    if data["model_type"] == "bayesian_ridge":
        assert "coefficients_info" in data
        assert data["available_endpoints"]["coefficients"] == "/metrics"
    else:  # XGBoost
        assert "trees" in data["available_endpoints"]


def test_root_endpoint_enhanced():
    """Test the enhanced root endpoint that now includes model info."""
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    
    data = r.json()
    assert "message" in data
    assert "model_type" in data
    assert data["model_type"] in ["bayesian_ridge", "xgboost"]


def test_add_training_data_bulk():
    """
    Send 120 training samples in one bulk request so the server can retrain:
    Updated equations with prefix cache score:
      actual_ttft_ms = 2*input_token_length + 3*num_request_waiting +
                       4*num_request_running + 50*kv_cache_percentage + 
                       30*prefix_cache_score + 95
      actual_tpot_ms = 100*kv_cache_percentage + 0.5*input_token_length + 1*num_tokens_generated +
                       5*num_request_running + 9
    """
    entries = []
    common = {
        "kv_cache_percentage": 0.5,
        "num_request_running": 1,
    }

    for i in range(1, 121):
        waiting = i % 10 + 1
        tokens = waiting
        inp_len = 10 * i
        kv = common["kv_cache_percentage"]
        running = common["num_request_running"]
        prefix_cache = random.uniform(0.1, 0.9)  # Added prefix cache score
        
        entries.append({
            "kv_cache_percentage": kv,
            "input_token_length": inp_len,
            "num_request_waiting": waiting,
            "num_request_running": running,
            # Updated TTFT formula to include prefix_cache_score
            "actual_ttft_ms": (inp_len*2.0 + waiting*3.0 + running*4.0 + kv*50.0 + prefix_cache*30.0) + 95,
            # TPOT formula remains unchanged
            "actual_tpot_ms": (kv*100.0 + inp_len*0.5 + tokens*1.0 + running*5.0) + 9,
            "num_tokens_generated": tokens,
            "prefix_cache_score": prefix_cache,  # Added prefix cache score
            "timestamp": time.time()  # FastAPI will coerce to datetime
        })

    payload = {"entries": entries}
    r = requests.post(f"{BASE_URL}/add_training_data_bulk", json=payload)
    assert r.status_code == 202, f"Expected 202, got {r.status_code}"
    assert r.json().get("message") == "Accepted 120 training samples."


def test_model_learns_equation():
    """
    After sending bulk data, poll /predict until the model's predictions
    match our linear equations within tolerance, or fail after 60s.
    Updated to include prefix_cache_score in the test equation.
    """
    # First check what model type we're using
    model_info_r = requests.get(f"{BASE_URL}/model/download/info")
    model_type = model_info_r.json().get("model_type", "unknown")
    
    features = {
        "kv_cache_percentage": 0.5,
        "input_token_length": 200,
        "num_request_waiting": 4,
        "num_request_running": 1,
        "num_tokens_generated": 4,
        "prefix_cache_score": 0.7,  # Added prefix cache score
    }
    
    # Updated expected TTFT to include prefix cache score
    expected_ttft = (
        features["input_token_length"] * 2.0
        + features["num_request_waiting"] * 3.0
        + features["num_request_running"] * 4.0
        + features["kv_cache_percentage"] * 50.0
        + features["prefix_cache_score"] * 30.0  # New term
        + 95
    )
    # TPOT formula remains unchanged
    expected_tpot = (
        features["kv_cache_percentage"] * 100.0
        + features["input_token_length"] * 0.5
        + features["num_tokens_generated"] * 1.0
        + features["num_request_running"] * 5.0 + 9
    )

    # Adjust tolerance based on model type
    # XGBoost might need more tolerance for tree-based predictions
    tolerance = 0.15 if model_type == "xgboost" else 0.1
    
    deadline = time.time() + 60.0
    last_ttft, last_tpot = None, None

    while time.time() < deadline:
        r = requests.post(f"{BASE_URL}/predict", json=features)
        if r.status_code != 200:
            time.sleep(1)
            continue

        body = r.json()
        last_ttft = body["ttft_ms"]
        last_tpot = body["tpot_ms"]
        
        # Verify the response includes model_type
        assert "model_type" in body, "Response should include model_type"
        assert body["model_type"] == model_type

        ttft_ok = abs(last_ttft - expected_ttft) <= tolerance * expected_ttft
        tpot_ok = abs(last_tpot - expected_tpot) <= tolerance * expected_tpot
        if ttft_ok and tpot_ok:
            print(f"Model converged with {model_type} in {60.0 - (deadline - time.time()):.1f}s")
            print(f"  Expected TTFT: {expected_ttft:.1f}, Got: {last_ttft:.1f}")
            print(f"  Expected TPOT: {expected_tpot:.1f}, Got: {last_tpot:.1f}")
            break

        time.sleep(1)

    assert last_ttft is not None, "Never got a successful prediction."
    assert abs(last_ttft - expected_ttft) <= tolerance * expected_ttft, (
        f"TTFT={last_ttft:.1f} not within ±{tolerance*100}% of {expected_ttft:.1f} (model: {model_type})"
    )
    assert abs(last_tpot - expected_tpot) <= tolerance * expected_tpot, (
        f"TPOT={last_tpot:.1f} not within ±{tolerance*100}% of {expected_tpot:.1f} (model: {model_type})"
    )


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
    
    r = requests.post(f"{BASE_URL}/predict", json=features)
    assert r.status_code == 422  # Should fail validation
    
    print("✓ Prediction correctly failed when prefix_cache_score was missing")


def test_prefix_cache_score_impact_on_ttft():
    """
    Test that prefix_cache_score has the expected impact on TTFT predictions.
    Since our test equation has +30*prefix_cache_score, higher scores should increase TTFT.
    """
    print("Testing prefix cache score impact on TTFT predictions...")
    
    base_features = {
        "kv_cache_percentage": 0.5,
        "input_token_length": 300,
        "num_request_waiting": 4,
        "num_request_running": 2,
        "num_tokens_generated": 15,
    }
    
    prefix_cache_scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    predictions = []
    
    for prefix_score in prefix_cache_scores:
        test_features = {**base_features, "prefix_cache_score": prefix_score}
        
        pred_r = requests.post(f"{BASE_URL}/predict", json=test_features, timeout=10)
        assert pred_r.status_code == 200
        
        pred_data = pred_r.json()
        predictions.append({
            "prefix_cache_score": prefix_score,
            "ttft_ms": pred_data["ttft_ms"],
            "tpot_ms": pred_data["tpot_ms"]
        })
        
        print(f"  Prefix cache {prefix_score:.1f}: TTFT={pred_data['ttft_ms']:.1f}ms, TPOT={pred_data['tpot_ms']:.1f}ms")
    
    # Check that TTFT increases as prefix cache score increases
    # (since our test equation has +30*prefix_cache_score)
    ttft_values = [p["ttft_ms"] for p in predictions]
    
    # Calculate correlation between prefix cache score and TTFT
    first_half_avg = sum(ttft_values[:3]) / 3  # Low prefix cache scores
    second_half_avg = sum(ttft_values[3:]) / 3  # High prefix cache scores
    
    print(f"Low prefix cache avg TTFT: {first_half_avg:.1f}ms")
    print(f"High prefix cache avg TTFT: {second_half_avg:.1f}ms")
    
    # Since our training equation has +30*prefix_cache_score, higher prefix cache should increase TTFT
    ttft_difference = second_half_avg - first_half_avg
    print(f"TTFT difference (high - low prefix cache): {ttft_difference:.1f}ms")
    
    # Should be positive difference (higher prefix cache = higher TTFT in our test equation)
    assert ttft_difference > 10, f"Expected TTFT to increase with prefix cache score, got difference: {ttft_difference:.1f}ms"
    
    # TPOT should not be significantly affected by prefix cache score
    tpot_values = [p["tpot_ms"] for p in predictions]
    tpot_first_half = sum(tpot_values[:3]) / 3
    tpot_second_half = sum(tpot_values[3:]) / 3
    tpot_difference = abs(tpot_second_half - tpot_first_half)
    
    print(f"TPOT difference (should be small): {tpot_difference:.1f}ms")
    assert tpot_difference < 5, f"TPOT should not be significantly affected by prefix cache, got difference: {tpot_difference:.1f}ms"
    
    print("✓ Prefix cache score impact test passed")


def test_prediction_response_format():
    """Test that prediction responses include all expected fields including new model_type."""
    features = generate_random_prediction_payload()
    
    r = requests.post(f"{BASE_URL}/predict", json=features)
    assert r.status_code == 200
    
    data = r.json()
    required_fields = [
        "ttft_ms", "tpot_ms", "ttft_uncertainty", "tpot_uncertainty",
        "ttft_prediction_bounds", "tpot_prediction_bounds", 
        "predicted_at", "model_type"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Verify model_type is valid
    assert data["model_type"] in ["bayesian_ridge", "xgboost"]
    
    # Verify numeric fields are reasonable
    assert data["ttft_ms"] >= 0
    assert data["tpot_ms"] >= 0
    assert data["ttft_uncertainty"] >= 0
    assert data["tpot_uncertainty"] >= 0
    
    # Verify bounds are tuples
    assert len(data["ttft_prediction_bounds"]) == 2
    assert len(data["tpot_prediction_bounds"]) == 2


def test_metrics_endpoint_enhanced():
    """Test that metrics endpoint includes model-specific information with proper coefficients."""
    r = requests.get(f"{BASE_URL}/metrics")
    assert r.status_code == 200
    
    content = r.text
    
    # Should contain model type metric
    assert "model_type{" in content
    
    # Should contain either coefficients (Bayesian Ridge) or importance (XGBoost)
    has_coef = "ttft_coef{" in content or "tpot_coef{" in content
    has_importance = "ttft_importance{" in content or "tpot_importance{" in content
    
    assert has_coef or has_importance, "Should have either coefficients or feature importance metrics"
    
    # Should have standard metrics
    assert "ttft_r2_score{" in content
    assert "tpot_r2_score{" in content
    assert "training_samples_count" in content
    
    # Check for prefix_cache_score in TTFT metrics
    if has_coef:
        assert 'feature="prefix_cache_score"' in content, "Should have prefix_cache_score coefficient for TTFT model"
    if has_importance:
        assert 'feature="prefix_cache_score"' in content, "Should have prefix_cache_score importance for TTFT model"
    
    # Parse and validate coefficient values for Bayesian Ridge
    model_info_r = requests.get(f"{BASE_URL}/model/download/info")
    model_type = model_info_r.json().get("model_type")
    
    if model_type == "bayesian_ridge":
        # Check that coefficients are present and reasonable
        lines = content.split('\n')
        ttft_intercept = None
        ttft_coefs = {}
        tpot_intercept = None
        tpot_coefs = {}
        
        for line in lines:
            if line.startswith('ttft_intercept{'):
                ttft_intercept = float(line.split('}')[1].strip())
            elif line.startswith('ttft_coef{'):
                feature = line.split('feature="')[1].split('"')[0]
                value = float(line.split('}')[1].strip())
                ttft_coefs[feature] = value
            elif line.startswith('tpot_intercept{'):
                tpot_intercept = float(line.split('}')[1].strip())
            elif line.startswith('tpot_coef{'):
                feature = line.split('feature="')[1].split('"')[0]
                value = float(line.split('}')[1].strip())
                tpot_coefs[feature] = value
        
        # Validate coefficients are present
        assert ttft_intercept is not None, "TTFT intercept should be present"
        assert tpot_intercept is not None, "TPOT intercept should be present"
        
        # Updated expected features to include prefix_cache_score for TTFT
        expected_ttft_features = ["kv_cache_percentage", "input_token_length", "num_request_waiting", "num_request_running", "prefix_cache_score"]
        expected_tpot_features = ["kv_cache_percentage", "input_token_length", "num_request_waiting", "num_request_running", "num_tokens_generated"]
        
        for feature in expected_ttft_features:
            assert feature in ttft_coefs, f"TTFT coefficient for {feature} should be present"
            
        for feature in expected_tpot_features:
            assert feature in tpot_coefs, f"TPOT coefficient for {feature} should be present"
        
        print(f"✓ Bayesian Ridge coefficients validated:")
        print(f"  TTFT intercept: {ttft_intercept:.4f}")
        print(f"  TTFT coefficients: {ttft_coefs}")
        print(f"  TPOT intercept: {tpot_intercept:.4f}")
        print(f"  TPOT coefficients: {tpot_coefs}")
        
        # Validate prefix_cache_score coefficient is reasonable
        if "prefix_cache_score" in ttft_coefs:
            prefix_coef = ttft_coefs["prefix_cache_score"]
            print(f"  Prefix cache coefficient: {prefix_coef:.4f}")
            # Should be positive and reasonably close to our training value of 30
            assert 10 < prefix_coef < 50, f"Prefix cache coefficient should be reasonable: {prefix_coef}"

    print("✓ Training server metrics endpoint working correctly with prefix cache support")


def test_xgboost_tree_endpoints():
    """Test XGBoost tree endpoints if XGBoost is being used."""
    model_info_r = requests.get(f"{BASE_URL}/model/download/info")
    model_type = model_info_r.json().get("model_type")
    
    if model_type != "xgboost":
        print("Skipping XGBoost tree tests - not using XGBoost model")
        return
    
    print("Testing XGBoost tree endpoints...")
    
    # Test TTFT trees
    ttft_response = requests.get(f"{BASE_URL}/model/ttft/xgb/json")
    assert ttft_response.status_code == 200, "TTFT XGBoost trees should be available"
    ttft_trees = ttft_response.json()
    assert isinstance(ttft_trees, list), "TTFT trees should be a list"
    assert len(ttft_trees) > 0, "Should have TTFT trees"
    assert isinstance(ttft_trees[0], dict), "Each tree should be a dict"
    
    # Test TPOT trees
    tpot_response = requests.get(f"{BASE_URL}/model/tpot/xgb/json")
    assert tpot_response.status_code == 200, "TPOT XGBoost trees should be available"
    tpot_trees = tpot_response.json()
    assert isinstance(tpot_trees, list), "TPOT trees should be a list"
    assert len(tpot_trees) > 0, "Should have TPOT trees"
    assert isinstance(tpot_trees[0], dict), "Each tree should be a dict"
    
    print(f"✓ XGBoost trees available: {len(ttft_trees)} TTFT trees, {len(tpot_trees)} TPOT trees")


def test_bayesian_ridge_coefficients():
    """Test that Bayesian Ridge coefficients are properly descaled and stored."""
    model_info_r = requests.get(f"{BASE_URL}/model/download/info")
    model_type = model_info_r.json().get("model_type")
    
    if model_type != "bayesian_ridge":
        print("Skipping Bayesian Ridge coefficient tests - not using Bayesian Ridge model")
        return
    
    print("Testing Bayesian Ridge coefficient storage and retrieval...")
    
    # Get coefficients from metrics
    r = requests.get(f"{BASE_URL}/metrics")
    assert r.status_code == 200
    content = r.text
    
    # Parse coefficients from metrics
    lines = content.split('\n')
    ttft_coefs = {}
    tpot_coefs = {}
    
    for line in lines:
        if line.startswith('ttft_coef{'):
            feature = line.split('feature="')[1].split('"')[0]
            value = float(line.split('}')[1].strip())
            ttft_coefs[feature] = value
        elif line.startswith('tpot_coef{'):
            feature = line.split('feature="')[1].split('"')[0]
            value = float(line.split('}')[1].strip())
            tpot_coefs[feature] = value
    
    # Test a prediction to see if coefficients make sense
    test_features = {
        "kv_cache_percentage": 0.5,
        "input_token_length": 100,
        "num_request_waiting": 2,
        "num_request_running": 1,
        "num_tokens_generated": 5,
        "prefix_cache_score": 0.8,  # Added prefix cache score
    }
    
    # Make prediction via API
    pred_response = requests.post(f"{BASE_URL}/predict", json=test_features)
    assert pred_response.status_code == 200
    api_prediction = pred_response.json()
    
    print(f"✓ Coefficients extracted from metrics:")
    print(f"  TTFT coefficients: {ttft_coefs}")
    print(f"  TPOT coefficients: {tpot_coefs}")
    print(f"  API TTFT prediction: {api_prediction['ttft_ms']:.2f}")
    print(f"  API TPOT prediction: {api_prediction['tpot_ms']:.2f}")
    
    # Verify prefix_cache_score coefficient exists for TTFT
    assert "prefix_cache_score" in ttft_coefs, "prefix_cache_score should be in TTFT coefficients"
    assert "prefix_cache_score" not in tpot_coefs, "prefix_cache_score should NOT be in TPOT coefficients"


def test_model_endpoints_by_type():
    """Test the appropriate endpoints based on model type."""
    model_info_r = requests.get(f"{BASE_URL}/model/download/info")
    model_info = model_info_r.json()
    model_type = model_info["model_type"]
    
    print(f"Testing endpoints for model type: {model_type}")
    
    if model_type == "bayesian_ridge":
        # For Bayesian Ridge, we should have coefficients in metrics
        test_bayesian_ridge_coefficients()
        
        # XGBoost endpoints should return 404
        ttft_xgb_response = requests.get(f"{BASE_URL}/model/ttft/xgb/json")
        assert ttft_xgb_response.status_code == 404, "XGBoost endpoints should not be available for Bayesian Ridge"
        
        print("✓ Bayesian Ridge: coefficients available in metrics, XGBoost endpoints properly blocked")
        
    else:  # XGBoost
        # For XGBoost, we should have tree endpoints
        test_xgboost_tree_endpoints()
        
        print("✓ XGBoost: tree endpoints available")


def generate_random_prediction_payload():
    """Generate a random prediction payload for stress testing including prefix_cache_score."""
    return {
        "kv_cache_percentage": random.uniform(0.1, 0.9),
        "input_token_length": random.randint(10, 1000),
        "num_request_waiting": random.randint(1, 20),
        "num_request_running": random.randint(1, 10),
        "num_tokens_generated": random.randint(1, 20),
        "prefix_cache_score": random.uniform(0.0, 1.0),  # Added prefix cache score
    }


def generate_random_training_payload():
    """Generate a random training data payload for stress testing with updated TTFT formula."""
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
        # Updated linear TTFT with noise - now includes prefix_cache_score
        "actual_ttft_ms": (
            input_tokens * 2.0
            + waiting_requests * 3.0
            + running_requests * 4.0
            + kv * 50.0
            + prefix_cache * 30.0  # New term for prefix cache
            + 95 + random.uniform(-10, 10)
        ),
        # TPOT formula remains unchanged
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


def generate_bulk_training_payload(size=1000):
    """Generate a bulk training payload with specified number of entries."""
    entries = []
    for _ in range(size):
        entries.append(generate_random_training_payload())
    return {"entries": entries}


async def async_post_request(session, url, payload, request_id):
    """Make an async POST request and return result with metadata."""
    start_time = time.time()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as response:
            end_time = time.time()
            response_data = await response.json()
            return {
                'request_id': request_id,
                'status_code': response.status,
                'response_time': end_time - start_time,
                'success': response.status in [200, 202],
                'response_data': response_data,
                'request_type': 'predict' if '/predict' in url else 'training',
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
            'request_type': 'predict' if '/predict' in url else 'training',
            'model_type': None
        }

async def run_stress_test_async(duration_seconds=10, target_qps=300):
    interval = 1.0/target_qps
    start = time.time()
    connector = aiohttp.TCPConnector(limit=10000, limit_per_host=10000, ttl_dns_cache=300, use_dns_cache=True)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=2)) as sess:
        tasks = []
        req_id = 0
        next_time = start
        while time.time() - start < duration_seconds:
            now = time.time()
            while next_time <= now:
                req_id += 1
                if random.random()<0.5:
                    url = f"{BASE_URL}/predict"
                    payload = generate_random_prediction_payload()
                else:
                    url = f"{BASE_URL}/add_training_data_bulk"
                    payload = {"entries":[ generate_random_training_payload() ]}
                tasks.append(asyncio.create_task(async_post_request(sess, url, payload, req_id)))
                next_time += interval
            await asyncio.sleep(0.0001)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = [r for r in results if isinstance(r, dict)]
    
        # Calculate actual QPS achieved
        if valid_results:
            actual_duration = duration_seconds
            actual_qps = len(valid_results) / actual_duration
            print(f"Target QPS: {target_qps}, Actual QPS: {actual_qps:.0f}")
    
        return valid_results


def fetch_and_parse_xgb_json(path_suffix):
    """
    Download the XGBoost JSON dump for `path_suffix` (ttft or tpot),
    parse into a Python list of dicts, and return it.
    """
    url = f"{BASE_URL}/model/{path_suffix}/xgb/json"
    r = requests.get(url, timeout=10)
    assert r.status_code == 200, f"Failed to fetch JSON for {path_suffix}"
    trees = r.json()
    assert isinstance(trees, list), "Expected a JSON array of trees"
    assert len(trees) > 0, "Tree list should not be empty"
    assert isinstance(trees[0], dict), "Each tree must be a JSON object"
    return trees


async def async_fetch_and_parse_xgb_json(session, suffix, request_id):
    """
    Async GET /model/<suffix>/xgb/json and return timing + status.
    """
    url = f"{BASE_URL}/model/{suffix}/xgb/json"
    start = time.time()
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
            elapsed = time.time() - start
            return {
                'request_id': request_id,
                'request_type': f'download_{suffix}',
                'status_code': resp.status,
                'response_time': elapsed,
                'success': resp.status == 200,
                'tree_count': len(data) if isinstance(data, list) else None
            }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'request_id': request_id,
            'request_type': f'download_{suffix}',
            'status_code': 0,
            'response_time': elapsed,
            'success': False,
            'error': str(e)
        }


async def run_simplified_stress_test(duration_seconds=10, target_qps=2):
    """
    Simplified stress test: bulk training vs predictions and tree downloads (XGBoost only).
    """
    info_r = requests.get(f"{BASE_URL}/model/download/info", timeout=5.0)
    model_type = info_r.json().get("model_type", "bayesian_ridge")

    interval = 1.0 / target_qps
    start = time.time()
    connector = aiohttp.TCPConnector(limit=1000, limit_per_host=1000)
    async with aiohttp.ClientSession(connector=connector) as sess:
        tasks = []
        req_id = 0
        next_time = start

        while time.time() - start < duration_seconds:
            now = time.time()
            while next_time <= now:
                req_id += 1

                if random.random() < 0.5:
                    # Either predictions or tree downloads (XGBoost only)
                    if random.random() < 0.7:  # 70% predictions
                        url = f"{BASE_URL}/predict"
                        payload = generate_random_prediction_payload()
                        task = asyncio.create_task(
                            async_post_request_with_timeout(
                                sess, url, payload, req_id,
                                aiohttp.ClientTimeout(total=5), "predict"
                            )
                        )
                    else:  # 30% tree downloads (only for XGBoost)
                        if model_type == "xgboost":
                            suffix = random.choice(["ttft", "tpot"])
                            task = asyncio.create_task(
                                async_fetch_and_parse_xgb_json(sess, suffix, req_id)
                            )
                        else:
                            # For Bayesian Ridge, just do another prediction
                            url = f"{BASE_URL}/predict"
                            payload = generate_random_prediction_payload()
                            task = asyncio.create_task(
                                async_post_request_with_timeout(
                                    sess, url, payload, req_id,
                                    aiohttp.ClientTimeout(total=5), "predict"
                                )
                            )
                else:
                    # bulk training
                    url = f"{BASE_URL}/add_training_data_bulk"
                    payload = generate_bulk_training_payload(1000)
                    task = asyncio.create_task(
                        async_post_request_with_timeout(
                            sess, url, payload, req_id,
                            aiohttp.ClientTimeout(total=30), "bulk_training"
                        )
                    )

                tasks.append(task)
                next_time += interval

            await asyncio.sleep(0.001)

        print(f"Waiting for {len(tasks)} requests to complete…")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, dict)]

        if valid:
            actual_qps = len(valid) / duration_seconds
            print(f"Target QPS: {target_qps}, Actual QPS: {actual_qps:.2f}")

        return valid


async def async_post_request_with_timeout(session, url, payload, request_id, timeout, request_type):
    """Make an async POST request with custom timeout and return result with metadata."""
    start_time = time.time()
    try:
        async with session.post(url, json=payload, timeout=timeout) as response:
            end_time = time.time()
            response_data = await response.json()
            
            # Count training entries for bulk requests
            training_entries = len(payload.get("entries", [])) if request_type == "bulk_training" else 1
            
            return {
                'request_id': request_id,
                'status_code': response.status,
                'response_time': end_time - start_time,
                'success': response.status in [200, 202],
                'response_data': response_data,
                'request_type': request_type,
                'training_entries': training_entries if request_type == "bulk_training" else 0,
                'model_type': response_data.get('model_type') if response.status == 200 and request_type == 'predict' else None
            }
    except Exception as e:
        end_time = time.time()
        training_entries = len(payload.get("entries", [])) if request_type == "bulk_training" else 1
        return {
            'request_id': request_id,
            'status_code': 0,
            'response_time': end_time - start_time,
            'success': False,
            'error': str(e),
            'request_type': request_type,
            'training_entries': training_entries if request_type == "bulk_training" else 0,
            'model_type': None
        }


def analyze_stress_test_results(results):
    """Analyze and print stress test results with model type information."""
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
    
    request_types = defaultdict(int)
    for r in results:
        request_types[r.get('request_type', 'unknown')] += 1
    
    # Analyze model types in prediction responses
    model_types = defaultdict(int)
    for r in results:
        if r.get('model_type'):
            model_types[r['model_type']] += 1
    
    test_duration = max(response_times) if response_times else 0
    actual_qps = total_requests / test_duration if test_duration > 0 else 0
    
    print(f"\n{'='*50}")
    print("STRESS TEST RESULTS")
    print(f"{'='*50}")
    print(f"Total Requests: {total_requests}")
    print(f"Successful: {successful_requests} ({successful_requests/total_requests*100:.1f}%)")
    print(f"Failed: {failed_requests} ({failed_requests/total_requests*100:.1f}%)")
    print(f"Average Response Time: {avg_response_time*1000:.2f}ms")
    print(f"Actual QPS: {actual_qps:.0f}")
    print(f"\nRequest Types:")
    for req_type, count in request_types.items():
        print(f"  {req_type}: {count}")
    print(f"\nStatus Code Distribution:")
    for status, count in status_codes.items():
        print(f"  {status}: {count}")
    
    if model_types:
        print(f"\nModel Types in Predictions:")
        for model_type, count in model_types.items():
            print(f"  {model_type}: {count}")
    
    if response_times:
        sorted_times = sorted(response_times)
        p50 = sorted_times[int(len(sorted_times) * 0.5)] * 1000
        p95 = sorted_times[int(len(sorted_times) * 0.95)] * 1000
        p99 = sorted_times[int(len(sorted_times) * 0.99)] * 1000
        print(f"\nResponse Time Percentiles:")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")


def analyze_bulk_training_results(results):
    """Analyze and print bulk training stress test results with additional metrics."""
    if not results:
        print("No results to analyze")
        return
    
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r.get('success', False))
    failed_requests = total_requests - successful_requests
    
    # Separate analysis by request type
    prediction_results = [r for r in results if r.get('request_type') == 'predict']
    bulk_training_results = [r for r in results if r.get('request_type') == 'bulk_training']
    download_results = [r for r in results if r.get('request_type', '').startswith('download_')]
    
    # Calculate total training entries processed
    total_training_entries = sum(r.get('training_entries', 0) for r in bulk_training_results)
    
    # Analyze model types in prediction responses
    model_types = defaultdict(int)
    for r in prediction_results:
        if r.get('model_type'):
            model_types[r['model_type']] += 1
    
    response_times = [r['response_time'] for r in results if r.get('response_time')]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    status_codes = defaultdict(int)
    for r in results:
        status_codes[r.get('status_code', 0)] += 1
    
    request_types = defaultdict(int)
    for r in results:
        request_types[r.get('request_type', 'unknown')] += 1
    
    print(f"\n{'='*60}")
    print("BULK TRAINING STRESS TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Requests: {total_requests}")
    print(f"Successful: {successful_requests} ({successful_requests/total_requests*100:.1f}%)")
    print(f"Failed: {failed_requests} ({failed_requests/total_requests*100:.1f}%)")
    print(f"Average Response Time: {avg_response_time*1000:.2f}ms")
    
    print(f"\nRequest Type Breakdown:")
    print(f"  Prediction requests: {len(prediction_results)}")
    print(f"  Bulk training requests: {len(bulk_training_results)}")
    print(f"  Model download requests: {len(download_results)}")
    print(f"  Total training entries processed: {total_training_entries}")
    
    if model_types:
        print(f"\nModel Types in Predictions:")
        for model_type, count in model_types.items():
            print(f"  {model_type}: {count}")
    
    print(f"\nStatus Code Distribution:")
    for status, count in status_codes.items():
        print(f"  {status}: {count}")
    
    # Response time analysis by request type
    if prediction_results:
        pred_times = [r['response_time'] for r in prediction_results if r.get('response_time')]
        if pred_times:
            avg_pred_time = sum(pred_times) / len(pred_times)
            print(f"\nPrediction Request Response Times:")
            print(f"  Average: {avg_pred_time*1000:.2f}ms")
            print(f"  Min: {min(pred_times)*1000:.2f}ms")
            print(f"  Max: {max(pred_times)*1000:.2f}ms")
    
    if bulk_training_results:
        bulk_times = [r['response_time'] for r in bulk_training_results if r.get('response_time')]
        if bulk_times:
            avg_bulk_time = sum(bulk_times) / len(bulk_times)
            print(f"\nBulk Training Request Response Times:")
            print(f"  Average: {avg_bulk_time*1000:.2f}ms")
            print(f"  Min: {min(bulk_times)*1000:.2f}ms")
            print(f"  Max: {max(bulk_times)*1000:.2f}ms")
    
    if download_results:
        download_times = [r['response_time'] for r in download_results if r.get('response_time')]
        if download_times:
            avg_download_time = sum(download_times) / len(download_times)
            print(f"\nModel Download Request Response Times:")
            print(f"  Average: {avg_download_time*1000:.2f}ms")
            print(f"  Min: {min(download_times)*1000:.2f}ms")
            print(f"  Max: {max(download_times)*1000:.2f}ms")
    
    if response_times:
        sorted_times = sorted(response_times)
        p50 = sorted_times[int(len(sorted_times) * 0.5)] * 1000
        p95 = sorted_times[int(len(sorted_times) * 0.95)] * 1000
        p99 = sorted_times[int(len(sorted_times) * 0.99)] * 1000
        print(f"\nOverall Response Time Percentiles:")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")


def test_stress_test_high_qps():
    """
    Stress test with 300 QPS for 10 seconds.
    Sends predictions and training data in parallel.
    """
    results = asyncio.run(run_stress_test_async(duration_seconds=10, target_qps=300))
    
    analyze_stress_test_results(results)
    
    assert len(results) > 0, "No requests were made"
    
    successful_requests = sum(1 for r in results if r.get('success', False))
    success_rate = successful_requests / len(results)
    
    assert success_rate > 0.8, f"Success rate too low: {success_rate*100:.1f}%"
    
    print(f"Stress test completed successfully with {success_rate*100:.1f}% success rate")


def test_stress_test_mixed_load():
    """
    Alternative stress test with mixed load patterns.
    Tests server stability under varying load conditions.
    """
    print("Running mixed load stress test...")
    
    print("Phase 1: Ramping up load...")
    results_phase1 = asyncio.run(run_stress_test_async(duration_seconds=5, target_qps=100))
    
    print("Phase 2: High sustained load...")
    results_phase2 = asyncio.run(run_stress_test_async(duration_seconds=10, target_qps=300))
    
    print("Phase 3: Cooling down...")
    results_phase3 = asyncio.run(run_stress_test_async(duration_seconds=5, target_qps=50))
    
    all_results = results_phase1 + results_phase2 + results_phase3
    
    print("\nCOMBINED RESULTS FOR ALL PHASES:")
    analyze_stress_test_results(all_results)
    
    assert len(all_results) > 0, "No requests were made"
    
    successful_requests = sum(1 for r in all_results if r.get('success', False))
    success_rate = successful_requests / len(all_results)
    
    assert success_rate > 0.75, f"Overall success rate too low: {success_rate*100:.1f}%"
    
    print(f"Mixed load stress test completed with {success_rate*100:.1f}% success rate")


def test_simplified_stress_test():
    """Simplified stress test focusing on predictions, training, and tree downloads with prefix cache."""
    print("Running simplified stress test with prefix cache score support...")
    print("Configuration: 2 QPS, 50% bulk training, 35% predictions, 15% tree downloads (XGBoost only)")
    
    results = asyncio.run(run_simplified_stress_test(duration_seconds=60, target_qps=2))
    
    analyze_bulk_training_results(results)
    
    assert len(results) > 0, "No requests were made"
    
    successful_requests = sum(1 for r in results if r.get('success', False))
    success_rate = successful_requests / len(results)
    
    # Count request types
    prediction_count = sum(1 for r in results if r.get('request_type') == 'predict')
    bulk_training_count = sum(1 for r in results if r.get('request_type') == 'bulk_training')
    download_count = sum(1 for r in results if r.get('request_type', '').startswith('download_'))
    
    assert success_rate > 0.8, f"Success rate too low: {success_rate*100:.1f}%"
    assert prediction_count > 0, "No prediction requests were made"
    assert bulk_training_count > 0, "No bulk training requests were made"
    
    print(f"✓ Simplified stress test with prefix cache completed:")
    print(f"  Success rate: {success_rate*100:.1f}%")
    print(f"  Prediction requests: {prediction_count}")
    print(f"  Tree download requests: {download_count}")
    print(f"  Bulk training requests: {bulk_training_count}")


def test_model_type_consistency():
    """
    Test that the model type is consistent across all API endpoints.
    """
    print("Testing model type consistency across endpoints...")
    
    # Get model type from different endpoints
    root_response = requests.get(f"{BASE_URL}/")
    model_info_response = requests.get(f"{BASE_URL}/model/download/info")
    
    # Make a prediction to get model type from prediction response
    prediction_request = generate_random_prediction_payload()
    prediction_response = requests.post(f"{BASE_URL}/predict", json=prediction_request)
    
    # Extract model types
    root_model_type = root_response.json().get("model_type")
    model_info_model_type = model_info_response.json().get("model_type")
    prediction_model_type = prediction_response.json().get("model_type")
    
    # Check consistency
    assert root_model_type == model_info_model_type == prediction_model_type, (
        f"Model type inconsistency: root={root_model_type}, "
        f"model_info={model_info_model_type}, prediction={prediction_model_type}"
    )
    
    print(f"Model type consistent across all endpoints: {root_model_type}")


def test_xgboost_vs_bayesian_ridge_performance():
    """
    Performance comparison test (if both models are available).
    This test will check model performance differences.
    """
    model_info_r = requests.get(f"{BASE_URL}/model/download/info")
    model_info = model_info_r.json()
    
    print(f"Current model: {model_info['model_type']}")
    
    # Generate test predictions with prefix cache scores
    test_cases = [generate_random_prediction_payload() for _ in range(10)]
    
    predictions = []
    response_times = []
    
    for test_case in test_cases:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/predict", json=test_case)
        end_time = time.time()
        
        assert response.status_code == 200
        predictions.append(response.json())
        response_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_response_time = sum(response_times) / len(response_times)
    avg_prefix_cache = sum(tc['prefix_cache_score'] for tc in test_cases) / len(test_cases)
    
    print(f"Model: {predictions[0]['model_type']}")
    print(f"Average response time: {avg_response_time:.2f}ms")
    print(f"Average prefix cache score: {avg_prefix_cache:.2f}")
    print(f"Average TTFT prediction: {sum(p['ttft_ms'] for p in predictions)/len(predictions):.2f}ms")
    print(f"Average TPOT prediction: {sum(p['tpot_ms'] for p in predictions)/len(predictions):.2f}ms")
    print(f"Average TTFT uncertainty: {sum(p['ttft_uncertainty'] for p in predictions)/len(predictions):.2f}")
    print(f"Average TPOT uncertainty: {sum(p['tpot_uncertainty'] for p in predictions)/len(predictions):.2f}")
    
    # Basic sanity checks
    assert avg_response_time < 1000, f"Response time too slow: {avg_response_time:.2f}ms"
    assert all(p['ttft_ms'] > 0 for p in predictions), "All TTFT predictions should be positive"
    assert all(p['tpot_ms'] > 0 for p in predictions), "All TPOT predictions should be positive"


def test_uncertainty_estimation_quality():
    """
    Test the quality of uncertainty estimation for both model types.
    """
    model_info_r = requests.get(f"{BASE_URL}/model/download/info")
    model_type = model_info_r.json().get("model_type")
    
    # Generate multiple predictions for the same input
    test_payload = {
        "kv_cache_percentage": 0.5,
        "input_token_length": 100,
        "num_request_waiting": 2,
        "num_request_running": 1,
        "num_tokens_generated": 5,
        "prefix_cache_score": 0.8,  # Added prefix cache score
    }
    
    predictions = []
    for _ in range(5):  # Make multiple identical requests
        response = requests.post(f"{BASE_URL}/predict", json=test_payload)
        assert response.status_code == 200
        predictions.append(response.json())
    
    # Check that predictions are consistent (should be identical for same input)
    ttft_values = [p['ttft_ms'] for p in predictions]
    tpot_values = [p['tpot_ms'] for p in predictions]
    
    ttft_std = sum((x - ttft_values[0])**2 for x in ttft_values)**0.5 / len(ttft_values)
    tpot_std = sum((x - tpot_values[0])**2 for x in tpot_values)**0.5 / len(tpot_values)
    
    # For deterministic models, predictions should be identical
    if model_type == "bayesian_ridge":
        assert ttft_std < 0.01, f"TTFT predictions should be consistent, got std: {ttft_std}"
        assert tpot_std < 0.01, f"TPOT predictions should be consistent, got std: {tpot_std}"
    
    # Check uncertainty values are reasonable
    pred = predictions[0]
    ttft_uncertainty_ratio = pred['ttft_uncertainty'] / pred['ttft_ms']
    tpot_uncertainty_ratio = pred['tpot_uncertainty'] / pred['tpot_ms']
    
    print(f"Model: {model_type}")
    print(f"Prefix cache score: {test_payload['prefix_cache_score']}")
    print(f"TTFT: {pred['ttft_ms']:.2f} ± {pred['ttft_uncertainty']:.2f} ({ttft_uncertainty_ratio*100:.1f}%)")
    print(f"TPOT: {pred['tpot_ms']:.2f} ± {pred['tpot_uncertainty']:.2f} ({tpot_uncertainty_ratio*100:.1f}%)")
    
    # Uncertainty should be reasonable (not too high or too low)
    assert 0.01 < ttft_uncertainty_ratio < 0.5, f"TTFT uncertainty ratio should be reasonable: {ttft_uncertainty_ratio}"
    assert 0.01 < tpot_uncertainty_ratio < 0.5, f"TPOT uncertainty ratio should be reasonable: {tpot_uncertainty_ratio}"
    
    # Check prediction bounds contain the prediction
    ttft_bounds = pred['ttft_prediction_bounds']
    tpot_bounds = pred['tpot_prediction_bounds']
    
    assert ttft_bounds[0] <= pred['ttft_ms'] <= ttft_bounds[1], "TTFT should be within prediction bounds"
    assert tpot_bounds[0] <= pred['tpot_ms'] <= tpot_bounds[1], "TPOT should be within prediction bounds"


def test_edge_cases():
    """
    Test edge cases and boundary conditions with prefix cache score.
    """
    # Test minimum values
    min_payload = {
        "kv_cache_percentage": 0.0,
        "input_token_length": 1,
        "num_request_waiting": 0,
        "num_request_running": 0,
        "num_tokens_generated": 1,
        "prefix_cache_score": 0.0,  # Added prefix cache score
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=min_payload)
    assert response.status_code == 200
    data = response.json()
    assert data['ttft_ms'] > 0
    assert data['tpot_ms'] > 0
    
    # Test maximum reasonable values
    max_payload = {
        "kv_cache_percentage": 1.0,
        "input_token_length": 10000,
        "num_request_waiting": 100,
        "num_request_running": 50,
        "num_tokens_generated": 1000,
        "prefix_cache_score": 1.0,  # Added prefix cache score
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=max_payload)
    assert response.status_code == 200
    data = response.json()
    assert data['ttft_ms'] > 0
    assert data['tpot_ms'] > 0
    
    # Test invalid values (should fail validation)
    invalid_payloads = [
        {"kv_cache_percentage": -0.1, "input_token_length": 100, "num_request_waiting": 1, "num_request_running": 1, "num_tokens_generated": 10, "prefix_cache_score": 0.5},
        {"kv_cache_percentage": 1.1, "input_token_length": 100, "num_request_waiting": 1, "num_request_running": 1, "num_tokens_generated": 10, "prefix_cache_score": 0.5},
        {"kv_cache_percentage": 0.5, "input_token_length": -1, "num_request_waiting": 1, "num_request_running": 1, "num_tokens_generated": 10, "prefix_cache_score": 0.5},
        {"kv_cache_percentage": 0.5, "input_token_length": 100, "num_request_waiting": -1, "num_request_running": 1, "num_tokens_generated": 10, "prefix_cache_score": 0.5},
        {"kv_cache_percentage": 0.5, "input_token_length": 100, "num_request_waiting": 1, "num_request_running": -1, "num_tokens_generated": 10, "prefix_cache_score": 0.5},
        {"kv_cache_percentage": 0.5, "input_token_length": 100, "num_request_waiting": 1, "num_request_running": 1, "num_tokens_generated": -1, "prefix_cache_score": 0.5},
        {"kv_cache_percentage": 0.5, "input_token_length": 100, "num_request_waiting": 1, "num_request_running": 1, "num_tokens_generated": 10, "prefix_cache_score": -0.1},  # Invalid prefix cache
        {"kv_cache_percentage": 0.5, "input_token_length": 100, "num_request_waiting": 1, "num_request_running": 1, "num_tokens_generated": 10, "prefix_cache_score": 1.1},   # Invalid prefix cache
    ]
    
    for invalid_payload in invalid_payloads:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_payload)
        assert response.status_code == 422, f"Should reject invalid payload: {invalid_payload}"


def test_concurrent_training_and_prediction():
    """
    Test that training and prediction can happen concurrently without issues.
    """
    print("Testing concurrent training and prediction with prefix cache...")
    
    def make_predictions():
        results = []
        for _ in range(20):
            payload = generate_random_prediction_payload()
            try:
                response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
                results.append(response.status_code == 200)
            except:
                results.append(False)
            time.sleep(0.1)
        return results
    
    def send_training_data():
        results = []
        for _ in range(5):
            payload = generate_bulk_training_payload(100)  # Smaller batches for faster processing
            try:
                response = requests.post(f"{BASE_URL}/add_training_data_bulk", json=payload, timeout=10)
                results.append(response.status_code == 202)
            except:
                results.append(False)
            time.sleep(0.5)
        return results
    
    # Run both functions concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        prediction_future = executor.submit(make_predictions)
        training_future = executor.submit(send_training_data)
        
        prediction_results = prediction_future.result()
        training_results = training_future.result()
    
    prediction_success_rate = sum(prediction_results) / len(prediction_results)
    training_success_rate = sum(training_results) / len(training_results)
    
    print(f"Prediction success rate: {prediction_success_rate*100:.1f}%")