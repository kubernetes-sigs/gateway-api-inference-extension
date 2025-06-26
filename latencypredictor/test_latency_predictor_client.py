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

# Base URL of your running FastAPI server
BASE_URL = os.getenv("LATENCY_SERVER_URL", "http://34.168.179.22:80")

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


def test_add_training_data_bulk():
    """
    Send 120 training samples in one bulk request so the server can retrain:
      actual_ttft_ms = 2*input_token_length + 3*num_request_waiting +
                       4*num_request_running + 50*kv_cache_percentage + 95
      actual_tpot_ms = 100*kv_cache_percentage + 1*num_tokens_generated +
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
        entries.append({
            "kv_cache_percentage": kv,
            "input_token_length": inp_len,
            "num_request_waiting": waiting,
            "num_request_running": running,
            "actual_ttft_ms": (inp_len*2.0 + waiting*3.0 + running*4.0 + kv*50.0) + 95,
            "actual_tpot_ms": (kv*100.0 + tokens*1.0 + running*5.0) + 9,
            "num_tokens_generated": tokens,
            "timestamp": time.time()  # FastAPI will coerce to datetime
        })

    payload = {"entries": entries}
    r = requests.post(f"{BASE_URL}/add_training_data_bulk", json=payload)
    assert r.status_code == 202, f"Expected 202, got {r.status_code}"
    assert r.json().get("message") == "Accepted 120 training samples."


def test_model_learns_equation():
    """
    After sending bulk data, poll /predict until the model's predictions
    match our linear equations within ±10%, or fail after 60s.
    """
    features = {
        "kv_cache_percentage": 0.5,
        "input_token_length": 200,
        "num_request_waiting": 4,
        "num_request_running": 1,
        "num_tokens_generated": 4,
    }
    expected_ttft = (
        features["input_token_length"] * 2.0
        + features["num_request_waiting"] * 3.0
        + features["num_request_running"] * 4.0
        + features["kv_cache_percentage"] * 50.0 + 95
    )
    expected_tpot = (
        features["kv_cache_percentage"] * 100.0
        + features["num_tokens_generated"] * 1.0
        + features["num_request_running"] * 5.0 + 9
    )

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

        ttft_ok = abs(last_ttft - expected_ttft) <= 0.1 * expected_ttft
        tpot_ok = abs(last_tpot - expected_tpot) <= 0.1 * expected_tpot
        if ttft_ok and tpot_ok:
            break

        time.sleep(1)

    assert last_ttft is not None, "Never got a successful prediction."
    assert abs(last_ttft - expected_ttft) <= 0.1 * expected_ttft, (
        f"TTFT={last_ttft:.1f} not within ±10% of {expected_ttft:.1f}"
    )
    assert abs(last_tpot - expected_tpot) <= 0.1 * expected_tpot, (
        f"TPOT={last_tpot:.1f} not within ±10% of {expected_tpot:.1f}"
    )


def generate_random_prediction_payload():
    """Generate a random prediction payload for stress testing including new feature."""
    return {
        "kv_cache_percentage": random.uniform(0.1, 0.9),
        "input_token_length": random.randint(10, 1000),
        "num_request_waiting": random.randint(1, 20),
        "num_request_running": random.randint(1, 10),
        "num_tokens_generated": random.randint(1, 20),
    }


def generate_random_training_payload():
    """Generate a random training data payload for stress testing."""
    input_tokens = random.randint(10, 1000)
    waiting_requests = random.randint(1, 20)
    running_requests = random.randint(1, 10)
    kv = random.uniform(0.01, 0.99)
    
    return {
        "kv_cache_percentage": kv,
        "input_token_length": input_tokens,
        "num_request_waiting": waiting_requests,
        "num_request_running": running_requests,
        # linear TTFT with noise
        "actual_ttft_ms": (
            input_tokens * 2.0
            + waiting_requests * 3.0
            + running_requests * 4.0
            + kv * 50.0
            + 95 + random.uniform(-10, 10)
        ),
        # linear TPOT with noise
        "actual_tpot_ms": (
            kv * 100.0
            + waiting_requests * 1.0
            + running_requests * 5.0
            + 5 + random.uniform(-5, 5)
        ),
        "num_tokens_generated": waiting_requests,
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
                'request_type': 'predict' if '/predict' in url else 'training'
            }
    except Exception as e:
        end_time = time.time()
        return {
            'request_id': request_id,
            'status_code': 0,
            'response_time': end_time - start_time,
            'success': False,
            'error': str(e),
            'request_type': 'predict' if '/predict' in url else 'training'
        }

async def run_stress_test_async(duration_seconds=10, target_qps=1000):
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


async def run_bulk_training_stress_test(duration_seconds=10, target_qps=2):
    """
    Stress test with bulk training (1000 entries) and individual predictions at 50-50 split.
    Sends requests at specified QPS.
    """
    interval = 1.0 / target_qps
    start = time.time()
    connector = aiohttp.TCPConnector(limit=1000, limit_per_host=1000, ttl_dns_cache=300, use_dns_cache=True)
    
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=30)) as sess:
        tasks = []
        req_id = 0
        next_time = start
        
        while time.time() - start < duration_seconds:
            now = time.time()
            while next_time <= now:
                req_id += 1
                if random.random() < 0.5:
                    # Send individual prediction request
                    url = f"{BASE_URL}/predict"
                    payload = generate_random_prediction_payload()
                    request_type = "predict"
                else:
                    # Send bulk training request with 1000 entries
                    url = f"{BASE_URL}/add_training_data_bulk"
                    payload = generate_bulk_training_payload(1000)
                    request_type = "bulk_training"
                
                # Create task with extended timeout for bulk requests
                timeout = aiohttp.ClientTimeout(total=30 if request_type == "bulk_training" else 5)
                task = asyncio.create_task(
                    async_post_request_with_timeout(sess, url, payload, req_id, timeout, request_type)
                )
                tasks.append(task)
                next_time += interval
            
            await asyncio.sleep(0.001)  # Small sleep to prevent tight loop

        print(f"Waiting for {len(tasks)} requests to complete...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = [r for r in results if isinstance(r, dict)]
    
        # Calculate actual QPS achieved
        if valid_results:
            actual_duration = duration_seconds
            actual_qps = len(valid_results) / actual_duration
            print(f"Target QPS: {target_qps}, Actual QPS: {actual_qps:.2f}")
    
        return valid_results


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
                'training_entries': training_entries if request_type == "bulk_training" else 0
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
            'training_entries': training_entries if request_type == "bulk_training" else 0
        }


def analyze_stress_test_results(results):
    """Analyze and print stress test results."""
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
    
    # Calculate total training entries processed
    total_training_entries = sum(r.get('training_entries', 0) for r in bulk_training_results)
    
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
    print(f"  Total training entries processed: {total_training_entries}")
    
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
    
    if response_times:
        sorted_times = sorted(response_times)
        p50 = sorted_times[int(len(sorted_times) * 0.5)] * 1000
        p95 = sorted_times[int(len(sorted_times) * 0.95)] * 1000
        p99 = sorted_times[int(len(sorted_times) * 0.99)] * 1000
        print(f"\nOverall Response Time Percentiles:")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")


def test_stress_test_1k_qps():
    """
    Stress test with 40k QPS for 10 seconds.
    Sends predictions and training data in parallel.
    """
    results = asyncio.run(run_stress_test_async(duration_seconds=10, target_qps=1000))
    
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
    results_phase1 = asyncio.run(run_stress_test_async(duration_seconds=5, target_qps=800))
    
    print("Phase 2: High sustained load...")
    results_phase2 = asyncio.run(run_stress_test_async(duration_seconds=10, target_qps=1000))
    
    print("Phase 3: Cooling down...")
    results_phase3 = asyncio.run(run_stress_test_async(duration_seconds=5, target_qps=500))
    
    all_results = results_phase1 + results_phase2 + results_phase3
    
    print("\nCOMBINED RESULTS FOR ALL PHASES:")
    analyze_stress_test_results(all_results)
    
    assert len(all_results) > 0, "No requests were made"
    
    successful_requests = sum(1 for r in all_results if r.get('success', False))
    success_rate = successful_requests / len(all_results)
    
    assert success_rate > 0.75, f"Overall success rate too low: {success_rate*100:.1f}%"
    
    print(f"Mixed load stress test completed with {success_rate*100:.1f}% success rate")


def test_bulk_training_stress_test():
    """
    New stress test with bulk training (1000 entries per request) and predictions.
    Sends 50-50 split of bulk training and prediction requests at 2 QPS for 30 seconds.
    """
    print("Running bulk training stress test...")
    print("Configuration: 2 QPS, 50% bulk training (1000 entries), 50% predictions, 1000 seconds")
    
    results = asyncio.run(run_bulk_training_stress_test(duration_seconds=300, target_qps=2))
    
    analyze_bulk_training_results(results)
    
    assert len(results) > 0, "No requests were made"
    
    successful_requests = sum(1 for r in results if r.get('success', False))
    success_rate = successful_requests / len(results)
    
    # Count training vs prediction requests
    prediction_count = sum(1 for r in results if r.get('request_type') == 'predict')
    bulk_training_count = sum(1 for r in results if r.get('request_type') == 'bulk_training')
    total_training_entries = sum(r.get('training_entries', 0) for r in results if r.get('request_type') == 'bulk_training')
    
    # Assertions
    assert success_rate > 0.7, f"Success rate too low: {success_rate*100:.1f}%"
    assert prediction_count > 0, "No prediction requests were made"
    assert bulk_training_count > 0, "No bulk training requests were made"
    assert total_training_entries >= bulk_training_count * 1000, "Bulk requests should contain 1000 entries each"
    
    print(f"\nBulk training stress test completed successfully:")
    print(f"  Success rate: {success_rate*100:.1f}%")
    print(f"  Prediction requests: {prediction_count}")
    print(f"  Bulk training requests: {bulk_training_count}")
    print(f"  Total training entries processed: {total_training_entries}")


if __name__ == "__main__":
    print("Running stress tests directly...")
    test_bulk_training_stress_test()