#!/usr/bin/env python3
"""
Quick setup script to send initial training data and trigger model training.
Run this before stress tests to ensure models are trained.
"""
import os
import time
import requests
import numpy as np

TRAINING_URL = os.getenv("TRAINING_SERVER_URL", "http://training-service:8000")
PREDICTION_URL = os.getenv("PREDICTION_SERVER_URL", "http://prediction-service:80")

def send_training_data(num_samples=1000):
    """Send training data to training server."""
    print(f"Sending {num_samples} training samples...")

    np.random.seed(42)
    entries = []

    for i in range(num_samples):
        kv = np.random.uniform(0.1, 0.9)
        input_len = np.random.randint(50, 800)
        waiting = np.random.randint(0, 8)
        running = np.random.randint(1, 4)
        tokens_gen = np.random.randint(1, 25)
        prefix = np.random.uniform(0.0, 1.0)

        # Generate with known pattern + noise
        ttft_mu = (input_len*2.0 + waiting*3.0 + running*4.0 + kv*50.0 + prefix*30.0 + 95)
        tpot_mu = (kv*100.0 + input_len*0.5 + tokens_gen*1.0 + running*5.0 + 9)

        entries.append({
            "kv_cache_percentage": float(kv),
            "input_token_length": int(input_len),
            "num_request_waiting": int(waiting),
            "num_request_running": int(running),
            "actual_ttft_ms": float(max(1.0, ttft_mu + np.random.normal(0, 20))),
            "actual_tpot_ms": float(max(1.0, tpot_mu + np.random.normal(0, 10))),
            "num_tokens_generated": int(tokens_gen),
            "prefix_cache_score": float(prefix),
        })

    response = requests.post(f"{TRAINING_URL}/add_training_data_bulk",
                            json={"entries": entries},
                            timeout=60)

    if response.status_code == 202:
        print(f"✓ Successfully sent {num_samples} training samples")
        return True
    else:
        print(f"✗ Failed to send training data: {response.status_code}")
        return False

def wait_for_training(max_wait=60):
    """Wait for training to complete."""
    print(f"Waiting for training to complete (max {max_wait}s)...")
    time.sleep(30)  # Initial wait for training

    for i in range(max_wait // 5):
        try:
            response = requests.post(f"{PREDICTION_URL}/reload", timeout=20)
            if response.status_code == 200:
                data = response.json()
                if data.get("is_ready"):
                    print("✓ Models trained and loaded")
                    return True
        except Exception as e:
            print(f"Waiting... ({i*5}s)")
        time.sleep(5)

    print("✗ Models not ready after timeout")
    return False

if __name__ == "__main__":
    print("="*50)
    print("Setting up training data for stress tests")
    print("="*50)

    if send_training_data(1000):
        if wait_for_training(60):
            print("\n✓ Setup complete! Ready for stress testing.")
            exit(0)

    print("\n✗ Setup failed")
    exit(1)
