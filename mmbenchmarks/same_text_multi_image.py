#!/usr/bin/env python3

# Copyright 2026 The Kubernetes Authors.
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

import time
import requests
import json
import argparse

# --- Configuration ---
DEFAULT_IP = "35.208.170.236"
DEFAULT_PORT = "80"
DEFAULT_MODEL = "qwen/Qwen2.5-VL-7B-Instruct"

# --- Hardcoded Image URLs ---
IMAGE_URLS = {
    "https://cdn2.thecatapi.com/images/dui.jpg": 12,
    "https://cdn2.thecatapi.com/images/ebv.jpg": 12,
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg": 12,
    "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg": 3,
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg": 3
}

QUESTION = "Describe this image in one sentence."

def run_benchmark(ip, port, model, enable_request_stats=False):
    url_endpoint = f"http://{ip}:{port}/v1/chat/completions"
    start_time = time.time()

    for url, count in IMAGE_URLS.items():
        for i in range(count):
            print("-" * 42)
            print(f"Processing: {url} (Call {i+1}/{count})")
            print(f"Question: {QUESTION}")

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": QUESTION},
                            {"type": "image_url", "image_url": {"url": url}}
                        ]
                    }
                ]
            }

            if enable_request_stats:
                payload["include_metrics"] = True

            try:
                response = requests.post(url_endpoint, json=payload, headers={"Content-Type": "application/json"})
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(json.dumps(data, indent=2))
            except Exception as e:
                print(f"Error: {e}")

            print("\n")

    end_time = time.time()
    duration = end_time - start_time
    print("-" * 35)
    print(f"Query completed in: {duration:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMBenchmark Python Client (Same Text, Multi URL)")
    parser.add_argument("--ip", type=str, default=DEFAULT_IP, help="IP address of the endpoint")
    parser.add_argument("--port", type=str, default=DEFAULT_PORT, help="Port of the endpoint")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name to use")
    parser.add_argument("--enable-request-stats", action="store_true", help="Enable per-request metrics (requires vLLM --enable-per-request-metrics)")
    args = parser.parse_args()

    run_benchmark(args.ip, args.port, args.model, args.enable_request_stats)
