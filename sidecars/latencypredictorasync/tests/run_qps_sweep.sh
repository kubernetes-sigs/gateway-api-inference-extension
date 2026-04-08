#!/bin/bash
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

# QPS sweep test: runs 1, 10, 100, 1000, 2500, 5000, 7500, 10000 QPS
# Each test runs for 300s. No warmup needed (training server already full).
# Results written to /tmp/qps_results.txt
#
# Usage:
#   ./run_qps_sweep.sh [OPTIONS]
#
# Options:
#   --pred-base-url    Base URL for prediction servers (port appended per sidecar)
#                      Default: http://vllm-qwen3-32b-epp
#   --pred-start-port  Port for the first prediction server; incremented per sidecar
#                      Default: 8001
#   --train-url        Full URL for the training server
#                      Default: http://vllm-qwen3-32b-epp:8000
#
# Examples:
#   # Default (EPP sidecar mode):
#   ./run_qps_sweep.sh
#
#   # Standalone dual-server deployment (prediction-service serves on port 80):
#   ./run_qps_sweep.sh \
#     --pred-base-url http://prediction-service \
#     --pred-start-port 80 \
#     --train-url http://training-service:8000

set -e

# Defaults
PRED_BASE_URL="http://vllm-qwen3-32b-epp"
PRED_START_PORT=8001
TRAIN_URL="http://vllm-qwen3-32b-epp:8000"

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pred-base-url)
      PRED_BASE_URL="$2"; shift 2 ;;
    --pred-start-port)
      PRED_START_PORT="$2"; shift 2 ;;
    --train-url)
      TRAIN_URL="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--pred-base-url URL] [--pred-start-port PORT] [--train-url URL]"
      exit 1 ;;
  esac
done

IMAGE="${LOAD_TEST_IMAGE:-us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/latency-predictor-load-test:latest}"
RESULTS_FILE=/tmp/qps_results.txt
echo "QPS Sweep Results - $(date)" > $RESULTS_FILE
echo "  pred-base-url:    ${PRED_BASE_URL}" >> $RESULTS_FILE
echo "  pred-start-port:  ${PRED_START_PORT}" >> $RESULTS_FILE
echo "  train-url:        ${TRAIN_URL}" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE

run_test() {
  local QPS=$1 DISP=$2 INF=$3 SIDECARS=$4
  echo ""
  echo ">>> Running QPS=$QPS  sidecars=$SIDECARS  dispatches=$DISP  inflight=$INF"

  # Build PREDICTION_SERVERS from sidecar count
  local PRED_SERVERS=""
  for i in $(seq 1 $SIDECARS); do
    local PORT=$(( PRED_START_PORT + i - 1 ))
    if [ -z "$PRED_SERVERS" ]; then
      PRED_SERVERS="${PRED_BASE_URL}:${PORT}"
    else
      PRED_SERVERS="${PRED_SERVERS},${PRED_BASE_URL}:${PORT}"
    fi
  done

  kubectl delete job predictor-client-test-sweep --ignore-not-found 2>/dev/null
  sleep 2

  cat <<YAML | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: predictor-client-test-sweep
  namespace: default
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 600
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: predictor-client-test
        image: ${IMAGE}
        imagePullPolicy: Always
        env:
        - name: PREDICTION_SERVERS
          value: "${PRED_SERVERS}"
        - name: TRAINING_SERVER_URL
          value: "${TRAIN_URL}"
        - name: TEST_QPS
          value: "${QPS}"
        - name: TEST_DURATION_SECONDS
          value: "300"
        - name: MAX_BULK_SIZE
          value: "1000"
        - name: NUM_ENDPOINTS_PER_REQUEST
          value: "100"
        - name: MAX_CONCURRENT_DISPATCHES
          value: "${DISP}"
        - name: MAX_INFLIGHT_REQUESTS
          value: "${INF}"
        - name: COALESCE_WINDOW_MS
          value: "1"
        - name: MAX_COALESCED_CALLERS
          value: "10"
        - name: TRAINING_BATCH_SIZE
          value: "1000"
        - name: TRAINING_INTERVAL_MS
          value: "500"
        resources:
          requests:
            cpu: "1000m"
            memory: "1Gi"
          limits:
            cpu: "4000m"
            memory: "8Gi"
YAML

  echo "  Waiting for job to complete..."
  kubectl wait --for=condition=complete job/predictor-client-test-sweep --timeout=480s

  PRED=$(kubectl logs job/predictor-client-test-sweep 2>/dev/null | grep "TEST RESULTS - Predictions")
  echo "QPS_TARGET=${QPS} | ${PRED}" | tee -a $RESULTS_FILE

  kubectl delete job predictor-client-test-sweep --ignore-not-found 2>/dev/null
  sleep 3
}

#       QPS    dispatches  inflight  sidecars
run_test 1      36          5         1
run_test 10     36          5         1
run_test 100    36          20        1
run_test 1000   36          85        1
run_test 2500   36          215       1
run_test 5000   36          430       1
#run_test 5000   36          430       2
#run_test 7500   92          640       3
#run_test 10000  120         850       4

echo ""
echo "========================================" >> $RESULTS_FILE
echo "SWEEP COMPLETE" >> $RESULTS_FILE
echo ""
echo "=== FINAL RESULTS ==="
cat $RESULTS_FILE
