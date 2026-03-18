# Latency Predictor Load Tests

Load tests for the latency predictor sidecar system. These tests exercise the Go coalescer client against the prediction and training servers at various QPS levels.

## Components

- `main.go` — Go test client that generates prediction and training requests at a configurable QPS
- `Dockerfile` — Builds the test client image
- `run_qps_sweep.sh` — Runs an automated QPS sweep (1 to 10,000 QPS)
- `manifests/predictor_warmup.yaml` — K8s Job to warm up training buckets before running load tests

## Prerequisites

1. A running EPP deployment with the latency predictor sidecar (training + prediction servers)
2. The prediction server ports must be accessible from the test pod (e.g., via a K8s Service)

## Building the Test Image

From the repository root:

```bash
docker build -t <your-registry>/predictor-client-test:latest \
  -f sidecars/latencypredictorasync/tests/Dockerfile .
docker push <your-registry>/predictor-client-test:latest
```

## Running the Load Tests

### Step 1: Warm Up Training Buckets

The prediction servers need trained models before they can serve predictions. Run the warmup job to fill training buckets with synthetic data:

```bash
# Edit the manifest to point to your EPP service and image
kubectl apply -f sidecars/latencypredictorasync/tests/manifests/predictor_warmup.yaml

# Wait for warmup to complete (~5 minutes)
kubectl wait --for=condition=complete job/predictor-warmup --timeout=600s

# Verify training server has data
kubectl exec <epp-pod> -c training-server-1 -- \
  python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/metrics').read().decode())" \
  | grep training_samples_count
```

### Step 2: Run a Single Load Test

Deploy a test job with desired parameters:

```bash
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: predictor-load-test
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 600
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: predictor-client-test
        image: <your-registry>/predictor-client-test:latest
        env:
        - name: PREDICTION_SERVERS
          value: "http://<epp-service>:8001"
        - name: TRAINING_SERVER_URL
          value: "http://<epp-service>:8000"
        - name: TEST_QPS
          value: "100"
        - name: TEST_DURATION_SECONDS
          value: "120"
        - name: NUM_ENDPOINTS_PER_REQUEST
          value: "10"
        - name: MAX_CONCURRENT_DISPATCHES
          value: "36"
        resources:
          requests:
            cpu: "1000m"
            memory: "1Gi"
          limits:
            cpu: "4000m"
            memory: "8Gi"
EOF
```

### Step 3: Run a QPS Sweep

The sweep script runs tests at 1, 10, 100, 1000, 2500, and 5000 QPS (300s each):

```bash
./sidecars/latencypredictorasync/tests/run_qps_sweep.sh \
  --pred-base-url http://<epp-service> \
  --pred-start-port 8001 \
  --train-url http://<epp-service>:8000
```

Results are written to `/tmp/qps_results.txt`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PREDICTION_SERVERS` | (required) | Comma-separated prediction server URLs |
| `TRAINING_SERVER_URL` | `http://training-service:8000` | Training server URL |
| `TEST_QPS` | `100` | Target queries per second |
| `TEST_DURATION_SECONDS` | `120` | Test duration in seconds |
| `MAX_BULK_SIZE` | `200` | Max predictions per bulk request |
| `NUM_ENDPOINTS_PER_REQUEST` | `10` | Number of endpoints per prediction request |
| `TRAINING_BATCH_SIZE` | `100` | Training entries per batch |
| `TRAINING_INTERVAL_MS` | `500` | Interval between training batches (ms) |
| `MAX_CONCURRENT_DISPATCHES` | `8` | Max concurrent HTTP dispatches to prediction servers |
| `MAX_INFLIGHT_REQUESTS` | `QPS/5+500` | Max concurrent prediction goroutines |
| `COALESCE_WINDOW_MS` | `5` | Coalesce window for batching requests |
| `MAX_COALESCED_CALLERS` | `50` | Max callers coalesced into a single batch |

## Sizing Guide

| QPS Range | Sidecars | MAX_CONCURRENT_DISPATCHES | MAX_INFLIGHT_REQUESTS |
|-----------|----------|--------------------------|----------------------|
| 1-2,500 | 1 | 36 | 215 |
| 2,501-5,000 | 2 | 64 | 430 |
| 5,001-7,500 | 3 | 92 | 640 |
| 7,501-10,000 | 4 | 120 | 850 |
