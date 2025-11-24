# SLO-Aware Routing with Latency Prediction

This document describes the modifications made to the InferencePool Helm chart to support SLO-aware routing with latency prediction sidecars.

## Overview

The SLO-aware routing feature enables intelligent request routing based on predicted latency using machine learning models. The system consists of:

1. **EPP (Endpoint Picker) Container**: Main routing logic with latency prediction enabled
2. **Training Server Sidecar**: Continuously trains XGBoost models on observed latency metrics
3. **Prediction Server Sidecars**: Multiple replicas that serve latency predictions for TTFT (Time to First Token) and TPOT (Time Per Output Token)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    EPP Pod                           │
├──────────────┬──────────────┬──────────────────────┤
│     EPP      │   Training   │  Prediction Servers  │
│  Container   │    Server    │  (3 replicas)        │
│              │              │                       │
│  Port 9002   │  Port 8000   │  Ports 8001-8003     │
│  (ext-proc)  │  (training)  │  (prediction)        │
└──────────────┴──────────────┴──────────────────────┘
       │              │                  │
       │              └──────┬───────────┘
       │                     │
       │              Model Training
       │              & Synchronization
       │
    Routing Decision
    (with latency prediction)
```

## Modified Files

### 1. `templates/epp-deployment.yaml`
- Added support for `sidecars.trainingServer` configuration
- Added support for `sidecars.predictionServers` with configurable replicas
- Automatically creates volumes for model storage
- Injects ConfigMaps for training and prediction server configuration

### 2. `templates/epp-service.yaml`
- Automatically exposes ports for training server (8000)
- Automatically exposes ports for prediction servers (8001-8003 by default)
- Ports are only added when sidecars are enabled

### 3. `templates/latency-predictor-config.yaml` (NEW)
- Creates ConfigMap for training server configuration
- Creates ConfigMap for prediction server configuration
- Supports customizable model paths, retraining intervals, and other parameters

### 4. `values.yaml`
- Added comprehensive `sidecars` section with commented examples
- Supports configuration for training and prediction server images, resources, and behavior

### 5. `values-slo-example.yaml` (NEW)
- Complete working example of SLO-aware routing configuration
- Demonstrates all required settings including EPP flags, environment variables, and plugin configuration

## Usage

### Quick Start with Example Configuration

```bash
# Install with SLO-aware routing enabled
helm install my-slo-pool oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool \
  --namespace inference \
  --values values-slo-example.yaml \
  --set inferencePool.modelServers.matchLabels.app=my-model-server
```

### Custom Configuration

Create a custom values file:

```yaml
inferenceExtension:
  image:
    hub: quay.io/your-org
    name: epp
    tag: slo-experimental

  flags:
    - name: enable-latency-predictor
      value: "true"
    - name: v
      value: "4"

  env:
    - name: PREDICTION_SERVER_URL
      value: "http://localhost:8001,http://localhost:8002,http://localhost:8003"
    - name: TRAINING_SERVER_URL
      value: "http://localhost:8000"
    - name: LATENCY_MAX_SAMPLE_SIZE
      value: "10000"

  pluginsCustomConfig:
    slo-plugins.yaml: |
      apiVersion: inference.networking.x-k8s.io/v1alpha1
      kind: EndpointPickerConfig
      plugins:
      - type: slo-request-tracker
      - type: slo-scorer
      - type: slo-aware-profile-handler
      schedulingProfiles:
      - name: slo
        plugins:
        - pluginRef: slo-request-tracker
        - pluginRef: slo-scorer

  sidecars:
    trainingServer:
      enabled: true
      image:
        hub: quay.io/your-org
        name: latency-training
        tag: latest
      resources:
        requests:
          cpu: "2000m"
          memory: "4Gi"
        limits:
          cpu: "4000m"
          memory: "8Gi"

    predictionServers:
      enabled: true
      replicas: 3
      image:
        hub: quay.io/your-org
        name: latency-prediction
        tag: latest
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
        limits:
          cpu: "1000m"
          memory: "2Gi"
```

## Configuration Reference

### Training Server Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sidecars.trainingServer.enabled` | Enable training server sidecar | `false` |
| `sidecars.trainingServer.image.hub` | Container registry | `us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension` |
| `sidecars.trainingServer.image.name` | Image name | `latency-training` |
| `sidecars.trainingServer.image.tag` | Image tag | `latest` |
| `sidecars.trainingServer.config.retrainingIntervalSec` | Retraining interval in seconds | `1` |
| `sidecars.trainingServer.config.minSamplesForRetrain` | Minimum samples before retraining | `100` |
| `sidecars.trainingServer.config.modelType` | ML model type | `xgboost` |
| `sidecars.trainingServer.persistence.enabled` | Enable persistent storage for models | `false` |

### Prediction Server Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sidecars.predictionServers.enabled` | Enable prediction server sidecars | `false` |
| `sidecars.predictionServers.replicas` | Number of prediction server replicas | `3` |
| `sidecars.predictionServers.image.hub` | Container registry | `us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension` |
| `sidecars.predictionServers.image.name` | Image name | `latency-prediction` |
| `sidecars.predictionServers.image.tag` | Image tag | `latest` |
| `sidecars.predictionServers.config.modelSyncIntervalSec` | Model sync interval in seconds | `10` |
| `sidecars.predictionServers.config.modelType` | ML model type | `xgboost` |

### EPP Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PREDICTION_SERVER_URL` | Comma-separated prediction server URLs | `http://localhost:8001,http://localhost:8002,http://localhost:8003` |
| `TRAINING_SERVER_URL` | Training server URL | `http://localhost:8000` |
| `LATENCY_MAX_SAMPLE_SIZE` | Maximum sample size for latency prediction | `10000` |
| `NEG_HEADROOM_TPOT_WEIGHT` | Weight for TPOT in negative headroom calculation | `0.2` |
| `NEG_HEADROOM_TTFT_WEIGHT` | Weight for TTFT in negative headroom calculation | `0.8` |

## Building Container Images

### Prerequisites

```bash
cd /path/to/gateway-api-inference-extension
git checkout slo-prediction-experimental
```

### Build EPP Image

```bash
export IMAGE_REGISTRY="quay.io/your-org"
export EPP_TAG="slo-experimental"
make image-build image-push
```

### Build Latency Predictor Images

```bash
cd latencypredictor-v1

# Edit build-deploy.sh to set your registry
# Then build and push:
./build-deploy.sh build

# Tag and push manually
docker tag latencypredictor-v2-training-server:latest ${IMAGE_REGISTRY}/latency-training:slo-experimental
docker tag latencypredictor-v2-prediction-server:latest ${IMAGE_REGISTRY}/latency-prediction:slo-experimental
docker push ${IMAGE_REGISTRY}/latency-training:slo-experimental
docker push ${IMAGE_REGISTRY}/latency-prediction:slo-experimental
```

## Verification

After deployment, verify all containers are running:

```bash
# Check pod status
kubectl get pods -n your-namespace

# Expected: 1 pod with 5 containers (1 EPP + 1 training + 3 prediction)

# Check EPP logs
kubectl logs -n your-namespace <pod-name> -c epp

# Check training server logs
kubectl logs -n your-namespace <pod-name> -c training-server

# Check prediction server logs
kubectl logs -n your-namespace <pod-name> -c prediction-server-1
```

## Service Ports

When sidecars are enabled, the service automatically exposes these ports:

- `9002`: EPP gRPC ext-proc (always)
- `9090`: EPP metrics (always)
- `8000`: Training server (when `trainingServer.enabled: true`)
- `8001-800N`: Prediction servers (when `predictionServers.enabled: true`, N = replicas)

## Plugins

The SLO-aware routing requires these plugins:

- `slo-request-tracker`: Tracks request SLO requirements
- `slo-scorer`: Scores endpoints based on predicted latency vs SLO
- `slo-aware-profile-handler`: Handles different scheduling profiles
- `max-score-picker`: Selects endpoint with maximum score

### Scheduling Profiles

- **default**: Standard routing with queue and kv-cache scoring
- **slo**: SLO-aware routing using latency predictions

## Troubleshooting

### Sidecars Not Starting

Check if images are accessible:
```bash
kubectl describe pod <pod-name> -n your-namespace
```

### Training Server Issues

Check ConfigMap and logs:
```bash
kubectl get configmap latency-predictor-config -n your-namespace -o yaml
kubectl logs <pod-name> -c training-server -n your-namespace
```

### Prediction Server Issues

Verify prediction servers can reach training server:
```bash
kubectl exec <pod-name> -c prediction-server-1 -n your-namespace -- \
  curl http://localhost:8000/healthz
```

## Integration with llm-d

To use this chart in llm-d, update your helmfile:

```yaml
releases:
  - name: gaie-slo
    namespace: llm-d-slo
    chart: oci://quay.io/your-org/charts/inferencepool
    version: v1.0.1-slo
    values:
      - gaie-slo/values.yaml
      - gaie-slo/values-slo.yaml
```

See the main documentation for complete integration instructions.
