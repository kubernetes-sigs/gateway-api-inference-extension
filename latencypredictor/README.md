# Latency Predictor - Build Guide

This directory contains the Latency Predictor component with dual server architecture (training and prediction servers). Use the provided `build-deploy.sh` script to build and deploy container images to Google Cloud Platform.

## Prerequisites

- Docker (latest version)
- Google Cloud SDK (`gcloud`) configured and authenticated
- Required files in directory:
  - `training_server.py`
  - `prediction_server.py` 
  - `requirements.txt`
  - `Dockerfile-training`
  - `Dockerfile-prediction`
  - `dual-server-deployment.yaml`

**Optional (for deployment and testing):**
- kubectl configured for GKE cluster access

## Configuration

Before running the script, update the configuration variables in `build-deploy.sh`:

```bash
# Edit these values in the script
PROJECT_ID="your-gcp-project-id"
REGION="your-gcp-region"
REPOSITORY="your-artifact-registry-repo"
TRAINING_IMAGE="latencypredictor-training-server"
PREDICTION_IMAGE="latencypredictor-prediction-server"
TAG="latest"
```

## Usage

### Build Images Only

```bash
# Make script executable
chmod +x build-deploy.sh

# Build and push images to registry
./build-deploy.sh build
./build-deploy.sh push
```

### Complete Build and Deploy (Optional)

```bash
# Run complete process (build, push, deploy, test)
# Note: This requires GKE cluster access
./build-deploy.sh all
```

### Individual Commands

```bash
# Check if all required files exist
./build-deploy.sh check

# Build Docker images only
./build-deploy.sh build

# Push images to Google Artifact Registry
./build-deploy.sh push

# Optional: Deploy to GKE cluster (requires cluster access)
./build-deploy.sh deploy

# Optional: Get service information and IPs
./build-deploy.sh info

# Optional: Test the deployed services
./build-deploy.sh test
```

## What the Script Does

### Build Phase (`./build-deploy.sh build`)
- Builds training server image from `Dockerfile-training`
- Builds prediction server image from `Dockerfile-prediction` 
- Tags images for Google Artifact Registry
- Images created:
  - `latencypredictor-training-server:latest`
  - `latencypredictor-prediction-server:latest`

### Push Phase (`./build-deploy.sh push`)
- Configures Docker for Artifact Registry authentication
- Pushes both images to:
  - `us-docker.pkg.dev/PROJECT_ID/REPOSITORY/latencypredictor-training-server:latest`
  - `us-docker.pkg.dev/PROJECT_ID/REPOSITORY/latencypredictor-prediction-server:latest`

### Deploy Phase (`./build-deploy.sh deploy`) - Optional
- Applies Kubernetes manifests from `dual-server-deployment.yaml`
- Waits for deployments to be ready (5-minute timeout)
- Creates services:
  - `training-service-external` (LoadBalancer)
  - `prediction-service` (LoadBalancer)

### Test Phase (`./build-deploy.sh test`) - Optional
- Tests health endpoint: `/healthz`
- Tests prediction endpoint: `/predict` with sample data
- Sample prediction request:
  ```json
  {
    "kv_cache_percentage": 0.3,
    "input_token_length": 100,
    "num_request_waiting": 2,
    "num_request_running": 1,
    "num_tokens_generated": 50
  }
  ```

## Setup Instructions

1. **Configure GCP Authentication**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Configure kubectl for GKE (Optional - only needed for deployment)**:
   ```bash
   gcloud container clusters get-credentials CLUSTER_NAME --zone ZONE
   ```

3. **Update Script Configuration**:
   ```bash
   # Edit build-deploy.sh with your project details
   nano build-deploy.sh
   ```

4. **Build Images**:
   ```bash
   ./build-deploy.sh build
   ./build-deploy.sh push
   ```

5. **Optional: Deploy and Test**:
   ```bash
   ./build-deploy.sh deploy
   ./build-deploy.sh test
   # Or run everything at once
   ./build-deploy.sh all
   ```

## Troubleshooting

### Permission Issues
```bash
chmod +x build-deploy.sh
```

### GCP Authentication
```bash
gcloud auth configure-docker us-docker.pkg.dev
```

### Check Cluster Access
```bash
kubectl cluster-info
kubectl get nodes
```

### View Service Status
```bash
./build-deploy.sh info
kubectl get services
kubectl get pods
```

### Check Logs
```bash
# Training server logs
kubectl logs -l app=training-server

# Prediction server logs  
kubectl logs -l app=prediction-server
```

## Development Workflow

1. **Make code changes** to `training_server.py` or `prediction_server.py`
2. **Test locally** (optional):
   ```bash
   python training_server.py
   python prediction_server.py
   ```
3. **Build and push images**:
   ```bash
   ./build-deploy.sh build
   ./build-deploy.sh push
   ```

4. **Optional: Deploy and test**:
   ```bash
   ./build-deploy.sh deploy
   ./build-deploy.sh test
   ```

## Service Endpoints

After successful deployment:

- **Training Service**: External LoadBalancer IP (check with `./build-deploy.sh info`)
- **Prediction Service**: External LoadBalancer IP (check with `./build-deploy.sh info`)
- **Health Check**: `http://PREDICTION_IP/healthz`
- **Prediction API**: `http://PREDICTION_IP/predict` (POST)

## Manual Build (Alternative)

If you need to build manually:

```bash
# Build training server
docker build -f Dockerfile-training -t training-server .

# Build prediction server  
docker build -f Dockerfile-prediction -t prediction-server .
```