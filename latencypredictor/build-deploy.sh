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
set -e

# Configuration
PROJECT_ID="your-gcp-project-id"
REGION="your-gcp-region"
REPOSITORY="your-artifact-registry-repo"
TRAINING_IMAGE="latencypredictor-training-server"
PREDICTION_IMAGE="latencypredictor-prediction-server"
TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required files exist
check_files() {
    echo_status "Checking required files..."
    
    local files=("training_server.py" "prediction_server.py" "requirements.txt" "Dockerfile-training" "Dockerfile-prediction")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo_error "Required file $file not found!"
            exit 1
        fi
    done
    
    echo_status "All required files found."
}

# Build Docker images
build_images() {
    echo_status "Building Docker images..."
    
    # Build training server image
    echo_status "Building training server image..."
    docker build -f Dockerfile-training  -t ${TRAINING_IMAGE}:${TAG} .

    # Tag for training server
    docker tag ${TRAINING_IMAGE}:${TAG} \
        us-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${TRAINING_IMAGE}:${TAG}
    
    # Build prediction server image
    echo_status "Building prediction server image..."
    docker build -f Dockerfile-prediction -t ${PREDICTION_IMAGE}:${TAG} .

    # Tag for prediction server
    docker tag ${PREDICTION_IMAGE}:${TAG} \
        us-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${PREDICTION_IMAGE}:${TAG}
    
    echo_status "Images built successfully."
}

# Push images to Artifact Registry
push_images() {
    echo_status "Pushing images to Artifact Registry..."
    
    # Configure Docker for Artifact Registry
    gcloud auth configure-docker us-docker.pkg.dev --quiet
    
    # Push training server
    echo_status "Pushing training server image..."
    docker push us-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${TRAINING_IMAGE}:${TAG}
    
    # Push prediction server
    echo_status "Pushing prediction server image..."
    docker push us-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${PREDICTION_IMAGE}:${TAG}
    
    echo_status "Images pushed successfully."
}

# Deploy to GKE
deploy_to_gke() {
    echo_status "Deploying to GKE..."
    
    # Apply the Kubernetes manifests
    kubectl apply -f dual-server-deployment.yaml
    
    # Wait for deployments to be ready
    echo_status "Waiting for training server deployment..."
    kubectl rollout status deployment/training-server-deployment --timeout=300s
    
    echo_status "Waiting for prediction server deployment..."
    kubectl rollout status deployment/prediction-server-deployment --timeout=300s
    
    echo_status "Deployment completed successfully."
}

# Get service information
get_service_info() {
    echo_status "Getting service information..."
    
    echo_status "Training Service:"
    kubectl get service training-service-external -o wide
    
    echo_status "Prediction Service:"
    kubectl get service prediction-service -o wide
    
    echo_status "Getting external IPs (may take a few minutes)..."
    
    # Wait for external IPs
    echo_status "Waiting for training service external IP..."
    kubectl get service training-service-external --watch --timeout=300s &
    TRAINING_PID=$!
    
    echo_status "Waiting for prediction service external IP..."
    kubectl get service prediction-service --watch --timeout=300s &
    PREDICTION_PID=$!
    
    # Kill background processes after timeout
    sleep 10
    kill $TRAINING_PID $PREDICTION_PID 2>/dev/null || true
    
    echo_status "Current service status:"
    kubectl get services
}

# Test the deployment
test_deployment() {
    echo_status "Testing deployment..."
    
    # Get prediction service external IP
    PREDICTION_IP=$(kubectl get service prediction-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [[ -n "$PREDICTION_IP" ]]; then
        echo_status "Testing prediction endpoint at http://${PREDICTION_IP}/"
        
        # Test health endpoint
        if curl -f -s "http://${PREDICTION_IP}/healthz" > /dev/null; then
            echo_status "Health check passed!"
        else
            echo_warning "Health check failed or service not ready yet."
        fi
        
        # Test prediction endpoint
        echo_status "Testing prediction with sample data..."
        curl -X POST "http://${PREDICTION_IP}/predict" \
            -H "Content-Type: application/json" \
            -d '{
                "kv_cache_percentage": 0.3,
                "input_token_length": 100,
                "num_request_waiting": 2,
                "num_request_running": 1,
                "num_tokens_generated": 50
            }' || echo_warning "Prediction test failed or service not ready yet."
    else
        echo_warning "External IP not assigned yet. You can test later using:"
        echo "kubectl get services"
    fi
}

# Cleanup function
cleanup() {
    echo_status "Cleaning up..."
    docker system prune -f
}

# Main execution
main() {
    echo_status "Starting build and deployment process..."
    
    case "${1:-all}" in
        "check")
            check_files
            ;;
        "build")
            check_files
            build_images
            ;;
        "push")
            push_images
            ;;
        "deploy")
            deploy_to_gke
            ;;
        "info")
            get_service_info
            ;;
        "test")
            test_deployment
            ;;
        "all")
            check_files
            build_images
            push_images
            deploy_to_gke
            get_service_info
            test_deployment
            cleanup
            ;;
        *)
            echo "Usage: $0 {check|build|push|deploy|info|test|all}"
            echo ""
            echo "Commands:"
            echo "  check  - Check if required files exist"
            echo "  build  - Build Docker images"
            echo "  push   - Push images to Artifact Registry"
            echo "  deploy - Deploy to GKE"
            echo "  info   - Get service information"
            echo "  test   - Test the deployment"
            echo "  all    - Run complete build and deployment process"
            exit 1
            ;;
    esac
    
    echo_status "Process completed successfully!"
}

# Run main function
main "$@"