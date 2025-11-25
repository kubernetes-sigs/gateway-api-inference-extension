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
TEST_IMAGE="latencypredictor-test"
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
    
    # Check for test-specific files
    local test_files=("Dockerfile-test")
    for file in "${test_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo_warning "Test file $file not found - test image will not be built"
            TEST_BUILD_ENABLED=false
            return
        fi
    done
    
    TEST_BUILD_ENABLED=true
    echo_status "All required files found (including test files)."
}

# Build Docker images
build_images() {
    echo_status "Building Docker images..."
    
    # Build training server image
    echo_status "Building training server image..."
    docker build -f Dockerfile-training -t ${TRAINING_IMAGE}:${TAG} .

    # Tag for training server
    docker tag ${TRAINING_IMAGE}:${TAG} \
        us-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${TRAINING_IMAGE}:${TAG}
    
    # Build prediction server image
    echo_status "Building prediction server image..."
    docker build -f Dockerfile-prediction -t ${PREDICTION_IMAGE}:${TAG} .

    # Tag for prediction server
    docker tag ${PREDICTION_IMAGE}:${TAG} \
        us-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${PREDICTION_IMAGE}:${TAG}
    
    # Build test image if enabled
    if [[ "$TEST_BUILD_ENABLED" == "true" ]]; then
        echo_status "Building test image..."
        docker build -f Dockerfile-test -t ${TEST_IMAGE}:${TAG} .

        # Tag for test image
        docker tag ${TEST_IMAGE}:${TAG} \
            us-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${TEST_IMAGE}:${TAG}
        
        echo_status "All images (including test) built successfully."
    else
        echo_status "Images built successfully (test image skipped)."
    fi
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
    
    # Push test image if enabled
    if [[ "$TEST_BUILD_ENABLED" == "true" ]]; then
        echo_status "Pushing test image..."
        docker push us-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${TEST_IMAGE}:${TAG}
        echo_status "All images (including test) pushed successfully."
    else
        echo_status "Images pushed successfully (test image skipped)."
    fi
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

# Deploy test job
deploy_test() {
    echo_status "Deploying test job..."
    
    if [[ "$TEST_BUILD_ENABLED" != "true" ]]; then
        echo_warning "Test image not available. Skipping test deployment."
        return
    fi
    
    # Check if test manifest exists
    if [[ ! -f "test-job.yaml" ]]; then
        echo_warning "test-job.yaml not found. Creating a basic test job..."
        create_test_manifest
    fi
    
    # Delete existing test job if it exists
    kubectl delete job latency-predictor-test --ignore-not-found=true
    
    # Apply test job
    kubectl apply -f test-job.yaml
    
    echo_status "Test job deployed. Monitor with: kubectl logs -f job/latency-predictor-test"
}

# Create a basic test manifest
create_test_manifest() {
    cat > test-job.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: latency-predictor-test
  namespace: default
  labels:
    app: latency-predictor-test
    component: test
spec:
  template:
    metadata:
      labels:
        app: latency-predictor-test
        component: test
    spec:
      nodeSelector:
        cloud.google.com/gke-nodepool: "pool-2"
      restartPolicy: Never
      containers:
      - name: test-runner
        image: us-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${TEST_IMAGE}:${TAG}
        imagePullPolicy: Always
        command: ["pytest"]
        args: ["-v", "-s", "test_dual_server_client.py"]
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "1000m"
            memory: "2Gi"
        env:
        - name: TRAINING_SERVER_URL
          value: "http://training-service:8000"
        - name: PREDICTION_SERVER_URL
          value: "http://prediction-service:80"
        - name: TEST_TIMEOUT
          value: "300"
        volumeMounts:
        - name: test-results
          mountPath: /test-results
      volumes:
      - name: test-results
        emptyDir: {}
  backoffLimit: 3
EOF
    echo_status "Created basic test-job.yaml manifest."
}

# Run tests
run_tests() {
    echo_status "Running tests..."
    
    if [[ "$TEST_BUILD_ENABLED" != "true" ]]; then
        echo_warning "Test image not available. Running basic connectivity tests instead..."
        test_deployment
        return
    fi
    
    # Deploy and run test job
    deploy_test
    
    # Wait for job completion and show logs
    echo_status "Waiting for test job to complete..."
    kubectl wait --for=condition=complete job/latency-predictor-test --timeout=600s || {
        echo_error "Test job did not complete successfully"
        kubectl describe job latency-predictor-test
        kubectl logs job/latency-predictor-test
        return 1
    }
    
    echo_status "Test job completed. Showing logs:"
    kubectl logs job/latency-predictor-test
    
    # Clean up test job
    echo_status "Cleaning up test job..."
    kubectl delete job latency-predictor-test
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

# Test the deployment (basic connectivity tests)
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

# List built images
list_images() {
    echo_status "Listing built images..."
    
    echo_status "Local images:"
    docker images | grep -E "${TRAINING_IMAGE}|${PREDICTION_IMAGE}|${TEST_IMAGE}" || echo "No local images found"
    
    echo_status "Remote images in Artifact Registry:"
    gcloud artifacts docker images list us-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY} \
        --include-tags --filter="package~(${TRAINING_IMAGE}|${PREDICTION_IMAGE}|${TEST_IMAGE})" || echo "No remote images found"
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
            check_files
            push_images
            ;;
        "deploy")
            deploy_to_gke
            ;;
        "test-deploy")
            check_files
            deploy_test
            ;;
        "test")
            check_files
            run_tests
            ;;
        "info")
            get_service_info
            ;;
        "images")
            list_images
            ;;
        "basic-test")
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
        "full")
            check_files
            build_images
            push_images
            deploy_to_gke
            get_service_info
            run_tests
            cleanup
            ;;
        *)
            echo "Usage: $0 {check|build|push|deploy|test-deploy|test|info|images|basic-test|all|full}"
            echo ""
            echo "Commands:"
            echo "  check      - Check if required files exist"
            echo "  build      - Build Docker images (including test if Dockerfile-test exists)"
            echo "  push       - Push images to Artifact Registry"
            echo "  deploy     - Deploy to GKE"
            echo "  test-deploy- Deploy test job only"
            echo "  test       - Run comprehensive tests using test image"
            echo "  info       - Get service information"
            echo "  images     - List built images (local and remote)"
            echo "  basic-test - Run basic connectivity tests"
            echo "  all        - Run complete build and deployment process (no tests)"
            echo "  full       - Run complete process including comprehensive tests"
            exit 1
            ;;
    esac
    
    echo_status "Process completed successfully!"
}

# Run main function
main "$@"