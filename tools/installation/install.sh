#!/usr/bin/env bash
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

# install.sh - Interactive installer for Gateway API Inference Extension (GAIE) with BBR
#
# Installs GAIE with Body-Based Routing (BBR) following the official guide:
#   https://gateway-api-inference-extension.sigs.k8s.io/guides/#deploy-the-body-based-router-extension-optional
#
# Stack: kind cluster, MetalLB, Gateway API CRDs, vLLM Simulator, Inference Extension
# CRDs, Istio (with inference extension), Inference Gateway, InferencePool/EPP, and BBR.
#
# Prerequisites (must be installed): kubectl, helm, kind, jq, curl
# Optional: docker (for kind).
#
# Usage:
#   ./install.sh [--dry-run] [--cleanup] [--help]
#
# Options:
#   --dry-run    Print commands without executing
#   --cleanup    Remove all resources created by this installer
#   --help       Show help

set -euo pipefail

# ============================================================================
# A: Constants, Colors, Global Variables
# ============================================================================

readonly SCRIPT_VERSION="0.2.0"
readonly CLUSTER_NAME="inference-gateway"
readonly GUIDE_URL="https://gateway-api-inference-extension.sigs.k8s.io/guides/"
readonly GATEWAY_API_INSTALL_URL="https://gateway-api.sigs.k8s.io/guides/getting-started/#installing-gateway-api"

# Versions (fixed for compatibility with guide and older kind clusters)
# Gateway API v1.5.0+ requires Kubernetes 1.29+ (ValidatingAdmissionPolicy, CEL isIP).
# v1.2.1 works on older K8s; see https://gateway-api.sigs.k8s.io/guides/getting-started/#installing-gateway-api
readonly METALLB_VERSION="v0.14.9"
readonly GATEWAY_API_VERSION="v1.2.1"
readonly ISTIO_VERSION="1.28.0"

# Release and URLs (IGW_LATEST_RELEASE set at runtime from GitHub API)
IGW_LATEST_RELEASE=""
MANIFESTS_BASE=""
CRD_MANIFESTS_URL=""
INFERENCEPOOL_CHART=""
BBR_CHART=""

# Colors (disabled if not a terminal)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' BOLD='' NC=''
fi

# State tracking (fixed path: Istio + MetalLB + vLLM simulator + BBR)
DRY_RUN=false
CLEANUP_MODE=false
CLUSTER_CREATED=false
CLUSTER_TYPE=""
LB_TYPE=""
INFERENCE_POOL_NAME="vllm-llama3-8b-instruct"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# ============================================================================
# B: Utility Functions
# ============================================================================

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()    { echo -e "\n${BOLD}${CYAN}=== $* ===${NC}\n"; }

# Set release and derived URLs per official guide
set_release_urls() {
    if [[ -n "$IGW_LATEST_RELEASE" ]]; then
        return 0
    fi
    log_info "Fetching latest GAIE release from GitHub..."
    IGW_LATEST_RELEASE=$(curl -s https://api.github.com/repos/kubernetes-sigs/gateway-api-inference-extension/releases \
        | jq -r '.[] | select(.prerelease == false) | .tag_name' \
        | sort -V \
        | tail -n1)
    if [[ -z "$IGW_LATEST_RELEASE" || "$IGW_LATEST_RELEASE" == "null" ]]; then
        log_error "Could not determine latest release. Check network and GitHub API."
        exit 1
    fi
    log_success "Using GAIE release: $IGW_LATEST_RELEASE"
    MANIFESTS_BASE="https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests"
    CRD_MANIFESTS_URL="https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml"
    INFERENCEPOOL_CHART="oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool"
    BBR_CHART="oci://registry.k8s.io/gateway-api-inference-extension/charts/body-based-routing"
}

run_cmd() {
    local desc="${1:-}"
    shift
    if [[ -n "$desc" ]]; then
        log_info "$desc"
    fi
    echo -e "${YELLOW}\$ $*${NC}"
    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi
    "$@"
}

# run_cmd_quiet: run without printing the command, just description
run_cmd_quiet() {
    local desc="${1:-}"
    shift
    if [[ -n "$desc" ]]; then
        log_info "$desc"
    fi
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}\$ $*${NC}"
        return 0
    fi
    "$@"
}

show_menu() {
    local prompt="$1"
    shift
    local options=("$@")

    echo -e "\n${BOLD}${prompt}${NC}" >&2
    local i
    for i in "${!options[@]}"; do
        echo -e "  ${BOLD}$((i + 1))${NC}) ${options[$i]}" >&2
    done

    local choice
    while true; do
        read -r -p "Enter choice [1-${#options[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
            echo "$choice"
            return 0
        fi
        log_warn "Invalid choice. Please enter a number between 1 and ${#options[@]}."
    done
}

confirm() {
    local prompt="${1:-Continue?}"
    local answer
    read -r -p "${prompt} [y/N]: " answer
    [[ "$answer" =~ ^[Yy]$ ]]
}

wait_for_resource() {
    local resource="$1"
    local condition="${2:-condition=ready}"
    local timeout="${3:-300s}"
    local namespace="${4:-}"

    local ns_flag=""
    if [[ -n "$namespace" ]]; then
        ns_flag="-n $namespace"
    fi

    log_info "Waiting for $resource ($condition, timeout: $timeout)..."
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}\$ kubectl wait $ns_flag --for=$condition $resource --timeout=$timeout${NC}"
        return 0
    fi
    # shellcheck disable=SC2086
    kubectl wait $ns_flag --for="$condition" "$resource" --timeout="$timeout" 2>/dev/null || {
        log_warn "Timed out waiting for $resource. It may still be starting up."
        return 1
    }
    log_success "$resource is ready."
}

spinner() {
    local pid=$1
    local msg="${2:-Working...}"
    local spin_chars='|/-\\'
    local i=0

    if [[ "$DRY_RUN" == true ]] || [[ ! -t 1 ]]; then
        wait "$pid" 2>/dev/null
        return $?
    fi

    while kill -0 "$pid" 2>/dev/null; do
        printf "\r  %s %s" "${spin_chars:i++%4:1}" "$msg"
        sleep 0.2
    done
    printf "\r"
    wait "$pid" 2>/dev/null
    return $?
}

print_separator() {
    echo -e "${CYAN}$(printf '%.0s-' {1..60})${NC}"
}

# ============================================================================
# C: Step Functions
# ============================================================================

step_prerequisites() {
    log_step "Step 0: Checking Prerequisites"

    # Required by official guide: kubectl, helm, kind, jq; curl for fetching latest release
    local required=(kubectl helm kind jq curl)
    local missing_required=()
    local missing_optional=()

    for cmd in "${required[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            missing_required+=("$cmd")
        else
            log_success "$cmd found: $(command -v "$cmd")"
        fi
    done

    for cmd in curl docker; do
        if ! command -v "$cmd" &>/dev/null; then
            missing_optional+=("$cmd")
        else
            log_success "$cmd found: $(command -v "$cmd")"
        fi
    done

    if [[ ${#missing_required[@]} -gt 0 ]]; then
        log_error "Required tools missing: ${missing_required[*]}"
        log_error "Install them before proceeding. See: ${GUIDE_URL}"
        exit 1
    fi

    if [[ ${#missing_optional[@]} -gt 0 ]]; then
        log_warn "Optional tools missing: ${missing_optional[*]} (curl recommended for release fetch)"
    fi

    # Check inotify limits (required for kind clusters)
    if [[ -f /proc/sys/fs/inotify/max_user_instances ]]; then
        local inotify_instances
        inotify_instances=$(cat /proc/sys/fs/inotify/max_user_instances)
        if (( inotify_instances < 512 )); then
            log_warn "fs.inotify.max_user_instances is $inotify_instances (recommended: >= 512)"
            log_warn "Kind clusters may fail with 'too many open files'. Fix: sudo sysctl fs.inotify.max_user_instances=1024"
        else
            log_success "inotify limits OK (max_user_instances=$inotify_instances)"
        fi
    fi

    log_success "Prerequisites OK (kubectl, helm, kind, jq, curl)."
}

step_cluster_setup() {
    log_step "Step 1: Cluster Setup"

    local choice
    choice=$(show_menu "Select cluster:" "Create kind cluster ($CLUSTER_NAME)" "Skip (use existing cluster)")

    case "$choice" in
        1)
            CLUSTER_TYPE="kind"
            log_info "Creating kind cluster '$CLUSTER_NAME'..."

            if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
                log_warn "Kind cluster '$CLUSTER_NAME' already exists."
                if confirm "Delete and recreate it?"; then
                    run_cmd "Deleting existing kind cluster" kind delete cluster --name "$CLUSTER_NAME"
                else
                    log_info "Using existing cluster."
                    CLUSTER_CREATED=true
                    return
                fi
            fi

            echo "" >&2
            echo -e "  ${BOLD}Worker nodes: 0 = control-plane only (recommended for simulators), 1+ = workers${NC}" >&2
            read -r -p "Number of worker nodes [0]: " num_workers
            num_workers="${num_workers:-0}"
            [[ "$num_workers" =~ ^[0-9]+$ ]] || num_workers=0

            local kind_config="kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane"
            local w
            for (( w=0; w<num_workers; w++ )); do
                kind_config+=$'\n'"- role: worker"
            done

            run_cmd "Creating kind cluster" \
                kind create cluster --name "$CLUSTER_NAME" --wait 5m --config /dev/stdin <<< "$kind_config"
            CLUSTER_CREATED=true

            if [[ "$DRY_RUN" != true ]]; then
                log_info "Verifying cluster health..."
                kubectl wait --for=condition=ready pod -l k8s-app=kube-proxy -n kube-system --timeout=60s 2>/dev/null || {
                    log_error "kube-proxy not healthy (often inotify limits). Fix: sudo sysctl fs.inotify.max_user_instances=1024"
                    exit 1
                }
                log_success "Cluster healthy."
            fi
            log_success "Kind cluster '$CLUSTER_NAME' created."
            ;;
        2)
            log_info "Using existing cluster."
            if [[ "$DRY_RUN" != true ]]; then
                if ! kubectl cluster-info &>/dev/null; then
                    log_error "Cannot connect to cluster. Configure kubectl first."
                    exit 1
                fi
                log_success "Connected to existing cluster."
            fi
            ;;
    esac
}

step_loadbalancer() {
    log_step "Step 2: LoadBalancer (MetalLB) Setup"

    if [[ "$CLUSTER_CREATED" != true ]]; then
        log_info "Skipping LoadBalancer (cluster was not created by this script)."
        return
    fi

    # For kind, MetalLB is required for LoadBalancer services (per official guide).
    if confirm "Install MetalLB for LoadBalancer support on kind?"; then
        LB_TYPE="metallb"
        _install_metallb
    else
        log_warn "Without MetalLB, Gateway may not get an external IP on kind."
    fi
}

_install_metallb() {
    log_info "Installing MetalLB ${METALLB_VERSION}..."

    run_cmd "Applying MetalLB manifests" \
        kubectl apply -f "https://raw.githubusercontent.com/metallb/metallb/${METALLB_VERSION}/config/manifests/metallb-native.yaml"

    if [[ "$DRY_RUN" != true ]]; then
        # MetalLB has a chicken-and-egg problem: the validating webhook has
        # failurePolicy=Fail, but the webhook is served by the controller pod.
        # The controller can't start because the API server can't reach the
        # webhook to validate metallb.io resources during discovery.
        # Fix: temporarily set failurePolicy to Ignore, let the controller
        # start, then restore it.
        log_info "Patching MetalLB webhook failurePolicy to Ignore (temporary)..."
        kubectl patch validatingwebhookconfiguration metallb-webhook-configuration \
            --type=json \
            -p='[{"op":"replace","path":"/webhooks/0/failurePolicy","value":"Ignore"},{"op":"replace","path":"/webhooks/1/failurePolicy","value":"Ignore"},{"op":"replace","path":"/webhooks/2/failurePolicy","value":"Ignore"},{"op":"replace","path":"/webhooks/3/failurePolicy","value":"Ignore"}]' \
            2>/dev/null || true

        # Wait for controller deployment to become ready
        log_info "Waiting for MetalLB controller..."
        local retries=0
        while (( retries < 60 )); do
            local ready
            ready=$(kubectl get deployment controller -n metallb-system -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            if [[ "$ready" == "1" ]]; then
                log_success "MetalLB controller is ready."
                break
            fi
            sleep 5
            (( retries++ ))
            if (( retries % 12 == 0 )); then
                log_info "Still waiting... ($(( retries * 5 ))s elapsed)"
                kubectl get pods -n metallb-system --no-headers 2>/dev/null || true
            fi
        done

        if (( retries >= 60 )); then
            log_error "MetalLB controller failed to start. Logs:"
            kubectl logs -n metallb-system deployment/controller --tail=10 2>/dev/null || true
            log_error "MetalLB installation failed. Please check your cluster."
            exit 1
        fi

        # Wait for speaker pods to be ready too
        log_info "Waiting for MetalLB speaker..."
        kubectl wait --for=condition=ready pod -l component=speaker -n metallb-system --timeout=120s 2>/dev/null || true

        # Restore webhook failurePolicy to Fail now that everything is running
        log_info "Restoring MetalLB webhook failurePolicy to Fail..."
        kubectl patch validatingwebhookconfiguration metallb-webhook-configuration \
            --type=json \
            -p='[{"op":"replace","path":"/webhooks/0/failurePolicy","value":"Fail"},{"op":"replace","path":"/webhooks/1/failurePolicy","value":"Fail"},{"op":"replace","path":"/webhooks/2/failurePolicy","value":"Fail"},{"op":"replace","path":"/webhooks/3/failurePolicy","value":"Fail"}]' \
            2>/dev/null || true
    fi

    # Detect IP range based on cluster type
    local ip_range=""
    if [[ "$CLUSTER_TYPE" == "kind" ]]; then
        if [[ "$DRY_RUN" != true ]]; then
            local subnet
            subnet=$(docker network inspect kind -f '{{(index .IPAM.Config 0).Subnet}}' 2>/dev/null || echo "172.18.0.0/16")
            # Extract first two octets
            local prefix
            prefix=$(echo "$subnet" | cut -d'.' -f1-2)
            ip_range="${prefix}.255.200-${prefix}.255.250"
        else
            ip_range="172.18.255.200-172.18.255.250"
        fi
    elif [[ "$CLUSTER_TYPE" == "minikube" ]]; then
        if [[ "$DRY_RUN" != true ]]; then
            local minikube_ip
            minikube_ip=$(minikube ip --profile="$CLUSTER_NAME" 2>/dev/null || echo "192.168.49.2")
            local prefix
            prefix=$(echo "$minikube_ip" | cut -d'.' -f1-3)
            ip_range="${prefix}.100-${prefix}.120"
        else
            ip_range="192.168.49.100-192.168.49.120"
        fi
    fi

    log_info "Configuring MetalLB with IP range: $ip_range"

    run_cmd "Applying MetalLB IP pool configuration" \
        kubectl apply -f - <<EOF
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: default-pool
  namespace: metallb-system
spec:
  addresses:
  - ${ip_range}
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: default
  namespace: metallb-system
spec:
  ipAddressPools:
  - default-pool
EOF

    log_success "MetalLB installed and configured."
}

step_gateway_api_crds() {
    log_step "Step 3: Gateway API CRDs (required before Istio)"

    # Per https://gateway-api.sigs.k8s.io/guides/getting-started/#installing-gateway-api
    # Gateway API CRDs must be installed before installing a Gateway controller (e.g. Istio).
    log_info "Installing Gateway API per: ${GATEWAY_API_INSTALL_URL}"

    local choice
    choice=$(show_menu "Gateway API CRDs:" "Standard (recommended)" "Experimental" "Skip (already installed)")

    case "$choice" in
        1)
            run_cmd "Installing Gateway API CRDs (Standard, server-side apply)" \
                kubectl apply --server-side -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"
            log_success "Gateway API CRDs (Standard) installed."
            ;;
        2)
            run_cmd "Installing Gateway API CRDs (Experimental, server-side apply)" \
                kubectl apply --server-side -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/experimental-install.yaml"
            log_success "Gateway API CRDs (Experimental) installed."
            ;;
        3)
            log_info "Skipping Gateway API CRDs."
            ;;
    esac
}

step_model_server() {
    log_step "Step 4: Deploy Model Server (vLLM Simulator)"

    # Per official guide: vLLM Simulator - no GPU, ideal for test/dev.
    log_info "Deploying vLLM Simulator (${INFERENCE_POOL_NAME}, model: ${MODEL_NAME})..."
    run_cmd "Deploying vLLM Simulator" \
        kubectl apply -f "${MANIFESTS_BASE}/vllm/sim-deployment.yaml"
    wait_for_resource "deployment/vllm-llama3-8b-instruct" "condition=available" "120s" || true
    log_success "vLLM Simulator deployed."
}

step_inference_crds() {
    log_step "Step 5: Install Inference Extension CRDs"

    run_cmd "Installing Inference Extension CRDs" \
        kubectl apply -f "$CRD_MANIFESTS_URL"

    log_success "Inference Extension CRDs installed."
}

step_gateway_controller() {
    log_step "Step 6: Install Gateway Controller (Istio)"

    log_info "Installing Istio with Inference Extension support (per guide)..."
    _install_istio
    log_success "Istio configured as gateway provider."
}

_install_istio() {
    log_info "Installing Istio ${ISTIO_VERSION}..."

    local tmpdir
    tmpdir=$(mktemp -d)
    local istioctl_bin="${tmpdir}/istio-${ISTIO_VERSION}/bin/istioctl"

    run_cmd "Downloading Istio ${ISTIO_VERSION}" \
        bash -c "cd ${tmpdir} && curl -sL https://istio.io/downloadIstio | ISTIO_VERSION=${ISTIO_VERSION} sh -"

    if [[ "$DRY_RUN" != true ]] && [[ ! -x "$istioctl_bin" ]]; then
        log_error "istioctl not found after download. Check the Istio installation."
        rm -rf "$tmpdir"
        exit 1
    fi

    run_cmd "Installing Istio with Inference Extension support" \
        "$istioctl_bin" install \
        --set values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true \
        -y

    # Clean up downloaded files
    rm -rf "$tmpdir"
    log_success "Istio ${ISTIO_VERSION} installed."
}

step_deploy_gateway() {
    log_step "Step 7: Deploy Inference Gateway"

    local gateway_url="${MANIFESTS_BASE}/gateway/istio/gateway.yaml"
    run_cmd "Deploying Inference Gateway (Istio)" \
        kubectl apply -f "$gateway_url"

    log_info "Waiting for gateway to be programmed..."
    if [[ "$DRY_RUN" != true ]]; then
        local retries=0
        local max_retries=30
        while (( retries < max_retries )); do
            local programmed
            programmed=$(kubectl get gateway inference-gateway -o jsonpath='{.status.conditions[?(@.type=="Programmed")].status}' 2>/dev/null || echo "")
            if [[ "$programmed" == "True" ]]; then
                break
            fi
            sleep 10
            (( retries++ ))
        done

        if (( retries >= max_retries )); then
            log_warn "Gateway did not reach Programmed=True within timeout."
            log_warn "Check status with: kubectl get gateway inference-gateway"
        else
            log_success "Inference Gateway is programmed and ready."
            kubectl get gateway inference-gateway
        fi
    else
        echo -e "${YELLOW}\$ kubectl get gateway inference-gateway${NC}"
    fi
}

step_deploy_inferencepool() {
    log_step "Step 8: Deploy InferencePool and EPP"

    run_cmd "Installing InferencePool via Helm" \
        helm install "${INFERENCE_POOL_NAME}" \
        --dependency-update \
        --set "inferencePool.modelServers.matchLabels.app=${INFERENCE_POOL_NAME}" \
        --set "provider.name=istio" \
        --set "experimentalHttpRoute.enabled=true" \
        --version "$IGW_LATEST_RELEASE" \
        "$INFERENCEPOOL_CHART"

    log_success "InferencePool and EPP deployed."
}

step_verify_status() {
    log_step "Step 9: Verify Status"

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}\$ kubectl get httproute ${INFERENCE_POOL_NAME} -o yaml${NC}"
        echo -e "${YELLOW}\$ kubectl get inferencepool ${INFERENCE_POOL_NAME} -o yaml${NC}"
        return
    fi

    log_info "Checking HttpRoute status..."
    local httproute_accepted
    httproute_accepted=$(kubectl get httproute "${INFERENCE_POOL_NAME}" \
        -o jsonpath='{.status.parents[0].conditions[?(@.type=="Accepted")].status}' 2>/dev/null || echo "Unknown")
    if [[ "$httproute_accepted" == "True" ]]; then
        log_success "HttpRoute Accepted=True"
    else
        log_warn "HttpRoute Accepted=$httproute_accepted (expected True)"
    fi

    log_info "Checking InferencePool status..."
    local pool_accepted
    pool_accepted=$(kubectl get inferencepool "${INFERENCE_POOL_NAME}" \
        -o jsonpath='{.status.conditions[?(@.type=="Accepted")].status}' 2>/dev/null || echo "Unknown")
    if [[ "$pool_accepted" == "True" ]]; then
        log_success "InferencePool Accepted=True"
    else
        log_warn "InferencePool Accepted=$pool_accepted (expected True)"
    fi
}

step_inference_objective() {
    log_step "Step 10: Deploy InferenceObjective (Optional)"

    if confirm "Deploy sample InferenceObjective (enables request priority)?"; then
        run_cmd "Deploying InferenceObjective" \
            kubectl apply -f "${MANIFESTS_BASE}/inferenceobjective.yaml"
        log_success "InferenceObjective deployed."
    else
        log_info "Skipping InferenceObjective."
    fi
}

step_test() {
    log_step "Step 11: Test the Setup"

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}\$ IP=\$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')${NC}"
        echo -e "${YELLOW}\$ PORT=80${NC}"
        echo -e "${YELLOW}\$ curl -i \${IP}:\${PORT}/v1/completions ...${NC}"
        return
    fi

    local ip
    ip=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}' 2>/dev/null || echo "")
    local port=80

    if [[ -z "$ip" ]]; then
        log_warn "Could not determine gateway IP address."
        log_info "Get it manually with: kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}'"
        return
    fi

    log_info "Gateway IP: $ip"
    log_info "Sending test request..."

    if ! command -v curl &>/dev/null; then
        log_warn "curl not installed. Run this manually:"
        echo ""
        echo "  curl -i ${ip}:${port}/v1/completions -H 'Content-Type: application/json' -d '{"
        echo '    "model": "food-review-1",'
        echo '    "prompt": "Write as if you were a critic: San Francisco",'
        echo '    "max_tokens": 100,'
        echo '    "temperature": 0'
        echo "  }'"
        return
    fi

    curl -i "${ip}:${port}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{
            \"model\": \"${MODEL_NAME}\",
            \"prompt\": \"Write as if you were a critic: San Francisco\",
            \"max_tokens\": 100,
            \"temperature\": 0
        }" 2>/dev/null || {
        log_warn "Test request failed. The model server may still be starting up."
        log_info "Retry with:"
        echo "  IP=${ip}; PORT=${port}"
        echo "  curl -i \${IP}:\${PORT}/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"${MODEL_NAME}\",\"prompt\":\"Hello\",\"max_tokens\":10,\"temperature\":0}'"
    }
    echo ""
}

step_bbr() {
    log_step "Step 12: Deploy Body-Based Routing (BBR) Extension"

    # Per guide: https://gateway-api-inference-extension.sigs.k8s.io/guides/#deploy-the-body-based-router-extension-optional
    log_info "Deploying BBR for model-aware routing (multiple InferencePools at same L7 path)..."

    # 1. Deploy BBR extension (Istio)
    run_cmd "Installing BBR extension via Helm" \
        helm install body-based-router \
        --set provider.name=istio \
        --version "$IGW_LATEST_RELEASE" \
        "$BBR_CHART"

    # 2. Deploy second model server (DeepSeek simulator)
    run_cmd "Deploying second model server (DeepSeek simulator)" \
        kubectl apply -f "${MANIFESTS_BASE}/bbr/sim-deployment.yaml"

    # 3. Deploy second InferencePool
    run_cmd "Installing second InferencePool (vllm-deepseek-r1)" \
        helm install vllm-deepseek-r1 \
        --dependency-update \
        --set inferencePool.modelServers.matchLabels.app=vllm-deepseek-r1 \
        --set "provider.name=istio" \
        --set experimentalHttpRoute.enabled=true \
        --set "experimentalHttpRoute.baseModel=deepseek/vllm-deepseek-r1" \
        --version "$IGW_LATEST_RELEASE" \
        "$INFERENCEPOOL_CHART"

    # 4. Apply LoRA configmap (base model + LoRA names)
    run_cmd "Applying LoRA adapter configmap" \
        kubectl apply -f "${MANIFESTS_BASE}/bbr/configmap.yaml"

    # 5. Upgrade first InferencePool with base model for BBR routing
    run_cmd "Upgrading first InferencePool with base model mapping" \
        helm upgrade "${INFERENCE_POOL_NAME}" "$INFERENCEPOOL_CHART" \
        --dependency-update \
        --set "inferencePool.modelServers.matchLabels.app=${INFERENCE_POOL_NAME}" \
        --set "provider.name=istio" \
        --set experimentalHttpRoute.enabled=true \
        --set "experimentalHttpRoute.baseModel=meta-llama/Llama-3.1-8B-Instruct" \
        --version "$IGW_LATEST_RELEASE"

    if [[ "$DRY_RUN" != true ]]; then
        log_info "Verifying BBR setup..."
        kubectl get inferencepools 2>/dev/null || true
        kubectl get pods 2>/dev/null || true
    fi

    log_success "BBR multi-pool setup complete."

    _test_bbr
}

_test_bbr() {
    if [[ "$DRY_RUN" == true ]]; then
        log_info "BBR test requests (dry-run, skipping actual curl)."
        return
    fi

    local ip
    ip=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}' 2>/dev/null || echo "")
    local port=80

    if [[ -z "$ip" ]] || ! command -v curl &>/dev/null; then
        log_info "Skipping BBR test (gateway IP not available or curl not installed)."
        return
    fi

    if confirm "Run BBR test requests?"; then
        log_info "Testing Llama base model..."
        curl -s -o /dev/null -w "HTTP %{http_code}" -X POST "${ip}:${port}/v1/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"Hello","max_tokens":10,"temperature":0}' || true
        echo ""

        log_info "Testing DeepSeek base model..."
        curl -s -o /dev/null -w "HTTP %{http_code}" -X POST "${ip}:${port}/v1/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"deepseek/vllm-deepseek-r1","prompt":"Hello","max_tokens":10,"temperature":0}' || true
        echo ""

        log_info "Testing LoRA adapter (food-review-1)..."
        curl -s -o /dev/null -w "HTTP %{http_code}" -X POST "${ip}:${port}/v1/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"food-review-1","prompt":"Hello","max_tokens":10,"temperature":0}' || true
        echo ""
    fi
}

# ============================================================================
# D: Cleanup Function
# ============================================================================

do_cleanup() {
    log_step "Cleanup"

    # Need release for manifest URLs
    set_release_urls

    log_warn "This will remove all resources created by this installer (Istio, MetalLB, GAIE, BBR)."
    if ! confirm "Proceed with cleanup?"; then
        log_info "Cleanup cancelled."
        return
    fi

    # BBR
    log_info "Cleaning up BBR resources..."
    helm uninstall body-based-router --ignore-not-found 2>/dev/null || true
    helm uninstall vllm-deepseek-r1 --ignore-not-found 2>/dev/null || true
    kubectl delete -f "${MANIFESTS_BASE}/bbr/sim-deployment.yaml" --ignore-not-found 2>/dev/null || true
    kubectl delete -f "${MANIFESTS_BASE}/bbr/configmap.yaml" --ignore-not-found 2>/dev/null || true

    # InferencePool and model server
    log_info "Cleaning up InferencePool and model server..."
    helm uninstall "${INFERENCE_POOL_NAME}" --ignore-not-found 2>/dev/null || true
    kubectl delete -f "${MANIFESTS_BASE}/inferenceobjective.yaml" --ignore-not-found 2>/dev/null || true
    kubectl delete -f "${MANIFESTS_BASE}/vllm/sim-deployment.yaml" --ignore-not-found 2>/dev/null || true
    kubectl delete secret hf-token --ignore-not-found 2>/dev/null || true

    # Inference Extension CRDs
    log_info "Cleaning up Inference Extension CRDs..."
    kubectl delete -f "$CRD_MANIFESTS_URL" --ignore-not-found 2>/dev/null || true

    # Gateway
    log_info "Cleaning up Inference Gateway..."
    kubectl delete -f "${MANIFESTS_BASE}/gateway/istio/gateway.yaml" --ignore-not-found 2>/dev/null || true

    # Istio
    log_info "Cleaning up Istio..."
    if command -v istioctl &>/dev/null; then
        istioctl uninstall -y --purge 2>/dev/null || true
    fi
    kubectl delete ns istio-system --ignore-not-found 2>/dev/null || true

    # MetalLB
    log_info "Cleaning up MetalLB..."
    kubectl delete -f "https://raw.githubusercontent.com/metallb/metallb/${METALLB_VERSION}/config/manifests/metallb-native.yaml" --ignore-not-found 2>/dev/null || true

    # Cluster
    if confirm "Delete the kind cluster '$CLUSTER_NAME' if it was created by this script?"; then
        if command -v kind &>/dev/null && kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
            run_cmd "Deleting kind cluster" kind delete cluster --name "$CLUSTER_NAME"
        fi
    fi

    log_success "Cleanup complete."
}

# ============================================================================
# E: Summary Function
# ============================================================================

print_summary() {
    echo ""
    print_separator
    echo -e "${BOLD}${GREEN}Installation Summary${NC}"
    print_separator
    echo -e "  Cluster:           ${CLUSTER_TYPE:-existing}"
    echo -e "  LoadBalancer:      ${LB_TYPE:-none}"
    echo -e "  Gateway:          Istio"
    echo -e "  Inference pools:  ${INFERENCE_POOL_NAME}, vllm-deepseek-r1 (BBR)"
    print_separator

    echo ""
    echo -e "${BOLD}Useful commands:${NC}"
    echo "  kubectl get gateway inference-gateway"
    echo "  kubectl get inferencepool"
    echo "  kubectl get httproute"
    echo "  kubectl get pods"
    echo ""
    echo -e "  ${BOLD}Get gateway IP:${NC}"
    echo '  IP=$(kubectl get gateway/inference-gateway -o jsonpath='"'"'{.status.addresses[0].value}'"'"')'
    echo ""
    echo -e "  ${BOLD}Test (base model or LoRA):${NC}"
    echo '  curl -i ${IP}:80/v1/completions -H "Content-Type: application/json" \'
    echo '    -d '"'"'{"model":"'"${MODEL_NAME}"'","prompt":"Hello","max_tokens":10,"temperature":0}'"'"
    echo "  # Or use model names: food-review-1, deepseek/vllm-deepseek-r1, ski-resorts, movie-critique"
    echo ""
    echo -e "  ${BOLD}Cleanup:${NC}"
    echo "  $(realpath "$0" 2>/dev/null || echo "$0") --cleanup"
    echo ""
    echo -e "  ${BOLD}Guide:${NC} ${GUIDE_URL}"
    print_separator
}

# ============================================================================
# F: Main Flow / Argument Parsing
# ============================================================================

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Interactive installer for Gateway API Inference Extension (GAIE) with Body-Based
Routing (BBR). Uses kind, MetalLB, Istio, vLLM Simulator. Follows:
  ${GUIDE_URL}

Prerequisites: kubectl, helm, kind, jq, curl

Options:
  --dry-run    Print commands without executing them
  --cleanup    Remove all resources created by this installer
  --help       Show this help message

Examples:
  ./install.sh
  ./install.sh --dry-run
  ./install.sh --cleanup
EOF
}

print_banner() {
    echo -e "${CYAN}"
    cat <<'BANNER'
  ____       _                           _    ____ ___
 / ___| __ _| |_ _____      ____ _ _   _   / \  |  _ \_ _|
| |  _ / _` | __/ _ \ \ /\ / / _` | | | | / _ \ | |_) | |
| |_| | (_| | ||  __/\ V  V / (_| | |_| |/ ___ \|  __/| |
 \____|\__,_|\__\___| \_/\_/ \__,_|\__, /_/   \_\_|  |___|
 ___        __                      |___/
|_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___
 | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
 | || | | |  _|  __/ | |  __/ | | | (_|  __/
|___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
 _____      _                 _
| ____|_  _| |_ ___ _ __  ___(_) ___  _ __
|  _| \ \/ / __/ _ \ '_ \/ __| |/ _ \| '_ \
| |___ >  <| ||  __/ | | \__ \ | (_) | | | |
|_____/_/\_\\__\___|_| |_|___/_|\___/|_| |_|

BANNER
    echo -e "${NC}"
    echo -e "  ${BOLD}Gateway API Inference Extension Installer${NC} v${SCRIPT_VERSION}"
    echo ""
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --cleanup)
                CLEANUP_MODE=true
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    print_banner

    if [[ "$CLEANUP_MODE" == true ]]; then
        do_cleanup
        exit 0
    fi

    # Fetch latest release and set URLs (per official guide)
    set_release_urls

    if [[ "$DRY_RUN" == true ]]; then
        log_warn "DRY RUN - commands will be printed but not executed."
        echo ""
    fi

    # Run installation steps
    step_prerequisites
    step_cluster_setup
    step_loadbalancer
    step_gateway_api_crds
    step_model_server
    step_inference_crds
    step_gateway_controller
    step_deploy_gateway
    step_deploy_inferencepool
    step_verify_status
    step_inference_objective
    step_test
    step_bbr

    print_summary
    log_success "Installation complete!"
}

main "$@"
