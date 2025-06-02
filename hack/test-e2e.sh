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

# This script verifies end-to-end connectivity for an example inference extension test environment based on
# resources from the quickstart guide or e2e test framework. It can optionally launch a "curl" client pod to
# run these tests within the cluster.
#
# USAGE: ./hack/e2e-test.sh
#
# OPTIONAL ENVIRONMENT VARIABLES:
#   - TIME:     The duration (in seconds) for which the test will run. Defaults to 1 second.
#   - CURL_POD: If set to "true", the script will use a Kubernetes pod named "curl" for making requests.
#   - IP:       Override the detected IP address. If not provided, the script attempts to use a Gateway based on
#               the quickstart guide or an Envoy service IP based on the e2e test framework.
#   - PORT:     Override the detected port. If not provided, the script attempts to use a Gateway based on the
#               quickstart guide or an Envoy service IP based on the e2e test framework.
#
# WHAT THE SCRIPT DOES:
#   1. Determines if there is a Gateway named "inference-gateway" in the "default" namespace. If found, it extracts the IP
#      address and port from the Gateway's "llm-gw" listener. Otherwise, it falls back to the Envoy service in the "default" namespace.
#   2. Optionally checks for (or creates) a "curl" pod, ensuring it is ready to execute requests.
#   3. Loops for $TIME seconds, sending requests every 5 seconds to the /v1/completions endpoint to confirm successful connectivity.

set -euo pipefail

# Determine the directory of this script and build an absolute path to client.yaml.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CLIENT_YAML="$SCRIPT_DIR/../test/testdata/client.yaml"

# TIME is the amount of time, in seconds, to run the test.
TIME=${TIME:-1}
# Optionally use a client curl pod for executing the curl command.
CURL_POD=${CURL_POD:-false}

check_resource_exists() {
    local type=$1
    local name=$2
    local namespace=$3

    if kubectl get "$type" "$name" -n "$namespace" &>/dev/null; then
         return 0
    else
         return 1
    fi
}

check_pod_ready() {
    local pod_name=$1
    local namespace=$2
    # Check the Ready condition using jsonpath. Default to False if not found.
    local ready_status
    ready_status=$(kubectl get pod "$pod_name" -n "$namespace" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
    if [[ "$ready_status" == "True" ]]; then
        return 0
    else
        return 1
    fi
}

# Try to get the Gateway's IP and the port from the listener named "llm-gw" if it exists.
if check_resource_exists "gateway" "inference-gateway" "default"; then
    GATEWAY_IP=$(kubectl get gateway inference-gateway -n default -o jsonpath='{.status.addresses[0].value}')
    # Use JSONPath to select the port from the listener with name "http"
    GATEWAY_PORT=$(kubectl get gateway inference-gateway -n default -o jsonpath='{.spec.listeners[?(@.name=="http")].port}')
else
    GATEWAY_IP=""
    GATEWAY_PORT=""
fi

if [[ -n "$GATEWAY_IP" && -n "$GATEWAY_PORT" ]]; then
    echo "Using Gateway inference-gateway IP and port from listener 'llm-gw'."
    IP=${IP:-$GATEWAY_IP}
    PORT=${PORT:-$GATEWAY_PORT}
else
    echo "Gateway inference-gateway not found or missing IP/port. Falling back to Envoy service."
    # Ensure the Envoy service exists.
    if ! check_resource_exists "svc" "envoy" "default"; then
        echo "Error: Envoy service not found in namespace 'default'."
        exit 1
    fi
    IP=${IP:-$(kubectl get svc envoy -n default -o jsonpath='{.spec.clusterIP}')}
    PORT=${PORT:-$(kubectl get svc envoy -n default -o jsonpath='{.spec.ports[0].port}')}
fi

# Optionally verify that the curl pod exists and is ready.
if [[ "$CURL_POD" == "true" ]]; then
    if ! check_resource_exists "pod" "curl" "default"; then
        echo "Pod 'curl' not found in namespace 'default'. Applying client.yaml from $CLIENT_YAML..."
        kubectl apply -f "$CLIENT_YAML"
    fi
    echo "Waiting for pod 'curl' to be ready..."
    # Retry every 5 seconds for up to 30 seconds (6 attempts)
    for i in {1..6}; do
        if check_pod_ready "curl" "default"; then
            echo "Pod 'curl' is now ready."
            break
        fi
        echo "Retry attempt $i: Pod 'curl' not ready; waiting 5 seconds..."
        sleep 5
    done

    if ! check_pod_ready "curl" "default"; then
        echo "Error: Pod 'curl' is still not ready in namespace 'default' after 30 seconds."
        exit 1
    fi
fi

# Validate that we have a non-empty IP and PORT.
if [[ -z "$IP" ]]; then
    echo "Error: Unable to determine a valid IP from either Gateway or Envoy service."
    exit 1
fi

if [[ -z "$PORT" ]]; then
    echo "Error: Unable to determine a valid port from either Gateway or Envoy service."
    exit 1
fi

echo "Using IP: $IP"
echo "Using PORT: $PORT"

# Run the test for the specified duration.
end=$((SECONDS + TIME))
if [[ "$CURL_POD" == "true" ]]; then
    while [ $SECONDS -lt $end ]; do
        kubectl exec po/curl -- curl -i "$IP:$PORT/v1/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model": "food-review","prompt": "Write as if you were a critic: San Francisco","max_tokens": 100,"temperature": 0}'
        sleep 5
    done
else
    while [ $SECONDS -lt $end ]; do
        curl -i "$IP:$PORT/v1/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model": "food-review","prompt": "Write as if you were a critic: San Francisco","max_tokens": 100,"temperature": 0}'
        sleep 5
    done
fi
