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

SCRIPT_ROOT=$(dirname "${BASH_SOURCE}")/..
# Read the first argument, default to "ci" if not provided
MODE=${1:-ci}

if [ "$MODE" == "local" ]; then
  # Local Mode: Permissive. Updates lock file automatically.
  DEP_CMD="update"
  echo "🔸 MODE: Local (Dev) - Using 'helm dependency update'"
else
  # CI/CD Mode (Default): Strict. Fails if lock file is out of sync.
  DEP_CMD="build"
  echo "🔹 MODE: CI/CD (Strict) - Using 'helm dependency build'"
fi

declare -A test_cases_inference_pool

# InferencePool Helm Chart test cases
test_cases_inference_pool["basic"]="--set inferencePool.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_inference_pool["gke-provider"]="--set provider.name=gke --set inferencePool.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_inference_pool["multiple-replicas"]="--set inferencePool.replicas=3 --set inferencePool.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_inference_pool["latency-predictor"]="--set inferenceExtension.latencyPredictor.enabled=true --set inferencePool.modelServers.matchLabels.app=llm-instance-gateway"

# Run the install command in case this script runs from a different bash
# source (such as in the verify-all script)
make helm-install

echo "Processing dependencies for inferencePool chart..."
${SCRIPT_ROOT}/bin/helm dependency ${DEP_CMD} ${SCRIPT_ROOT}/config/charts/inferencepool
if [ $? -ne 0 ]; then
  echo "Helm dependency ${DEP_CMD} failed."
  exit 1
fi

# Running tests cases
echo "Running helm template command for inferencePool chart..."
# Loop through the keys of the associative array
for key in "${!test_cases_inference_pool[@]}"; do
  echo "Running test: $key"
  ${SCRIPT_ROOT}/bin/helm template ${SCRIPT_ROOT}/config/charts/inferencepool ${test_cases_inference_pool[$key]} --output-dir="${SCRIPT_ROOT}/bin"
  if [ $? -ne 0 ]; then
    echo "Helm template command failed for test: $key"
    exit 1
  fi
done

declare -A test_cases_epp_standalone

# InferencePool Helm Chart test cases
test_cases_epp_standalone["basic"]="--set inferenceExtension.endpointsServer.endpointSelector='app=llm-instance-gateway'"
test_cases_epp_standalone["gke-provider"]="--set provider.name=gke --set inferenceExtension.endpointsServer.endpointSelector='app=llm-instance-gateway'"
test_cases_epp_standalone["latency-predictor"]="--set inferenceExtension.latencyPredictor.enabled=true --set inferenceExtension.endpointsServer.endpointSelector='app=llm-instance-gateway'"


echo "Processing dependencies for epp-standalone chart..."
${SCRIPT_ROOT}/bin/helm dependency ${DEP_CMD} ${SCRIPT_ROOT}/config/charts/epp-standalone
if [ $? -ne 0 ]; then
  echo "Helm dependency ${DEP_CMD} failed."
  exit 1
fi

# Running tests cases
echo "Running helm template command for epp-standalone chart..."
# Loop through the keys of the associative array
for key in "${!test_cases_epp_standalone[@]}"; do
  echo "Running test: $key"
  ${SCRIPT_ROOT}/bin/helm template ${SCRIPT_ROOT}/config/charts/epp-standalone ${test_cases_epp_standalone[$key]} --output-dir="${SCRIPT_ROOT}/bin"
  if [ $? -ne 0 ]; then
    echo "Helm template command failed for test: $key"
    exit 1
  fi
done

