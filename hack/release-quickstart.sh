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

#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Environment variables (defaults)
# -----------------------------------------------------------------------------
# MAJOR, MINOR, and PATCH are required (defaults provided here if not already set)
MAJOR="${MAJOR:-0}"
MINOR="${MINOR:-1}"
PATCH="${PATCH:-0}"

# If RC is defined (non-empty) then include the rc suffix; otherwise omit it.
if [[ -z "${RC-}" ]]; then
  RELEASE_TAG="v${MAJOR}.${MINOR}.${PATCH}"
else
  RELEASE_TAG="v${MAJOR}.${MINOR}.${PATCH}-rc.${RC}"
fi

# The vLLM image versions
# The GPU image is from https://hub.docker.com/r/vllm/vllm-openai/tags
VLLM_GPU="${VLLM_GPU:-0.18.1}"
# The CPU image is from https://hub.docker.com/r/vllm/vllm-openai-cpu/tags
VLLM_CPU="${VLLM_CPU:-0.18.1}"
# The sim image is from https://github.com/llm-d/llm-d-inference-sim/pkgs/container/llm-d-inference-sim
VLLM_SIM="${VLLM_SIM:-0.8.2}"
# The agentgateway image is from https://github.com/agentgateway/agentgateway/pkgs/container/agentgateway
AGENTGATEWAY_TAG="${AGENTGATEWAY_TAG:-${RELEASE_TAG}}"

echo "Using release tag: ${RELEASE_TAG}"
echo "Using vLLM GPU image version: ${VLLM_GPU}"
echo "Using vLLM CPU image version: ${VLLM_CPU}"
echo "Using vLLM Simulator image version: ${VLLM_SIM}"
echo "Using agentgateway image tag: ${AGENTGATEWAY_TAG}"

# -----------------------------------------------------------------------------
# Update version/version.go and generating CRDs with new version annotations
# -----------------------------------------------------------------------------
VERSION_FILE="version/version.go"
echo "Updating ${VERSION_FILE} ..."

# Replace bundleVersion in version.go
# This regex finds the line with "BundleVersion" and replaces the string within the quotes.
sed -i.bak -E "s|( *BundleVersion = \")[^\"]+(\")|\1${RELEASE_TAG}\2|g" "$VERSION_FILE"

UPDATED_CRD="config/crd/"
echo "Generating CRDs with new annotations in $UPDATED_CRD"
go run ./pkg/generator
echo "Generated CRDs with new annotations in $UPDATED_CRD"

# -----------------------------------------------------------------------------
# Update pkg/README.md
# -----------------------------------------------------------------------------
README="pkg/README.md"
echo "Updating ${README} ..."

# Replace URLs that refer to a tag (whether via refs/tags or releases/download)
# This regex matches any version in the form v<MAJOR>.<MINOR>.<PATCH>-rc[.]?<number>
sed -i.bak -E "s|(refs/tags/)v[0-9]+\.[0-9]+\.[0-9]+-rc\.?[0-9]+|\1${RELEASE_TAG}|g" "$README"
sed -i.bak -E "s|(releases/download/)v[0-9]+\.[0-9]+\.[0-9]+-rc\.?[0-9]+|\1${RELEASE_TAG}|g" "$README"

# Replace the CRD installation line: change "kubectl apply -k" to "kubectl apply -f" with the proper URL
sed -i.bak "s|kubectl apply -k https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd|kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${RELEASE_TAG}/manifests.yaml|g" "$README"

# -----------------------------------------------------------------------------
# Update the conformance module dependency
# -----------------------------------------------------------------------------
CONFORMANCE_GOMOD="conformance/go.mod"
CONFORMANCE_GOSUM="conformance/go.sum"
echo "Updating ${CONFORMANCE_GOMOD} and ${CONFORMANCE_GOSUM} ..."
(
  cd conformance
  go mod edit -require=sigs.k8s.io/gateway-api-inference-extension@"${RELEASE_TAG}"
)

# -----------------------------------------------------------------------------
# Update image references
# -----------------------------------------------------------------------------
#TODO: Put all helm values files into an array to loop over
EPP_HELM="config/charts/inferencepool/values.yaml"
LATENCY_ROUTING_HELM="config/charts/epplib/values.yaml"
BBR_HELM="config/charts/body-based-routing/values.yaml"
STANDALONE_HELM="config/charts/standalone/values.yaml"
CONFORMANCE_MANIFESTS="conformance/resources/base.yaml"
CONFORMANCE_EPP_STAGING_IMAGE="us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/epp"
AGENTGATEWAY_IMAGE_REGEX="cr\.agentgateway\.dev/agentgateway"
echo "Updating ${EPP_HELM}, ${LATENCY_ROUTING_HELM}, ${BBR_HELM}, ${STANDALONE_HELM} and ${CONFORMANCE_MANIFESTS} ..."

# Update the container tag.
sed -i.bak -E "s|(tag: )[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$EPP_HELM"
sed -i.bak -E "s|(tag: )[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$LATENCY_ROUTING_HELM"
sed -i.bak -E "s|(tag: )[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$BBR_HELM"
sed -i.bak -E "s|(tag: )[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$STANDALONE_HELM"
# Update the standalone agentgateway preset from the floating development tag
# to the stable tag selected for the release.
sed -i.bak -E "s|(${AGENTGATEWAY_IMAGE_REGEX}:)[^\"[:space:]]+|\1${AGENTGATEWAY_TAG}|g" "$STANDALONE_HELM"
# Update the conformance EPP image from the staging `main` tag to the release tag.
sed -i.bak -E "s|${CONFORMANCE_EPP_STAGING_IMAGE}:[^\"[:space:]]+|${CONFORMANCE_EPP_STAGING_IMAGE}:${RELEASE_TAG}|g" "$CONFORMANCE_MANIFESTS"
# Update the container image pull policy.
sed -i.bak '/us-central1-docker.pkg.dev\/k8s-staging-images\/gateway-api-inference-extension\/epp/{n;s/Always/IfNotPresent/;}' "$CONFORMANCE_MANIFESTS"

# Update the container registry.
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$EPP_HELM"
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$LATENCY_ROUTING_HELM"
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$BBR_HELM"
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$STANDALONE_HELM"
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$CONFORMANCE_MANIFESTS"

# -----------------------------------------------------------------------------
# Update vLLM deployment manifests
# -----------------------------------------------------------------------------
VLLM_GPU_DEPLOYS=(
  "config/manifests/vllm/gpu-deployment.yaml"
  "config/manifests/vllm/gpu-grpc-deployment.yaml"
  "config/manifests/vllm/gpu-prefix-cache-deployment.yaml"
)
echo "Updating ${VLLM_GPU_DEPLOYS[*]} ..."

for vllm_gpu_deploy in "${VLLM_GPU_DEPLOYS[@]}"; do
  # Replace the floating GPU tag with the latest semver tag for the release.
  sed -i.bak -E "s|(vllm/vllm-openai:)[^\"[:space:]]+|\1v${VLLM_GPU}|g" "$vllm_gpu_deploy"
  # Also change the imagePullPolicy from Always to IfNotPresent on lines containing the vLLM image.
  sed -i.bak '/vllm\/vllm-openai/{n;s/Always/IfNotPresent/;}' "$vllm_gpu_deploy"
done

VLLM_CPU_DEPLOY="config/manifests/vllm/cpu-deployment.yaml"
echo "Updating ${VLLM_CPU_DEPLOY} ..."

# Update the vLLM CPU image version
sed -i.bak -E "s|(vllm/vllm-openai-cpu:)[^\"[:space:]]+|\1v${VLLM_CPU}|g" "$VLLM_CPU_DEPLOY"

# Also change the imagePullPolicy from Always to IfNotPresent on lines containing the vLLM CPU image.
sed -i.bak '/vllm\/vllm-openai-cpu/{n;s/Always/IfNotPresent/;}' "$VLLM_CPU_DEPLOY"

VLLM_SIM_DEPLOYS=(
  "config/manifests/vllm/sim-deployment.yaml"
  "config/manifests/vllm/sim-grpc-deployment.yaml"
)
echo "Updating ${VLLM_SIM_DEPLOYS[*]} ..."

for vllm_sim_deploy in "${VLLM_SIM_DEPLOYS[@]}"; do
  # Replace the floating simulator tag with the latest semver tag for the release.
  sed -i.bak -E "s|(llm-d/llm-d-inference-sim:)[^\"[:space:]]+|\1v${VLLM_SIM}|g" "$vllm_sim_deploy"
  # Also change the imagePullPolicy from Always to IfNotPresent on lines containing the vLLM image.
  sed -i.bak '/llm-d\/llm-d-inference-sim/{n;s/Always/IfNotPresent/;}' "$vllm_sim_deploy"
done

# -----------------------------------------------------------------------------
# Stage the changes
# -----------------------------------------------------------------------------
echo "Staging $VERSION_FILE $UPDATED_CRD $README $CONFORMANCE_GOMOD $CONFORMANCE_GOSUM $EPP_HELM $LATENCY_ROUTING_HELM $BBR_HELM $STANDALONE_HELM $CONFORMANCE_MANIFESTS ${VLLM_GPU_DEPLOYS[*]} $VLLM_CPU_DEPLOY ${VLLM_SIM_DEPLOYS[*]} files..."
git add "$VERSION_FILE" "$UPDATED_CRD" "$README" "$CONFORMANCE_GOMOD" "$CONFORMANCE_GOSUM" "$EPP_HELM" "$LATENCY_ROUTING_HELM" "$BBR_HELM" "$STANDALONE_HELM" "$CONFORMANCE_MANIFESTS" "${VLLM_GPU_DEPLOYS[@]}" "$VLLM_CPU_DEPLOY" "${VLLM_SIM_DEPLOYS[@]}"

# -----------------------------------------------------------------------------
# Cleanup backup files and finish
# -----------------------------------------------------------------------------
echo "Cleaning up temporary backup files..."
find . -name "*.bak" -delete

echo "Release quickstart update complete."
