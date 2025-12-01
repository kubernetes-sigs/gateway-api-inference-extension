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
  RELEASE_TAG="v${MAJOR}.${MINOR}.0-rc.${RC}"
fi

# The vLLM image versions
# The GPU image is from https://hub.docker.com/r/vllm/vllm-openai/tags
VLLM_GPU="${VLLM_GPU:-0.10.0}"
# The CPU image is from https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo
VLLM_CPU="${VLLM_CPU:-0.10.0}"
# The sim image is from https://github.com/llm-d/llm-d-inference-sim/pkgs/container/llm-d-inference-sim
VLLM_SIM="${VLLM_SIM:-0.3.2-fix}"

echo "Using release tag: ${RELEASE_TAG}"
echo "Using vLLM GPU image version: ${VLLM_GPU}"
echo "Using vLLM CPU image version: ${VLLM_CPU}"
echo "Using vLLM Simulator image version: ${VLLM_SIM}"

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
# This regex matches any version in the form v<MAJOR>.<MINOR>.0-rc[.]?<number>
sed -i.bak -E "s|(refs/tags/)v[0-9]+\.[0-9]+\.0-rc\.?[0-9]+|\1${RELEASE_TAG}|g" "$README"
sed -i.bak -E "s|(releases/download/)v[0-9]+\.[0-9]+\.0-rc\.?[0-9]+|\1${RELEASE_TAG}|g" "$README"

# Replace the CRD installation line: change "kubectl apply -k" to "kubectl apply -f" with the proper URL
sed -i.bak "s|kubectl apply -k https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd|kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${RELEASE_TAG}/manifests.yaml|g" "$README"

# -----------------------------------------------------------------------------
# Update image references
# -----------------------------------------------------------------------------
#TODO: Put all helm values files into an array to loop over
EPP_HELM="config/charts/inferencepool/values.yaml"
BBR_HELM="config/charts/body-based-routing/values.yaml"
CONFORMANCE_MANIFESTS="conformance/resources/base.yaml"
echo "Updating ${EPP_HELM}, ${BBR_HELM}, and ${CONFORMANCE_MANIFESTS} ..."

# Update the container tag.
sed -i.bak -E "s|(tag: )[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$EPP_HELM"
sed -i.bak -E "s|(tag: )[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$BBR_HELM"
# Update epp
sed -i.bak -E "s|(gateway-api-inference-extension/epp:)[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$CONFORMANCE_MANIFESTS"
# Update the container image pull policy.
sed -i.bak '/us-central1-docker.pkg.dev\/k8s-staging-images\/gateway-api-inference-extension\/epp/{n;s/Always/IfNotPresent/;}' "$CONFORMANCE_MANIFESTS"

# Update the container registry.
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$EPP_HELM"
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$BBR_HELM"
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$CONFORMANCE_MANIFESTS"

# -----------------------------------------------------------------------------
# Update vLLM deployment manifests
# -----------------------------------------------------------------------------
VLLM_GPU_DEPLOY="config/manifests/vllm/gpu-deployment.yaml"
echo "Updating ${VLLM_GPU_DEPLOY} ..."

# Update the vLLM GPU image version
sed -i.bak -E "s|(vllm/vllm-openai:)[^\"[:space:]]+|\1v${VLLM_GPU}|g" "$VLLM_GPU_DEPLOY"

# Also change the imagePullPolicy from Always to IfNotPresent on lines containing the vLLM image.
sed -i.bak '/vllm\/vllm-openai/{n;s/Always/IfNotPresent/;}' "$VLLM_GPU_DEPLOY"

VLLM_CPU_DEPLOY="config/manifests/vllm/cpu-deployment.yaml"
echo "Updating ${VLLM_CPU_DEPLOY} ..."

# Update the vLLM CPU image version
sed -i.bak -E "s|(q9t5s3a7/vllm-cpu-release-repo:)[^\"[:space:]]+|\1v${VLLM_CPU}|g" "$VLLM_CPU_DEPLOY"

# Also change the imagePullPolicy from Always to IfNotPresent on lines containing the vLLM CPU image.
sed -i.bak '/q9t5s3a7\/vllm-cpu-release-repo/{n;s/Always/IfNotPresent/;}' "$VLLM_CPU_DEPLOY"

VLLM_SIM_DEPLOY="config/manifests/vllm/sim-deployment.yaml"
echo "Updating ${VLLM_SIM_DEPLOY} ..."

# Update the vLLM Simulator image version
sed -i.bak -E "s|(llm-d/llm-d-inference-sim:)[^\"[:space:]]+|\1v${VLLM_SIM}|g" "$VLLM_SIM_DEPLOY"

# Also change the imagePullPolicy from Always to IfNotPresent on lines containing the vLLM image.
sed -i.bak '/llm-d\/llm-d-inference-sim/{n;s/Always/IfNotPresent/;}' "$VLLM_SIM_DEPLOY"

# Update the container tag for lora-syncer in vLLM CPU and GPU deployment manifests.
sed -i.bak -E "s|(gateway-api-inference-extension/lora-syncer:)[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$VLLM_GPU_DEPLOY" "$VLLM_CPU_DEPLOY"
# Update the container image pull policy for lora-syncer in vLLM CPU and GPU deployment manifests.
sed -i.bak '/us-central1-docker.pkg.dev\/k8s-staging-images\/gateway-api-inference-extension\/lora-syncer/{n;s/Always/IfNotPresent/;}' "$VLLM_GPU_DEPLOY" "$VLLM_CPU_DEPLOY"

# Update the container registry for lora-syncer in vLLM CPU and GPU deployment manifests.
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$VLLM_GPU_DEPLOY" "$VLLM_CPU_DEPLOY"

# Update IGW_CHART_VERSION in quickstart guide to match the current release tag
GUIDES_INDEX="site-src/guides/index.md"
OLD_VERSION=$(grep -oE 'export IGW_CHART_VERSION=v[0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?' "$GUIDES_INDEX" | head -n1 | cut -d'=' -f2)
sed -i.bak -E "s/${OLD_VERSION}/${RELEASE_TAG}/g" "$GUIDES_INDEX"
# -----------------------------------------------------------------------------
# Stage the changes
# -----------------------------------------------------------------------------
echo "Staging $VERSION_FILE $UPDATED_CRD $README $EPP_HELM $BBR_HELM $CONFORMANCE_MANIFESTS $VLLM_GPU_DEPLOY $VLLM_CPU_DEPLOY $VLLM_SIM_DEPLOY $GUIDES_INDEX files..."
git add $VERSION_FILE $UPDATED_CRD $README $EPP_HELM $BBR_HELM $CONFORMANCE_MANIFESTS $VLLM_GPU_DEPLOY $VLLM_CPU_DEPLOY $VLLM_SIM_DEPLOY $GUIDES_INDEX

# -----------------------------------------------------------------------------
# Cleanup backup files and finish
# -----------------------------------------------------------------------------
echo "Cleaning up temporary backup files..."
find . -name "*.bak" -delete

echo "Release quickstart update complete."
