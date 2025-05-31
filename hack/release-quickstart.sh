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
# MAJOR and MINOR are required (defaults provided here if not already set)
MAJOR="${MAJOR:-0}"
MINOR="${MINOR:-1}"

# If RC is defined (non-empty) then include the rc suffix; otherwise omit it.
if [[ -z "${RC-}" ]]; then
  RELEASE_TAG="v${MAJOR}.${MINOR}.0"
else
  RELEASE_TAG="v${MAJOR}.${MINOR}.0-rc.${RC}"
fi

# vLLM image version (default to 0.7.2 if not defined)
VLLM="${VLLM:-0.7.2}"

echo "Using release tag: ${RELEASE_TAG}"
echo "Using vLLM image version: ${VLLM}"

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
EPP="config/manifests/inferencepool-resources.yaml"
#TODO: Put all helm values files into an array to loop over
EPP_HELM="config/charts/inferencepool/values.yaml"
BBR_HELM="config/charts/body-based-routing/values.yaml"
echo "Updating ${EPP} & ${EPP_HELM} ..."

# Update the container tag.
sed -i.bak -E "s|(us-central1-docker\.pkg\.dev/k8s-staging-images/gateway-api-inference-extension/epp:)[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$EPP"
sed -i.bak -E "s|(tag: )[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$EPP_HELM"
sed -i.bak -E "s|(tag: )[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$BBR_HELM"

# Update the container image pull policy.
sed -i.bak '/us-central1-docker.pkg.dev\/k8s-staging-images\/gateway-api-inference-extension\/epp/ { n; s/Always/IfNotPresent/ }' "$EPP"

# Update the container registry.
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$EPP"
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$EPP_HELM"
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$BBR_HELM"

# -----------------------------------------------------------------------------
# Update config/manifests/vllm/gpu-deployment.yaml
# -----------------------------------------------------------------------------
VLLM_DEPLOY="config/manifests/vllm/gpu-deployment.yaml"
echo "Updating ${VLLM_DEPLOY} ..."

# Update the vLLM image version
sed -i.bak -E "s|(vllm/vllm-openai:)[^\"[:space:]]+|\1v${VLLM}|g" "$VLLM_DEPLOY"

# Also change the imagePullPolicy from Always to IfNotPresent on lines containing the vLLM image.
sed -i.bak '/vllm\/vllm-openai/ { n; s/Always/IfNotPresent/ }' "$VLLM_DEPLOY"

# -----------------------------------------------------------------------------
# Stage the changes
# -----------------------------------------------------------------------------
echo "Staging $README $EPP $EPP_HELM $BBR_HELM $VLLM_DEPLOY files..."
git add $README $EPP $EPP_HELM $BBR_HELM $VLLM_DEPLOY

# -----------------------------------------------------------------------------
# Cleanup backup files and finish
# -----------------------------------------------------------------------------
echo "Cleaning up temporary backup files..."
find . -name "*.bak" -delete

echo "Release quickstart update complete."
