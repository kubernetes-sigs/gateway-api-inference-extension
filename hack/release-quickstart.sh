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
# Update config/manifests/ext_proc.yaml
# -----------------------------------------------------------------------------
EXT_PROC="config/manifests/ext_proc.yaml"
echo "Updating ${EXT_PROC} ..."

# Update the EPP container tag.
sed -i.bak -E "s|(us-central1-docker\.pkg\.dev/k8s-staging-images/gateway-api-inference-extension/epp:)[^\"[:space:]]+|\1${RELEASE_TAG}|g" "$EXT_PROC"

# Update the EPP container image pull policy.
sed -i.bak '/us-central1-docker.pkg.dev\/k8s-staging-images\/gateway-api-inference-extension\/epp/ { n; s/Always/IfNotPresent/ }' "$EXT_PROC"

# Update the EPP container registry.
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$EXT_PROC"

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
echo "Staging $README $EXT_PROC $VLLM_DEPLOY files..."
git add $README $EXT_PROC $VLLM_DEPLOY

# -----------------------------------------------------------------------------
# Cleanup backup files and finish
# -----------------------------------------------------------------------------
echo "Cleaning up temporary backup files..."
find . -name "*.bak" -delete

echo "Release quickstart update complete."
