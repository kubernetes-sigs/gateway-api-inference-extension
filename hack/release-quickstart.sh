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

echo "Using release tag: ${RELEASE_TAG}"

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
CONFORMANCE_MANIFESTS="conformance/resources/base.yaml"
CONFORMANCE_EPP_STAGING_IMAGE="us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/lwepp"
echo "Updating ${CONFORMANCE_MANIFESTS} ..."

# Update the conformance EPP image from the staging `main` tag to the release tag.
sed -i.bak -E "s|${CONFORMANCE_EPP_STAGING_IMAGE}:[^\"[:space:]]+|${CONFORMANCE_EPP_STAGING_IMAGE}:${RELEASE_TAG}|g" "$CONFORMANCE_MANIFESTS"
# Update the container image pull policy.
sed -i.bak '/us-central1-docker.pkg.dev\/k8s-staging-images\/gateway-api-inference-extension\/lwepp/{n;s/Always/IfNotPresent/;}' "$CONFORMANCE_MANIFESTS"

# Update the container registry.
sed -i.bak -E "s|us-central1-docker\.pkg\.dev/k8s-staging-images|registry.k8s.io|g" "$CONFORMANCE_MANIFESTS"


# -----------------------------------------------------------------------------
# Stage the changes
# -----------------------------------------------------------------------------
echo "Staging $VERSION_FILE $UPDATED_CRD $README $CONFORMANCE_GOMOD $CONFORMANCE_GOSUM $CONFORMANCE_MANIFESTS files..."
git add "$VERSION_FILE" "$UPDATED_CRD" "$README" "$CONFORMANCE_GOMOD" "$CONFORMANCE_GOSUM" "$CONFORMANCE_MANIFESTS"

# -----------------------------------------------------------------------------
# Cleanup backup files and finish
# -----------------------------------------------------------------------------
echo "Cleaning up temporary backup files..."
find . -name "*.bak" -delete

echo "Release quickstart update complete."
