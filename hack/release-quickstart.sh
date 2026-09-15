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
# Match the EPP image at either registry: on `main` it is pinned to the staging
# registry, but on a release branch a previous release already rewrote it to
# registry.k8s.io. Matching only the staging prefix makes every patch release
# after the first a silent no-op, leaving the manifests on the older tag.
CONFORMANCE_EPP_IMAGE_REGEX="(us-central1-docker\.pkg\.dev/k8s-staging-images|registry\.k8s\.io)/gateway-api-inference-extension/lwepp"
CONFORMANCE_EPP_RELEASE_IMAGE="registry.k8s.io/gateway-api-inference-extension/lwepp"
echo "Updating ${CONFORMANCE_MANIFESTS} ..."

# Point the conformance EPP at the promoted release image. `#` is the delimiter
# because the regex above contains `|`.
sed -i.bak -E "s#${CONFORMANCE_EPP_IMAGE_REGEX}:[^\"[:space:]]+#${CONFORMANCE_EPP_RELEASE_IMAGE}:${RELEASE_TAG}#g" "$CONFORMANCE_MANIFESTS"
# A released image is immutable, so it never needs re-pulling.
sed -i.bak -E "\#${CONFORMANCE_EPP_RELEASE_IMAGE}:#{n;s/Always/IfNotPresent/;}" "$CONFORMANCE_MANIFESTS"

# Fail loudly rather than tagging a release whose manifests point at an older EPP.
# This deliberately matches every EPP reference regardless of registry, rather
# than reusing the substitution pattern above: a check that only looks where the
# substitution already ran cannot catch a reference the substitution missed.
if grep -oE "[^\"[:space:]]*gateway-api-inference-extension/lwepp[^\"[:space:]]*" "$CONFORMANCE_MANIFESTS" |
  grep -vx "${CONFORMANCE_EPP_RELEASE_IMAGE}:${RELEASE_TAG}"; then
  echo "ERROR: the EPP references above are not pinned to ${CONFORMANCE_EPP_RELEASE_IMAGE}:${RELEASE_TAG}" >&2
  exit 1
fi


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
