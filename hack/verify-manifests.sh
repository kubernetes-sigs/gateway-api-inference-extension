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

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_ROOT=$(dirname "${BASH_SOURCE}")/..
GATEWAY_API_VERSION="${GATEWAY_API_VERSION:-v1.3.0}"
GKE_GATEWAY_API_VERSION="${GKE_GATEWAY_API_VERSION:-v1.4.0}"
ISTIO_VERSION="${ISTIO_VERSION:-1.26.2}"
TEMP_DIR=$(mktemp -d)

cleanup() {
  rm -rf "${TEMP_DIR}" || true
}
trap cleanup EXIT

fetch_crds() {
  local url="$1"
  curl -sL "${url}" -o "${TEMP_DIR}/$(basename "${url}")"
}

main() {
  cp ${SCRIPT_ROOT}/config/crd/bases/* "${TEMP_DIR}/"

  # Download external CRDs for validation
  fetch_crds "https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/refs/tags/${GATEWAY_API_VERSION}/config/crd/standard/gateway.networking.k8s.io_gatewayclasses.yaml"
  fetch_crds "https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/refs/tags/${GATEWAY_API_VERSION}/config/crd/standard/gateway.networking.k8s.io_gateways.yaml"
  fetch_crds "https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/refs/tags/${GATEWAY_API_VERSION}/config/crd/standard/gateway.networking.k8s.io_httproutes.yaml"
  fetch_crds "https://raw.githubusercontent.com/GoogleCloudPlatform/gke-gateway-api/refs/tags/${GKE_GATEWAY_API_VERSION}/config/crd/networking.gke.io_gcpbackendpolicies.yaml"
  fetch_crds "https://raw.githubusercontent.com/GoogleCloudPlatform/gke-gateway-api/refs/tags/${GKE_GATEWAY_API_VERSION}/config/crd/networking.gke.io_healthcheckpolicies.yaml"
  fetch_crds "https://raw.githubusercontent.com/istio/istio/refs/tags/${ISTIO_VERSION}/manifests/charts/base/files/crd-all.gen.yaml"

  make kubectl-validate

  ${SCRIPT_ROOT}/bin/kubectl-validate "${TEMP_DIR}"
  ${SCRIPT_ROOT}/bin/kubectl-validate "${SCRIPT_ROOT}/config/manifests" --local-crds "${TEMP_DIR}"
}

main
