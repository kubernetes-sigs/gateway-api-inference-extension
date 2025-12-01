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

set -euox pipefail

install_kind() {
  if ! command -v kind &>/dev/null; then
    echo "kind not found, installing..."
    [ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.29.0/kind-linux-amd64
    # For ARM64
    [ $(uname -m) = aarch64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.29.0/kind-linux-arm64
    chmod +x ./kind
    mv ./kind /usr/local/bin/kind
  else
    echo "kind is already installed."
  fi
}

if [ "$USE_KIND" = "true" ]; then
  install_kind # make sure kind cli is installed
  if ! kubectl config current-context >/dev/null 2>&1; then # if no active kind cluster found
    echo "No active kubecontext found. creating a kind cluster for running the tests..."
    kind create cluster --name inference-e2e
    KIND_CLUSTER=inference-e2e IMAGE_TAG=${E2E_IMAGE} make image-kind
  else 
    current_context=$(kubectl config current-context)
    current_kind_cluster="${current_context#kind-}"
    echo "Found an active kind cluster ${current_kind_cluster} for running the tests..."
    KIND_CLUSTER=${current_kind_cluster} IMAGE_TAG=${E2E_IMAGE} make image-kind
  fi 
else 
  # don't use kind. it's the caller responsibility to load the image into the cluster, we just run the tests.
  # this section is useful when one wants to run an official release or latest main against a cluster other than kind.
  if ! kubectl config current-context >/dev/null 2>&1; then # if no active cluster found
    echo "No active kubecontext found. exiting..."
    exit
  fi
fi

echo "Found an active cluster. Running Go e2e tests in ./epp..."
go test ./test/e2e/epp/ -v -ginkgo.v
