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

if kubectl config current-context >/dev/null 2>&1; then
  echo "Active kubecontext found. Running Go e2e tests in ./epp..."
else
  install_kind
  kind create cluster --name inference-e2e
  KIND_CLUSTER=inference-e2e make image-kind
  echo "Kind cluster created. Running Go e2e tests in ./epp..."
fi

go test ./test/e2e/epp/ -v -ginkgo.v
