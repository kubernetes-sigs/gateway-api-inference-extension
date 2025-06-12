#!/usr/bin/env bash

set -euox pipefail

if kubectl config current-context >/dev/null 2>&1; then
  echo "Active kubecontext found. Running Go e2e tests in ./epp..."
else
  echo "No active kubecontext found. Creating kind cluster..."
  kind create cluster --name inference-e2e
  KIND_CLUSTER=inference-e2e make image-kind
  # Add the Gateway API CRDs
  VERSION=v0.3.0
  echo "Kind cluster created. Running Go e2e tests in ./epp..."
fi

MANIFEST_PATH=$PROJECT_DIR/$E2E_MANIFEST_PATH go test ./test/e2e/epp/ -v -ginkgo.v
