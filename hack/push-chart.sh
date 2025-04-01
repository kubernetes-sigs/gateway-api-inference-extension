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

set -o errexit
set -o nounset
set -o pipefail

DEST_CHART_DIR=${DEST_CHART_DIR:-bin/}

EXTRA_TAG=${EXTRA_TAG:-$(git branch --show-current)} 
CHART_VERSION=${CHART_VERSION:-"v0"}

STAGING_IMAGE_REGISTRY=${STAGING_IMAGE_REGISTRY:-us-central1-docker.pkg.dev/k8s-staging-images}
IMAGE_REGISTRY=${IMAGE_REGISTRY:-${STAGING_IMAGE_REGISTRY}/gateway-api-inference-extension}
HELM_CHART_REPO=${HELM_CHART_REPO:-${STAGING_IMAGE_REGISTRY}/gateway-api-inference-extension/charts}
CHART=${CHART:-inferencepool}

HELM=${HELM:-./bin/helm}

readonly semver_regex='^v([0-9]+)(\.[0-9]+){1,2}(-rc.[0-9]+)?$'

chart_version=${CHART_VERSION}
if [[ ${EXTRA_TAG} =~ ${semver_regex} ]]
then
  ${YQ} -i '.inferenceExtension.image.tag=strenv(EXTRA_TAG)' config/charts/${CHART}/values.yaml
  chart_version=${EXTRA_TAG}
fi

# Create the package
${HELM} package --version "${chart_version}" --app-version "${chart_version}" "config/charts/${CHART}" -d "${DEST_CHART_DIR}"

# Push the package
echo "pushing chart to ${HELM_CHART_REPO}"
${HELM} push "bin/${CHART}-${chart_version}.tgz" "oci://${HELM_CHART_REPO}"
