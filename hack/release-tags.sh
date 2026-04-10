#!/usr/bin/env bash

# Copyright 2026 The Kubernetes Authors.
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

set -euo pipefail

: "${MAJOR:?MAJOR must be set}"
: "${MINOR:?MINOR must be set}"
: "${PATCH:?PATCH must be set}"

REMOTE="${REMOTE:-origin}"
PUSH="${PUSH:-true}"

if [[ -n "${RC-}" ]]; then
  RELEASE_TAG="v${MAJOR}.${MINOR}.${PATCH}-rc.${RC}"
  ROOT_MESSAGE="Gateway API Inference Extension ${RELEASE_TAG} Release Candidate"
  CONFORMANCE_MESSAGE="Gateway API Inference Extension Conformance ${RELEASE_TAG} Release Candidate"
else
  RELEASE_TAG="v${MAJOR}.${MINOR}.${PATCH}"
  ROOT_MESSAGE="Gateway API Inference Extension ${RELEASE_TAG} Release"
  CONFORMANCE_MESSAGE="Gateway API Inference Extension Conformance ${RELEASE_TAG} Release"
fi

CONFORMANCE_TAG="conformance/${RELEASE_TAG}"

create_tag() {
  local tag="$1"
  local message="$2"

  if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
    echo "error: tag ${tag} already exists" >&2
    exit 1
  fi

  echo "Creating signed tag ${tag}"
  git tag -s -a "${tag}" -m "${message}"
}

create_tag "${RELEASE_TAG}" "${ROOT_MESSAGE}"
create_tag "${CONFORMANCE_TAG}" "${CONFORMANCE_MESSAGE}"

if [[ "${PUSH}" == "false" ]]; then
  echo "Skipping push because PUSH=false"
  exit 0
fi

echo "Pushing ${RELEASE_TAG} and ${CONFORMANCE_TAG} to ${REMOTE}"
git push --atomic "${REMOTE}" "${RELEASE_TAG}" "${CONFORMANCE_TAG}"
