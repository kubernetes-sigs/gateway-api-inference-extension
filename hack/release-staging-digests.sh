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

#!/usr/bin/env bash
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  echo "error: gcloud is required" >&2
  exit 1
fi

: "${MAJOR:?MAJOR must be set}"
: "${MINOR:?MINOR must be set}"
: "${PATCH:?PATCH must be set}"

if [[ -n "${RC-}" ]]; then
  RELEASE_TAG="v${MAJOR}.${MINOR}.${PATCH}-rc.${RC}"
else
  RELEASE_TAG="v${MAJOR}.${MINOR}.${PATCH}"
fi

PROJECT="${PROJECT:-k8s-staging-images}"
LOCATION="${LOCATION:-us-central1}"
REPOSITORY="${REPOSITORY:-gateway-api-inference-extension}"
LIMIT="${LIMIT:-5000}"

images=(
  "charts/inferencepool"
  "charts/body-based-routing"
  "charts/standalone"
  "epp"
  "bbr"
  "latency-prediction-server"
  "latency-training-server"
)

find_digest() {
  local image="$1"
  local resource="${LOCATION}-docker.pkg.dev/${PROJECT}/${REPOSITORY}/${image}"
  local rows
  rows="$(
    gcloud artifacts docker images list "${resource}" \
      --include-tags \
      --limit="${LIMIT}" \
      --format='value(version,tags)' 2>/dev/null | grep '^sha256:' || true
  )"

  local digest tags tag
  while IFS=$'\t' read -r digest tags; do
    [[ -z "${digest}" || -z "${tags}" ]] && continue
    IFS=',' read -r -a tag_list <<< "${tags}"
    for tag in "${tag_list[@]}"; do
      if [[ "${tag}" == "${RELEASE_TAG}" ]]; then
        echo "${digest}"
        return 0
      fi
    done
  done <<< "${rows}"

  return 1
}

echo "Release tag: ${RELEASE_TAG}"
echo "Registry: ${LOCATION}-docker.pkg.dev/${PROJECT}/${REPOSITORY}"
echo

digests=()
missing=()

for image in "${images[@]}"; do
  digest="$(find_digest "${image}" || true)"
  if [[ -n "${digest}" ]]; then
    printf "  %-28s %s\n" "${image}" "${digest}"
    digests+=("${digest}")
  else
    printf "  %-28s %s\n" "${image}" "MISSING"
    digests+=("")
    missing+=("${image}")
  fi
done

echo
if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "Missing digest(s) for ${RELEASE_TAG}: ${missing[*]}" >&2
  echo "Wait for Artifact Registry publication to complete and rerun." >&2
  exit 1
fi

echo "k8s.io images.yaml entries for ${RELEASE_TAG}:"
for i in "${!images[@]}"; do
  printf "  - %s => \"%s\": [\"%s\"]\n" "${images[$i]}" "${digests[$i]}" "${RELEASE_TAG}"
done
