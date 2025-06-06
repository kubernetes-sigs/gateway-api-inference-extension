#!/bin/bash

# Copyright 2021 The Kubernetes Authors.
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

readonly SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE}")"/../.. && pwd)"
cd $SCRIPT_ROOT

# -----------------------------------------------------------------------------
# Version extraction from git branch
# -----------------------------------------------------------------------------
get_version_from_branch() {
    # Get current branch name
    local branch_name
    branch_name=$(git rev-parse --abbrev-ref HEAD)
    
    # If the branch is main, set the version to main (and not a MAJOR.MINOR version)
    if [[ "$branch_name" == "main" ]]; then
        VERSION="main"
    
    # Extract version from branch name (e.g., release-0.3 -> 0.3)
    elif [[ $branch_name =~ release-([0-9]+)\.([0-9]+) ]]; then
        MAJOR="${BASH_REMATCH[1]}"
        MINOR="${BASH_REMATCH[2]}"
        VERSION="${MAJOR}.${MINOR}"
    else
        echo "Error: Could not extract version from branch name: $branch_name"
        echo "Expected branch name format: 'release-X.Y' or 'main'"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Check if version should be marked as latest (ignore release candidates "-rc" or any hyphenated suffix)
# -----------------------------------------------------------------------------
is_latest_version() {
    # 1) List all tags matching semver-ish (vX.Y[.Z] or X.Y[.Z]), sort by version descending,
    # and pick the very first one.
    local latest_tag
    latest_tag=$(git tag --list 'v[0-9]*.[0-9]*' --list '[0-9]*.[0-9]*' --sort=-v:refname | head -n1)

    if [[ -z "$latest_tag" ]]; then
        echo "Error: Could not find any semver‐style tags."
        return 1
    fi

    # 2) Strip leading 'v', then drop anything after the first hyphen (e.g. "0.3.0-rc.1" → "0.3.0")
    local bare="${latest_tag#v}"      # remove leading "v" if present
    bare="${bare%%-*}"                # drop "-<anything>" (so "0.3.0-rc.1" → "0.3.0")

    # 3) Now extract MAJOR and MINOR from e.g. "0.3.0" or "2.5"
    if [[ "$bare" =~ ^([0-9]+)\.([0-9]+)(\.[0-9]+)?$ ]]; then
        local latest_major="${BASH_REMATCH[1]}"
        local latest_minor="${BASH_REMATCH[2]}"
    else
        echo "Error: Could not parse version from latest tag: ${latest_tag} (bare='${bare}')"
        return 1
    fi

    # 4) Compare numeric MAJOR/MINOR for exact match
    if (( MAJOR == latest_major && MINOR == latest_minor )); then
        return 0
    fi
    return 1
}

# Get version from current branch
get_version_from_branch

# -----------------------------------------------------------------------------
# Environment variables (defaults)
# -----------------------------------------------------------------------------
# VERSION is now set by get_version_from_branch()

# Wrap sed to deal with GNU and BSD sed flags.
run::sed() {
    local -r vers="$(sed --version < /dev/null 2>&1 | grep -q GNU && echo gnu || echo bsd)"
    case "$vers" in
        gnu) sed -i "$@" ;;
        *) sed -i '' "$@" ;;
    esac
}

# -----------------------------------------------------------------------------
# Build versioned docs
# -----------------------------------------------------------------------------

# Generate API docs

GOPATH=${GOPATH:-$(go env GOPATH)}

# "go env" doesn't print anything if GOBIN is the default, so we
# have to manually default it.
GOBIN=${GOBIN:-$(go env GOBIN)}
GOBIN=${GOBIN:-${GOPATH}/bin}

echo $GOBIN

go install github.com/elastic/crd-ref-docs

${GOBIN}/crd-ref-docs \
    --source-path=${PWD}/api \
    --config=crd-ref-docs.yaml \
    --renderer=markdown \
    --output-path=${PWD}/site-src/reference/spec.md

# Deploy docs with mike
echo "Deploying docs for version ${VERSION}"
if [[ "$VERSION" == "main" ]]; then
    echo "Deploying docs as 'main'."
    mike deploy --push --branch docs main
elif is_latest_version; then
    echo "This version will be deployed and marked as 'latest'."
    mike deploy --push --update-aliases --alias-type=copy --branch docs "${VERSION}" latest
else
    echo "This version will be deployed, but not marked as 'latest'."
    mike deploy --push --branch docs "${VERSION}"
fi

# Always set the default version to 'latest'
echo "Setting default version to 'latest'."
mike set-default --branch docs latest
