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
# Check if version should be marked as latest
# -----------------------------------------------------------------------------
is_latest_version() {
    # Get the latest release tag, checking more releases to find the true latest
    local latest_tag
    latest_tag=$(gh release list --limit 20 --json tagName --jq '.[0].tagName')
    
    if [[ -z "$latest_tag" ]]; then
        echo "Error: Could not find any releases."
        return 1
    fi
    
    # Extract version from tag (handles both v0.2.0 and 0.2 formats)
    if [[ $latest_tag =~ ^v?([0-9]+)\.([0-9]+)(\.[0-9]+)?$ ]]; then
        local latest_major="${BASH_REMATCH[1]}"
        local latest_minor="${BASH_REMATCH[2]}"
        
        # Compare versions
        if [[ "$MAJOR" -gt "$latest_major" ]] || \
           ([[ "$MAJOR" -eq "$latest_major" ]] && [[ "$MINOR" -ge "$latest_minor" ]]); then
            return 0  # Current version is newer or equal
        fi
    else
        echo "Error: Could not parse version from latest tag: $latest_tag"
        return 1
    fi
    
    return 1  # Current version is older
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
make api-ref-docs

# Deploy docs with mike
echo "Deploying docs for version ${VERSION}"
if [[ "$VERSION" == "main" ]]; then
    echo "Deploying docs as 'main'."
    mike deploy --push --branch docs main
elif is_latest_version; then
    echo "This version will be deployed and marked as 'latest'."
    mike deploy --push --update-aliases --branch docs "${VERSION}" latest
else
    echo "This version will be deployed, but not marked as 'latest'."
    mike deploy --push --branch docs "${VERSION}"
fi
