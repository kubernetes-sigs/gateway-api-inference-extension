---
name: New Release
about: Propose a new release
title: Release v0.x.0
labels: ''
assignees: ''

---

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Release Process](#release-process)
- [Announce the Release](#announce-the-release)
- [Final Steps](#final-steps)

## Introduction

This document defines the process for releasing Gateway API Inference Extension.

## Prerequisites

1. Permissions to push to the Gateway API Inference Extension repository.

2. Set the required environment variables based on the expected release number:

   ```shell
   export MAJOR=0
   export MINOR=1
   export PATCH=0
   export REMOTE=origin
   ```

3. If creating a release candidate, set the release candidate number.

   ```shell
   export RC=1
   ```

4. Refer to the [release-quickstart script][release-quickstart] for the default image tags used
   by the vLLM deployment manifests. If a newer [GPU][vllm-gpu-tag], [CPU][vllm-cpu-tag], or [Simulator][vllm-sim-tag]
   tag has been published, set the appropriate environment variable or update the script. For example:

   ```shell
   export VLLM_GPU=0.13.0
   export VLLM_CPU=0.9.3
   export VLLM_SIM=0.6.1
   ```

## Release Process

1. If needed, clone the Gateway API Inference Extension [repo][repo].

   ```shell
   git clone -o ${REMOTE} https://github.com/kubernetes-sigs/gateway-api-inference-extension.git
   ```

2. If you already have the repo cloned, ensure it’s up-to-date and your local branch is clean.

3. Release Branch Handling:
   - For a Release Candidate:
     Create a new release branch from the `main` branch. The branch should be named `release-${MAJOR}.${MINOR}`, for example, `release-0.1`:

     ```shell
     git checkout -b release-${MAJOR}.${MINOR}
     ```

   - For a Major, Minor or Patch Release:
     A release branch should already exist. In this case, check out the existing branch:

     ```shell
     git checkout -b release-${MAJOR}.${MINOR} ${REMOTE}/release-${MAJOR}.${MINOR}
     ```

4. Update release-specific content, generate release artifacts, pin the `conformance` module to the matching root-module version, and stage the changes.

   ```shell
   make release
   ```

5. Sign, commit, and push the changes to the Gateway API Inference Extension repo.

   For a release candidate:

    ```shell
    git commit -S -m "Updates artifacts for v${MAJOR}.${MINOR}.${PATCH}-rc.${RC} release"
    ```

   For a major, minor or patch release:

    ```shell
    git commit -S -m "Updates artifacts for v${MAJOR}.${MINOR}.${PATCH} release"
    ```

6. Push your release branch to the Gateway API Inference Extension remote.

    ```shell
    git push ${REMOTE} release-${MAJOR}.${MINOR}
    ```

7. Create and push signed tags for the repo and the `conformance` module.

   ```shell
   make release-tags
   ```

   This creates and pushes the following tags together:
   - `v${MAJOR}.${MINOR}.${PATCH}` or `v${MAJOR}.${MINOR}.${PATCH}-rc.${RC}`
   - `conformance/v${MAJOR}.${MINOR}.${PATCH}` or `conformance/v${MAJOR}.${MINOR}.${PATCH}-rc.${RC}`

   **Note:** A PGP key must be [registered] to your GitHub account.

8. Pushing the root tag triggers Prow to build and publish the container image to the [staging registry][].
9. Submit a PR against [k8s.io][] to add the staging image tag and SHA to [`k8s-staging-gateway-api-inference-extension/images.yaml`][yaml]. This will
    promote the image to the production registry, e.g. `registry.k8s.io/gateway-api-inference-extension/epp:v${MAJOR}.${MINOR}.${PATCH}`.
    1. Collect release digests from Artifact Registry:
       ```shell
       ./hack/release-staging-digests.sh
       ```
    2. Ensure the PR updates all release artifacts:
       - `charts/inferencepool`
       - `charts/body-based-routing`
       - `charts/standalone`
       - `epp`
       - `bbr`
       - `latency-prediction-server`
       - `latency-training-server`
    3. If an artifact does not yet have a section in `images.yaml` (for example a newly added chart), add a new section before adding the digest mapping.
    **Note:** Add a link to this issue when the PR is merged.
10. Test the steps in the tagged quickstart guide after the PR merges, for example: `https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v0.1.0-rc.1/pkg/README.md`.
11. Create a [new release][]:
    1. Choose the root tag that you created for the release, for example `v1.5.0`.
    2. Use the tag as the release title, i.e. `v0.1.0` refer to previous release for the content of the release body.
    3. Click "Generate release notes" and preview the release body.
    4. Click "Attach binaries by dropping them here or selecting them." and add the contents of the `artifacts` directory generated from `make release`.
    5. If this is a release candidate, select the "This is a pre-release" checkbox.
12. If you find any bugs in this process, create an [issue][].

## Announce the Release

Use the following steps to announce the release.

1. Send an announcement email to `wg-serving@kubernetes.io` and `dev@kubernetes.io` with the subject:

   ```shell
   [ANNOUNCE] Gateway API Inference Extension v${MAJOR}.${MINOR}.${PATCH} is released
   ```

2. Add a link to the release announcement in this issue. <!-- link to an example email once we have one -->

## Final Steps

1. Update docs in the `main` branch. <!-- link to example PR once we have one -->
2. Close this issue.

[repo]: https://github.com/kubernetes-sigs/gateway-api-inference-extension
[staging registry]: https://console.cloud.google.com/artifacts/docker/k8s-staging-images/us-central1/gateway-api-inference-extension/epp
[new release]: https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/new
[registered]: https://docs.github.com/en/authentication/managing-commit-signature-verification/checking-for-existing-gpg-keys
[k8s.io]: https://github.com/kubernetes/k8s.io
[yaml]: https://github.com/kubernetes/k8s.io/blob/main/registry.k8s.io/images/k8s-staging-gateway-api-inference-extension/images.yaml
[issue]: https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/new/choose
[vllm-gpu-tag]: https://hub.docker.com/r/vllm/vllm-openai/tags
[vllm-cpu-tag]: https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo
[vllm-sim-tag]: https://github.com/llm-d/llm-d-inference-sim/pkgs/container/llm-d-inference-sim
[release-quickstart]: https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/hack/release-quickstart.sh
