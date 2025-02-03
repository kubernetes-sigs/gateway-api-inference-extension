---
name: New Release
about: Propose a new release
title: Release v0.x.0
labels: ''
assignees: ''

---

## Release Checklist
- [ ] For major or minor releases (v$MAJ.$MIN.0), create a new release branch.
  - [ ] A maintainer creates a vanilla release branch with
        `git branch release-$MAJ.$MIN main`
  - [ ] A maintainer pushes the new release branch with
        `git push release-$MAJ.$MIN` (e.g., release-0.1)
- [ ] Update things like README, deployment templates, docs, configuration, test/e2e flags.
      Submit a PR against the release branch.
- [ ] A maintainer [prepares a draft release](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases)
  - [ ] Write the change log into the draft release.
  - [ ] Run
      `make artifacts`
      to generate the artifacts and upload the files in the `artifacts` folder to the draft release.
- [ ] A maintainer creates a signed tag running (e.g., VERSION=v0.1.0)
     `git tag -s $VERSION`
      and inserts the changelog into the tag description.
      To perform this step, you need [a PGP key registered on github](https://docs.github.com/en/authentication/managing-commit-signature-verification/checking-for-existing-gpg-keys).
- [ ] A maintainer pushes the tag with
      `git push $VERSION`
  - Triggers prow to build and publish a staging container image
    [`us-central1-docker.pkg.dev/k8s-staging-images/llm-instance-gateway/epp:$VERSION`](https://us-central1-docker.pkg.dev/k8s-staging-images/llm-instance-gateway/epp)
- [ ] Submit a PR against [k8s.io](https://github.com/kubernetes/k8s.io), 
      updating [`k8s-staging-gateway-api-inference-extension/images.yaml`](https://github.com/kubernetes/k8s.io/blob/main/registry.k8s.io/images/k8s-staging-gateway-api-inference-extension/images.yaml) to
      promote the container images from staging to production. <!-- link to example PR once we have one -->
- [ ] Wait for the PR to be merged and verify that the image `registry.k8s.io/gateway-api-inference-extension/epp:$VERSION` is available.
- [ ] Publish the draft release prepared at the [Github releases page](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases).
- [ ] Add a link to the tagged release in this issue: <!-- example https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/tag/v0.1.0 -->
- [ ] Send an announcement email to `wg-serving@kubernetes.io` and `dev@kubernetes.io` with the subject `[ANNOUNCE] Gateway API Inference Extension  $VERSION is released`
- [ ] Add a link to the release announcement in this issue: <!-- link to an example email once we have one -->
- [ ] Update docs in `main` branch: <!-- link to example PR once we have one -->
- [ ] Close this issue


## Changelog
<!--
Describe changes since the last release here.
-->
