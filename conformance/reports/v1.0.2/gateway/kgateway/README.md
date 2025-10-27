# Kgateway with agentgateway

This guide provides the steps for running Gateway conformance tests against [kgateway](https://kgateway.dev/) with the
([agentgateway](https://agentgateway.dev/)) data plane.

## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report                                                                     |
|--------------------------|----------------|------------------------|---------|----------------------------------------------------------------------------|
| v1.0.2                   | Gateway        | v2.1.1                 | default | [v2.1.1 report](./inference-v2.1.1-report.yaml)                            |

## Reproduce

This is a mirror of the kgateway [inference conformance GHA workflow](https://github.com/kgateway-dev/kgateway/blob/v2.0.x/.github/actions/kube-inference-extension-conformance-tests/action.yaml).

### Prerequisites

In order to run the conformance tests, the following prerequisites must be met:

- The [kubectl](https://kubernetes.io/docs/tasks/tools/) command-line tool installed and configured for the active cluster context.
- The [helm](https://github.com/helm/helm), [git](https://git-scm.com/downloads), and [make](https://www.gnu.org/software/make/) command-line tools installed.

### Steps

1. Set the environment variables use by the proceeding steps:

   ```sh
   # The kgateway version
   export VERSION=v2.1.1
   # Skip building and loading the kgateway images
   export SKIP_DOCKER=true
   # Install Gateway API and Inference Extension CRDs
   export CONFORMANCE=true
   ```

2. Clone the kgateway repository and checkout the release:

   ```sh
   git clone -b $VERSION https://github.com/kgateway-dev/kgateway.git && cd kgateway
   ```

3. Create a KinD cluster:

   ```sh
   make setup-base
   ```

4. Install the kgateway CRDs:

   ```sh
   helm upgrade -i --create-namespace --namespace kgateway-system \
   --version $VERSION kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds
   ```

5. Install kgateway with Inference Extension and agentgateway enabled:

   ```sh
   helm upgrade -i --namespace kgateway-system --version $VERSION \
   kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway \
   --set inferenceExtension.enabled=true --set agentgateway.enabled=true
   ```

6. Wait for the kgateway rollout to complete:

   ```sh
   kubectl rollout status deploy/kgateway -n kgateway-system
   ```

7. Run the conformance tests:

   ```sh
   CONFORMANCE_GATEWAY_CLASS=agentgateway make gie-conformance
   ```

8. View and verify the conformance report:

   ```sh
   cat _test/conformance/inference-$VERSION-report.yaml
   ```
