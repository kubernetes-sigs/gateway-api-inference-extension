# Envoy AI Gateway

## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report                                                                |
|--------------------------|----------------|------------------------|---------|-----------------------------------------------------------------------|
| v0.5.1                   | Gateway        | [latest](https://github.com/envoyproxy/ai-gateway)                  | default | [Conformance report](./aigw-latest-report.yaml) |
| ...                      | ...            | ...                    | ...     | ...                                                                   |

## Reproduce

This is a mirror of the envoy ai gateway [conformance e2e test](https://github.com/envoyproxy/ai-gateway/blob/main/.github/workflows/build_and_test.yaml), which includes the conformance tests for the Gateway API and Inference Extension.

### Prerequisites

In order to run the conformance tests, the following prerequisites must be met:

- The [kubectl](https://kubernetes.io/docs/tasks/tools/) command-line tool installed and configured for the active cluster context.
- The [helm](https://github.com/helm/helm),[kind](https://kind.sigs.k8s.io), [git](https://git-scm.com/downloads), and [make](https://www.gnu.org/software/make/) command-line tools installed.

### Steps

1. Clone the envoy-ai-gateway repository and checkout the release:

   ```sh
   git clone https://github.com/envoyproxy/ai-gateway.git && cd ai-gateway
   ```

2. Running the Gateway API Inference Extension conformance tests:

   ```sh
      make test-e2e GO_TEST_ARGS="-run TestGatewayAPIInferenceExtension -v" EG_VERSION=v0.0.0-latest TEST_KEEP_CLUSTER=true
   ```
