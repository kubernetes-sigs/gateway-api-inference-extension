# Agent Router

## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version                                                       | Mode    | Report                                                                        |
|--------------------------|----------------|------------------------------------------------------------------------------|---------|-------------------------------------------------------------------------------|
| v1.6.2                   | Gateway        | [v1.3.0](https://github.com/theagentrouter/agent-router/releases/tag/v1.3.0) | default | [Conformance report](./inference-extension-conformance-test-report.yaml)      |

## Reproduce

This is a mirror of the Agent Router [conformance e2e test](https://github.com/theagentrouter/agent-router/blob/main/.github/workflows/build_and_test.yaml), which includes the conformance tests for the Gateway API and Inference Extension.

### Prerequisites

In order to run the conformance tests, the following prerequisites must be met:

- The [kubectl](https://kubernetes.io/docs/tasks/tools/) command-line tool installed and configured for the active cluster context.
- The [helm](https://github.com/helm/helm), [kind](https://kind.sigs.k8s.io), [git](https://git-scm.com/downloads), and [make](https://www.gnu.org/software/make/) command-line tools installed.

### Steps

1. Clone the Agent Router repository and check out the tested release:

   ```sh
   git clone https://github.com/theagentrouter/agent-router.git && cd agent-router
   git checkout tags/v1.3.0
   ```

2. Run the Gateway API Inference Extension conformance tests:

   ```sh
   make test-e2e GO_TEST_ARGS="-run TestGatewayAPIInferenceExtension -v" EG_VERSION=v0.0.0-latest TEST_KEEP_CLUSTER=true
   ```
