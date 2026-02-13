
# Test Setup and Execution

This document provides steps to run the Gateway API Inference Extension conformance tests.

## Prerequisites

1.  You need a Kubernetes cluster with [LoadBalancer](https://kubernetes.io/docs/concepts/services-networking/service/#loadbalancer) support.

2.  Choose an Implementation -
Install an [existing implementation](https://gateway-api-inference-extension.sigs.k8s.io/implementations/gateways/). For setup instructions, refer to the [The Quickstart Guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/).  Alternatively run tests against your implementation after completing the [implementer's guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/implementers/#implementers-guide).

Note: Since the EPP (EndPoint Picker) takes the `InferencePool` name as an environment variable, each conformance test creates a corresponding EPP deployment for each `InferencePool` it defines. For conformance testing, the EPP is configured with the `HeaderBasedTestingFilter`. This is enabled by setting the `ENABLE_REQ_HEADER_BASED_SCHEDULER_FOR_TESTING=true` environment variable in the EPP deployment manifest. 

## Running Conformance Tests

1.  **Clone the Repository**:
    Create a local copy of the Gateway API Inference Extension repository:
    ```bash
    git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension.git
    cd gateway-api-inference-extension
    ```

2.  **Execute Tests**:
    Run the following command to execute all available tests. Replace `<your_gatewayclass_name>` with the GatewayClass used by the implementation under test.

    ```bash
    go test ./conformance -args -gateway-class <your_gatewayclass_name>
    ```

### Test Execution Options

* **Speeding up Reruns**: For repeated runs, you can add the flag `-cleanup-base-resources=false`. This will preserve resources such as namespaces and gateways between test runs, speeding up the process.
    ```bash
    go test ./conformance -args -gateway-class <your_gatewayclass_name> -cleanup-base-resources=false
    ```

* **Running Specific Tests**: To run a specific test, you can reference the test name by using the `-run-test` flag. For example:
    ```bash
    go test ./conformance -args -gateway-class <your_gatewayclass_name> -run-test HTTPRouteMultipleGatewaysDifferentPools
    ```

* **Detailed Logging**: To view detailed logs, you can enable logging mode by adding the `-v` as well as `-debug` flags.
    ```bash
    go test -v ./conformance -args -debug -gateway-class <your_gatewayclass_name> -cleanup-base-resources=false -run-test HTTPRouteMultipleGatewaysDifferentPools
    ```
