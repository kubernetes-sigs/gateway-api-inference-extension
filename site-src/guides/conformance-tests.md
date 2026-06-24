# Test Setup and Execution

This document provides steps to run the Gateway API Inference Extension conformance tests.

## Prerequisites

1.  You need a Kubernetes cluster with [LoadBalancer](https://kubernetes.io/docs/concepts/services-networking/service/#loadbalancer) support.

2.  Choose an Implementation -
Install an [existing implementation](https://gateway-api-inference-extension.sigs.k8s.io/implementations/gateways/). Alternatively run tests against your implementation after completing the [implementer's guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/implementers/#implementers-guide).

Note: This repository provides a **Lightweight Endpoint Picker (lwepp)** specifically designed for conformance testing. Since the EPP takes the `InferencePool` name and namespace as command-line arguments, each conformance test automatically spins up a corresponding `lwepp` deployment for each `InferencePool` it defines. To facilitate deterministic route verification, the `lwepp` natively supports routing requests to a specific target pod via the `test-epp-endpoint-selection` header, which the conformance tests use to steer and validate routing behavior. 

## Running Conformance Tests

1.  **Clone the Repository**:
    Create a local copy of the Gateway API Inference Extension repository:
    ```bash
    git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension.git
    cd gateway-api-inference-extension/conformance
    ```

2.  **Execute Tests**:
    Run the following command to execute all available tests. Replace `<your_gatewayclass_name>` with the GatewayClass used by the implementation under test.

    ```bash
    go test . -args -gateway-class <your_gatewayclass_name>
    ```

### Test Execution Options

* **Speeding up Reruns**: For repeated runs, you can add the flag `-cleanup-base-resources=false`. This will preserve resources such as namespaces and gateways between test runs, speeding up the process.
    ```bash
    go test . -args -gateway-class <your_gatewayclass_name> -cleanup-base-resources=false
    ```

* **Running Specific Tests**: To run a specific test, you can reference the test name by using the `-run-test` flag. For example:
    ```bash
    go test . -args -gateway-class <your_gatewayclass_name> -run-test HTTPRouteMultipleGatewaysDifferentPools
    ```

* **Detailed Logging**: To view detailed logs, you can enable logging mode by adding the `-v` as well as `-debug` flags.
    ```bash
    go test -v . -args -debug -gateway-class <your_gatewayclass_name> -cleanup-base-resources=false -run-test HTTPRouteMultipleGatewaysDifferentPools
    ```
