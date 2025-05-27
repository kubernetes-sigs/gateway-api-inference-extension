
# Test Setup and Execution

This document outlines the steps to set up your environment and run the conformance tests for the Gateway API Inference Extension.

## Prerequisites: External Istio/Envoy Setup

Before running the conformance tests, you need a functional Kubernetes cluster (e.g., GKE) with an Ingress/Gateway solution like Istio or Envoy configured. Refer to the following guides for assistance:

* **Google Cloud GKE Inference Gateway Tutorial**: [https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-with-gke-inference-gateway](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-with-gke-inference-gateway)
* **Gateway API Inference Extension Guides**: [https://gateway-api-inference-extension.sigs.k8s.io/guides/](https://gateway-api-inference-extension.sigs.k8s.io/guides)

## Running the Conformance Tests

1.  **Choose an Implementation**:
    Select one of the existing implementations (e.g., Istio/Envoy) that you have set up as per the prerequisites.

2.  **Cluster Setup and Verification**:
    * Follow the corresponding instructions from the guides above to set up a GKE (or other Kubernetes) cluster with the required Custom Resource Definitions (CRDs) for the Gateway API Inference Extension.
    * Ensure the cluster is fully functional by testing an HTTP call. 

3.  **Clone the Repository**:
    Create a local copy of the Gateway API Inference Extension repository:
    ```bash
    git clone [https://github.com/kubernetes-sigs/gateway-api-inference-extension.git](https://github.com/kubernetes-sigs/gateway-api-inference-extension.git)
    cd gateway-api-inference-extension
    ```

4.  **Execute Tests**:
    Navigate to the root of the cloned repository folder and run the following command to execute all available tests. Replace `<your gateway class name ex istio>` with the actual gateway class name you are using (e.g., `istio`).

    ```bash
    go test ./conformance -args -gateway-class <your gateway class name ex istio>
    ```

### Test Execution Options

* **Speeding up Reruns**: For repeated runs, you can add the flag `-cleanup-base-resources=false`. This will preserve resources such as namespaces and gateways between test runs, speeding up the process.
    ```bash
    go test ./conformance -args -gateway-class <your gateway class name ex istio> -cleanup-base-resources=false
    ```

* **Running Specific Tests**: To run a specific test, you can reference the test name by using the `-run-test` flag. For example:
    ```bash
    go test ./conformance -args -gateway-class <your gateway class name ex istio> -run-test HTTPRouteMultipleGatewaysDifferentPools
    ```

* **Detailed Logging**: To view detailed logs, you can enable logging mode by adding the `-v` as well as `-debug` flags.
    ```bash
    go test -v ./conformance -args -debug -gateway-class <your gateway class name ex istio> -cleanup-base-resources=false -run-test HTTPRouteMultipleGatewaysDifferentPools
    ```
