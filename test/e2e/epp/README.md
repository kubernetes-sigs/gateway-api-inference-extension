# End-to-End Tests

This document provides instructions on how to run the end-to-end tests.

## Overview

The end-to-end tests are designed to validate end-to-end Gateway API Inference Extension functionality. These tests are executed against a Kubernetes cluster and use the Ginkgo testing framework to ensure the extension behaves as expected.

## Prerequisites

- [Go](https://golang.org/doc/install) installed on your machine.
- [Make](https://www.gnu.org/software/make/manual/make.html) installed to run the end-to-end test target.
- (Optional) When using the GPU-based vLLM deployment, a Hugging Face Hub token with access to the
  [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model is required.
  After obtaining the token and being granted access to the model, set the `HF_TOKEN` environment variable:

   ```sh
   export HF_TOKEN=<MY_HF_TOKEN>
   ```

## Running the End-to-End Tests

Follow these steps to run the end-to-end tests:

1. **Clone the Repository**: Clone the `gateway-api-inference-extension` repository:

   ```sh
   git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension.git && cd gateway-api-inference-extension
   ```

1. **Optional Settings**

   - **Set the test namespace**: By default, the e2e test creates resources in the `inf-ext-e2e` namespace.
     If you would like to change this namespace, set the following environment variable:

     ```sh
     export E2E_NS=<MY_NS>
     ```

   - **Set the model server manifest**: By default, the e2e test uses the [vLLM Simulator](https://github.com/llm-d/llm-d-inference-sim)
     (`config/manifests/vllm/sim-deployment.yaml`) to simulate a backend model server. If you would like to change the model server
     deployment type, set the following environment variable to one of the following:

     ```sh
     export E2E_MANIFEST_PATH=[config/manifests/vllm/gpu-deployment.yaml|config/manifests/vllm/cpu-deployment.yaml]
     ```

   - **Enable leader election tests**: By default, the e2e test runs the EPP server as a single replica.
     To test the high-availability (HA) mode with leader election (3 replicas), set the following environment variable:

     ```sh
     export E2E_LEADER_ELECTION_ENABLED=true
     ```

   - **Pause before cleanup**: To pause the test run before cleaning up resources, set the `E2E_PAUSE_ON_EXIT` environment variable.
     This is useful for debugging the state of the cluster after the test has run.

     - To pause indefinitely, set it to `true`: `export E2E_PAUSE_ON_EXIT=true`
     - To pause for a specific duration, provide a duration string: `export E2E_PAUSE_ON_EXIT=10m`

1. **Run the Tests**: Run the `test-e2e` target:

   ```sh
   make test-e2e
   ```

   The test suite prints details for each step. Note that the `vllm-llama3-8b-instruct` model server deployment
   may take several minutes to report an `Available=True` status due to the time required for bootstrapping.
