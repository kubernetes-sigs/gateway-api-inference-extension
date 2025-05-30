# End-to-End Tests

This document provides instructions on how to run the end-to-end tests.

## Overview

The end-to-end tests are designed to validate end-to-end Gateway API Inference Extension functionality. These tests are executed against a Kubernetes cluster and use the Ginkgo testing framework to ensure the extension behaves as expected.

## Prerequisites

- [Go](https://golang.org/doc/install) installed on your machine.
- [Make](https://www.gnu.org/software/make/manual/make.html) installed to run the end-to-end test target.
- A Hugging Face Hub token with access to the [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model.

## Running the End-to-End Tests

Follow these steps to run the end-to-end tests:

1. **Clone the Repository**: Clone the `gateway-api-inference-extension` repository:

   ```sh
   git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension.git && cd gateway-api-inference-extension
   ```

1. **Export Your Hugging Face Hub Token**: The token is required to run the test model server:

   ```sh
   export HF_TOKEN=<MY_HF_TOKEN>
   ```

1. **Optional Settings**

   * Set the test namespace**: By default, the e2e test creates resources in the `inf-ext-e2e` namespace.
     If you would like to change this namespace, set the following environment variable:

     ```sh
     export E2E_NS=<MY_NS>
     ```

   * Set the model server manifest**: By default, the e2e test uses the [vLLM Simulator](https://github.com/llm-d/llm-d-inference-sim)
     (`config/manifests/vllm/sim-deployment.yaml`) to simulate a backend model server. If you would like to change the model server
     type, set the following environment variable:

     ```sh
     export E2E_MANIFEST_PATH=[config/manifests/vllm/gpu-deployment.yaml|config/manifests/vllm/cpu-deployment.yaml]
     ```

1. **Run the Tests**: Run the `test-e2e` target:

   ```sh
   make test-e2e
   ```

   The test suite prints details for each step. Note that the `vllm-llama3-8b-instruct` model server deployment
   may take several minutes to report an `Available=True` status due to the time required for bootstraping.
