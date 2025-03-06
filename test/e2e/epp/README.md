# End-to-End Tests

This document provides instructions on how to run the end-to-end tests.

## Overview

The end-to-end tests are designed to validate end-to-end Gateway API Inference Extension functionality. These tests are executed against a Kubernetes cluster and use the Ginkgo testing framework to ensure the extension behaves as expected.

## Prerequisites

- [Go](https://golang.org/doc/install) installed on your machine.
- [Make](https://www.gnu.org/software/make/manual/make.html) installed to run the end-to-end test target.
- A Hugging Face Hub token with access to the [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model.

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

1. **Run the Tests**: Run the `test-e2e` target:

   ```sh
   make test-e2e
   ```

   The test suite prints details for each step. Note that the `vllm-llama2-7b-pool` model server deployment
   may take several minutes to report an `Available=True` status due to the time required for bootstraping.
