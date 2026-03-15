# BBR End-to-End Tests

This document provides instructions on how to run the BBR (Body Based Router) end-to-end tests.

## Overview

The end-to-end tests validate BBR's core production behavior: request body parsing, model name extraction, base model lookup via ConfigMaps, header mutation (`X-Gateway-Base-Model-Name` with `ClearRouteCache`), and header-based routing to distinct backend clusters. These tests are executed against a Kubernetes cluster and use the Ginkgo testing framework.

## Prerequisites

- [Go](https://golang.org/doc/install) installed on your machine.
- [Make](https://www.gnu.org/software/make/manual/make.html) installed to run the end-to-end test target.
- [Docker](https://docs.docker.com/get-docker/) installed for building the BBR image.
- [kubectl](https://kubernetes.io/docs/tasks/tools/) configured with access to a Kubernetes cluster.

## Running the End-to-End Tests

Follow these steps to run the end-to-end tests:

1. **Clone the Repository**: Clone the `gateway-api-inference-extension` repository:

   ```sh
   git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension.git && cd gateway-api-inference-extension
   ```

1. **Optional Settings**

   - **Set the test namespace**: By default, the e2e test creates resources in the `bbr-e2e` namespace.
     If you would like to change this namespace, set the following environment variable:

     ```sh
     export E2E_NS=<MY_NS>
     ```

   - **Pause before cleanup**: To pause the test run before cleaning up resources, set the `E2E_PAUSE_ON_EXIT` environment variable.
     This is useful for debugging the state of the cluster after the test has run.

     - To pause indefinitely, set it to `true`: `export E2E_PAUSE_ON_EXIT=true`
     - To pause for a specific duration, provide a duration string: `export E2E_PAUSE_ON_EXIT=10m`

1. **Run the Tests**: Run the `test-e2e-bbr` target:

   ```sh
   make test-e2e-bbr
   ```

   The test suite deploys two model server simulators (Llama + DeepSeek), a BBR instance, an Envoy proxy
   with `ext_proc` routing, and a curl client pod. It then validates base model routing, LoRA adapter routing,
   streaming request handling, and BBR metrics exposure.
