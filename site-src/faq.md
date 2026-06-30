# Frequently Asked Questions (FAQ)

## How can I get involved with this project?
The [contributing](/contributing) page keeps track of how to get involved with
the project.

## Why isn't this project in the main Gateway API repo?
There is an active migration plan in progress to migrate the Gateway API Inference Extension into the main Gateway API and `llm-d` repositories. Specifically:
* The APIs (`InferencePool` and `InferencePoolImport`) and the conformance test suite will be migrated into the [Gateway API](https://gateway-api.sigs.k8s.io/) repository.
* The reference Endpoint Picker (EPP) server implementation, plugins, and benchmarks are being consolidated into the [llm-d-router](https://github.com/llm-d/llm-d-router) repository.
* The Body-Based Routing (BBR) and Latency Predictor packages are being extracted to their own standalone repositories under the `llm-d` organization.

## Will there be a default controller implementation?
No. Although this project previously hosted a reference Endpoint Picker (EPP)
implementation (which has now been moved to the [llm-d-router](https://github.com/llm-d/llm-d-router)
repository), this project now provides only a lightweight reference extension
to support conformance testing. Individual Gateway controllers should implement
their own extension or use an existing one. The scope of this project is to
define the API extension model, the conformance tests, and overall
documentation.
