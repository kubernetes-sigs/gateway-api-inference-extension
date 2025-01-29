# Inference Model

??? example "Alpha since v0.1.0"

    The `InferenceModel` resource is alpha and may have breaking changes in
    future releases of the API.

## Background

An InferenceModel allows the Inference Workload Owner to define:
- Which Model/LoRA adapter(s) to consume .
  - Mapping from a client facing model name to the target model name in the InferencePool.
  - InferenceModel allows for traffic splitting between adapters _in the same InferencePool_ to allow for new LoRA adapter versions to be easily rolled out.
- Criticality of the requests to the InferenceModel.
- The InferencePools this InferenceModel is relevant to.

## Spec

The full spec of the InferenceModel is defined [here](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/api/v1alpha1/inferencemodel_types.go).