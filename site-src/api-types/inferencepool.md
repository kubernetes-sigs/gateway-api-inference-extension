# Inference Pool

??? success example "GA since v1.0.0"

    The `InferencePool` resource has been graduated to v1 and is considered stable.

## Background

The **InferencePool** API defines a group of Pods (containers) dedicated to serving AI models. Pods within an InferencePool share the same compute configuration, accelerator type, base language model, and model server. This abstraction simplifies the management of AI model serving resources, providing a centralized point of administrative configuration for Platform Admins.

An InferencePool is typically bundled with an Endpoint Picker extension. This extension is responsible for tracking key metrics on each model server (i.e. the KV-cache utilization, queue length of pending requests, active LoRA adapters, etc.) and routing incoming inference requests to the optimal model server replica based on these metrics. An EPP can only be associated with a single InferencePool, though an HTTPRoute may reference multiple InferencePools as backendRefs.

Additionally, any Pod that seeks to join an InferencePool would need to support the [model server protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/003-model-server-protocol), defined by this project, to ensure the Endpoint Picker has adequate information to intelligently route requests.

Until release `v1.5.0`, the InferencePool field `endpointPickerRef` was required. Currently, it is optional, to allow usages of InferencePool without user-managed Endpoint Picker deployments. Consult the documentation of your Inference Gateway implementation to check if it supports omitting `endpointPickerRef`.

## How to Configure an InferencePool

The full spec of the InferencePool is defined [here](/reference/spec/#inferencepool).

In summary, the InferencePoolSpec consists of 3 major parts:

- The `selector` field specifies which Pods belong to this pool. The labels in this selector must exactly match the labels applied to your model server Pods. 
- The `targetPortNumber` field defines the port number that the Inference Gateway should route to on model server Pods that belong to this pool. 
- The `endpointPickerRef` field references the Endpoint Picker (EPP) service that monitors key metrics from model servers within the InferencePool and provides intelligent routing decisions.

### Example Configuration

Here is an example InferencePool configuration:

```
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: vllm-qwen3-32b
spec:
  targetPorts:
    - number: 8000
  selector:
    app: vllm-qwen3-32b
  endpointPickerRef:
    name: vllm-qwen3-32b-epp
    port: 9002
    failureMode: FailOpen
```

In this example: 

- An InferencePool named `vllm-qwen3-32b` is created in the `default` namespace.
- It will select Pods that have the label `app: vllm-qwen3-32b`.
- Traffic routed to this InferencePool will call out to the EPP service `vllm-qwen3-32b-epp` on port `9002` for making routing decisions. If EPP fails to pick an endpoint, or is not responsive, the request will be dropped.
- Traffic routed to this InferencePool will be forwarded to the port `8000` on the selected Pods.

## Overlap with Service

**InferencePool** has some small overlap with **Service**, displayed here:

<!-- Source: https://docs.google.com/presentation/d/11HEYCgFi-aya7FS91JvAfllHiIlvfgcp7qpi_Azjk4E/edit#slide=id.g292839eca6d_1_0 -->
<img src="/images/inferencepool-vs-service.png" alt="Comparing InferencePool with Service" class="center" width="550" />

The InferencePool is not intended to be a mask of the Service object. It provides a specialized abstraction tailored for managing and routing traffic to groups of LLM model servers, allowing Platform Admins to focus on pool-level management rather than low-level networking details.

