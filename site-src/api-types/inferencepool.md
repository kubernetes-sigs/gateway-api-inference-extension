# Inference Pool

??? example "Alpha since v0.1.0"

    The `InferencePool` resource is alpha and may have breaking changes in
    future releases of the API.

## Background

The **InferencePool** API defines a group of Pods (containers) dedicated to serving AI models. Pods within an InferencePool share the same compute configuration, accelerator type, base language model, and model server. This abstraction simplifies the management of AI model serving resources, providing a centralized point of administrative configuration for Platform Admins.

An InferencePool is expected to be bundled with an [Endpoint Picker](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/pkg/epp) extension. This extension is responsible for tracking key metrics on each model server (i.e. the KV-cache utilization, queue length of pending requests, active LoRA adapters, etc.) and routing incoming inference requests to the optimal model server replica based on these metrics.

Additionally, any Pod that seeks to join an InferencePool would need to support the [model server protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/003-model-server-protocol), defined by this project, to ensure the Endpoint Picker has adequate information to intelligently route requests.

## How to Configure an InferencePool

The full spec of the InferencePool is defined [here](/reference/spec/#inferencepool).

In summary, the InferencePoolSpec consists of 3 major parts:

- The `selector` field specifies which Pods belong to this pool. The labels in this selector must exactly match the labels applied to your model server Pods. 
- The `targetPortNumber` field defines the port number that the Inference Gateway should route to on model server Pods that belong to this pool. 
- The `extensionRef` field references the [endpoint picker extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/pkg/epp) (EPP) service that monitors key metrics from model servers within the InferencePool and provides intelligent routing decisions.

### Example Configuration

Here is an example InferencePool configuration:

```
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferencePool
metadata:
  name: vllm-llama3-8b-instruct
spec:
  targetPortNumber: 8000
  selector:
    app: vllm-llama3-8b-instruct
  extensionRef:
    name: vllm-llama3-8b-instruct-epp
    port: 9002
    failureMode: FailClose
```

In this example: 

- An InferencePool named `vllm-llama3-8b-instruct` is created in the `default` namespace.
- It will select Pods that have the label `app: vllm-llama3-8b-instruct`.
- Traffic routed to this InferencePool will call out to the EPP service `vllm-llama3-8b-instruct-epp` on port `9002` for making routing decisions. If EPP fails to pick an endpoint, or is not responsive, the request will be dropped.
- Traffic routed to this InferencePool will be forwarded to the port `8000` on the selected Pods.

## Overlap with Service

**InferencePool** has some small overlap with **Service**, displayed here:

<!-- Source: https://docs.google.com/presentation/d/11HEYCgFi-aya7FS91JvAfllHiIlvfgcp7qpi_Azjk4E/edit#slide=id.g292839eca6d_1_0 -->
<img src="/images/inferencepool-vs-service.png" alt="Comparing InferencePool with Service" class="center" width="550" />

The InferencePool is not intended to be a mask of the Service object. It provides a specialized abstraction tailored for managing and routing traffic to groups of LLM model servers, allowing Platform Admins to focus on pool-level management rather than low-level networking details.

## Replacing an InferencePool

Replacing an InferencePool is a powerful technique for performing various infrastructure and model updates with minimal disruption and built-in rollback capabilities. This method allows you to introduce changes incrementally, monitor their impact, and revert to the previous state if necessary. 

Use Cases for Replacing an InferencePool:

- Upgrading or replacing your model server framework
- Upgrading or replacing your base model
- Transitioning to new hardware

To rollout a new base model:

1. **Deploy new infrastructure**: Create a new InferencePool configured with the new base model that you chose.
1. **Configure traffic splitting**: Use an HTTPRoute to split traffic between the existing InferencePool (which uses the old base model) and the new InferencePool (using the new base model). The `backendRefs.weight` field controls the traffic percentage allocated to each pool.
1. **Maintain InferenceModel integrity**: Keep your InferenceModel configuration unchanged. This ensures that the system applies the same LoRA adapters consistently across both base model versions.
1. **Preserve rollback capability**: Retain the original nodes and InferencePool during the roll out to facilitate a rollback if necessary.

### Example

You start with an existing lnferencePool named `llm-pool-v1`. To replace the base model, you create a new InferencePool named `llm-pool-v2`. This pool deploys a new version of the base model on a new set of Pods. By configuring an **HTTPRoute**, as shown below, you can incrementally split traffic between the original `llm-pool-v1` and `llm-pool-v2`. This lets you control base model updates in your cluster.

1. Save the following sample manifest as `httproute.yaml`:

    ```
    apiVersion: gateway.networking.k8s.io/v1
    kind: HTTPRoute
    metadata:
      name: llm-route
    spec:
      parentRefs:
      - group: gateway.networking.k8s.io
        kind: Gateway
        name: inference-gateway
      rules:
        backendRefs:
        - group: inference.networking.x-k8s.io
          kind: InferencePool
          name: llm-pool-v1
          weight: 90
        - group: inference.networking.x-k8s.io
          kind: InferencePool
          name: llm-pool-v2
          weight: 10
    ```

1. Apply the sample manifest to your cluster:

    ```
    kubectl apply -f httproute.yaml
    ```

    The original `llm-pool-v1` InferencePool receives most of the traffic, while the `llm-pool-v2` InferencePool receives the rest. 

1. Increase the traffic weight gradually for the `llm-pool-v2` InferencePool to complete the base model update roll out.
