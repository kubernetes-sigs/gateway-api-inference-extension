# Replacing an InferencePool

## Background

Replacing an InferencePool is a powerful technique for performing various infrastructure and model updates with minimal disruption and built-in rollback capabilities. This method allows you to introduce changes incrementally, monitor their impact, and revert to the previous state if necessary. 

## Use Cases
Use Cases for Replacing an InferencePool:

- Upgrading or replacing your model server framework
- Upgrading or replacing your base model
- Transitioning to new hardware

## How to replace an InferencePool

To replacing an InferencePool:

1. **Deploy new infrastructure**: Create a new InferencePool configured with the new hardware / model server / base model that you chose.
1. **Configure traffic splitting**: Use an HTTPRoute to split traffic between the existing InferencePool and the new InferencePool. The `backendRefs.weight` field controls the traffic percentage allocated to each pool.
1. **Maintain InferenceModel integrity**: Keep your InferenceModel configuration unchanged. This ensures that the system applies the same LoRA adapters consistently across both base model versions.
1. **Preserve rollback capability**: Retain the original nodes and InferencePool during the roll out to facilitate a rollback if necessary.

### Example

You start with an existing lnferencePool named `llm-pool-v1`. To replace the original InferencePool, you create a new InferencePool named `llm-pool-v2`. By configuring an **HTTPRoute**, as shown below, you can incrementally split traffic between the original `llm-pool-v1` and new `llm-pool-v2`. 

1. Save the following sample manifest as `httproute.yaml`:

    ```yaml
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

1. Increase the traffic weight gradually for the `llm-pool-v2` InferencePool to complete the new InferencePool roll out.
