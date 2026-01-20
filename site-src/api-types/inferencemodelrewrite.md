# Inference Model Rewrite

??? example "Alpha since v1.2.1"

    The `InferenceModelRewrite` resource is alpha and may have breaking changes in
    future releases of the API.

## Background

The **InferenceModelRewrite** resource allows platform administrators and model owners to control how inference requests are routed to specific models within an InferencePool.
This capability is essential for managing model lifecycles without disrupting client applications.

## Usages

*   **Model Aliasing**: Map a model name in the request body (e.g., `small-segment-lora`) to a specific version (e.g., `small-segment-lora-v1`).
*   **Generic Fallbacks**: Redirect unknown model requests to a default model.
*   **Traffic Splitting**: Gradually roll out new model versions (Canary deployment) by splitting traffic between two models based on percentage weights.

    !!! note "Scope of Traffic Splitting"
        Traffic splitting with `InferenceModelRewrite` is scoped to traffic within a single `InferencePool`. It will not split traffic between two different pools, but rather between two adapters served by the same `InferencePool`.

## Spec

The full spec of the InferenceModelRewrite is defined [here](/reference/x-v1a2-spec/#inferencemodelrewrite).

## Usage Examples

### Model Aliasing

Map a virtual model name (e.g., `small-segment-lora`) to a specific backend model version (e.g., `small-segment-lora-v1`).

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModelRewrite
metadata:
  name: small-segment-lora-alias
spec:
  poolRef:
    group: inference.networking.k8s.io
    name: vllm-qwen3-32b
  rules:
    - matches:
        - model:
            type: Exact
            value: small-segment-lora
      targets:
        - modelRewrite: "small-segment-lora-v1"
```

### Generic (Wildcard) Rewrites

Redirect any request with an unrecognized or unspecified model name to a default model. An empty `matches` list implies that the rule applies to **all** requests not matched by previous rules.

```yaml
apiVersion: inference.networking.k8s.io/v1alpha2
kind: InferenceModelRewrite
metadata:
  name: generic-fallback
spec:
  poolRef:
    group: inference.networking.k8s.io
    name: vllm-qwen3-32b
  rules:
    - matches: [] # Empty means this rule matches everything
      targets: 
        - modelRewrite: "Qwen/Qwen3-32B"
```

### Traffic Splitting (Canary Rollout)

Divide incoming traffic for a single model name across multiple adapters within the InferencePool. This is useful for A/B testing or gradual rollouts for LoRA adapter updates.

```yaml
apiVersion: inference.networking.k8s.io/v1alpha2
kind: InferenceModelRewrite
metadata:
  name: small-segment-lora-canary
spec:
  poolRef:
    group: inference.networking.k8s.io
    name: vllm-qwen3-32b
  rules:
    - matches:
        - model:
            type: Exact
            value: small-segment-lora
      targets:
        - modelRewrite: "small-segment-lora-v1"
          weight: 90
        - modelRewrite: "small-segment-lora-v2"
          weight: 10
```

## Limitations

1.  **Status Reporting**: Currently, `InferenceModelRewrite` is simply a config read-only CR. It does not report status conditions (e.g., Valid or Ready) in the CRD status field.
2.  **Scheduler Assumptions**: Traffic splitting occurs before the scheduling algorithm. The system assumes that all model servers within the referenced `InferencePool` are capable of serving the target models. If a model is missing from a specific server in the pool, requests routed to it may fail.
3.  **Splitting algorithm**: The current traffic split is weighted-random.