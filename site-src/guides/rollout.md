# Rollout

The goal of this guide is to show you how to perform incremental roll out operations, 
which gradually deploy new versions of your inference infrastructure. 
You can update nodes, base models, and LoRA adapters with minimal service disruption. 
This page also provides guidance on traffic splitting and rollbacks to help ensure reliable deployments.

The following use cases are supported:

*   [LoRA adapter update roll out](#lora-adapter-rollout)
*   [Node (compute, accelerator) update roll out](#node-update-rollout)
*   [Base model update roll out](#basemodel-rollout)


## **Prerequisites**

Follow the steps in the [main guide](index.md)


## **lora-adapter-rollout** {:#lora-adapter-rollout}

LoRA adapter update roll outs let you deploy new versions of fine-tuned models in phases, 
without altering the underlying base model or infrastructure. 
Use LoRA adapter update roll outs to test improvements, bug fixes, or new features in your LoRA adapters.

### Load the new adapter version to the model servers

This guide leverages the LoRA syncer sidecar to dynamically manage adapters within a vLLM deployment, enabling users to add or remove them through a shared ConfigMap.


Modify the LoRA syncer ConfigMap to initiate loading of the new adapter version.


```bash
kubectl edit configmap vllm-llama3-8b-instruct-adapters
```

Change the ConfigMap to match the following (note the new entry under models):

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-llama3-8b-instruct-adapters
data:
  configmap.yaml: |
    vLLMLoRAConfig:
      name: vllm-llama3-8b-instruct-adapters
      port: 8000
      defaultBaseModel: meta-llama/Llama-3.1-8B-Instruct
      ensureExist:
        models:
        - id: food-review-1
          source: Kawon/llama3.1-food-finetune_v14_r8
        - id: food-review-2
          source: Kawon/llama3.1-food-finetune_v14_r8
```

The new adapter version is applied to the model servers live, without requiring a restart.


### Direct traffic to the new adapter version

Modify the InferenceModel to configure a canary rollout with traffic splitting. In this example, 10% of traffic for food-review model will be sent to the new ***food-review-2*** adapter.


```bash
kubectl edit inferencemodel food-review
```

Change the targetModels list in InferenceModel to match the following:


```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModel
metadata:
  name: food-review
spec:
  modelName: food-review
  criticality: Standard
  poolRef:
    name: vllm-llama3-8b-instruct
  targetModels:
  - name: food-review-1
    weight: 90
  - name: food-review-2
    weight: 10
```

The above configuration means one in every ten requests should be sent to the new version. Try it out:

1. Get the gateway IP:
```bash
IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}'); PORT=80
```

2. Send a few requests as follows:
```bash
curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
"model": "food-review",
"prompt": "Write as if you were a critic: San Francisco",
"max_tokens": 100,
"temperature": 0
}'
```

### Finish the rollout


Modify the InferenceModel to direct 100% of the traffic to the latest version of the adapter.

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModel
metadata:
  name: food-review
spec:
  modelName: food-review
  criticality: Standard
  poolRef:
    name: vllm-llama3-8b-instruct
  targetModels:
  - name: food-review-2
    weight: 100
```

Unload the older versions from the servers by updating the LoRA syncer ConfigMap to list the older version under the `ensureNotExist` list:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-llama3-8b-instruct-adapters
data:
  configmap.yaml: |
    vLLMLoRAConfig:
      name: vllm-llama3-8b-instruct-adapters
      port: 8000
      defaultBaseModel: meta-llama/Llama-3.1-8B-Instruct
      ensureExist:
        models:
        - id: food-review-2
          source: Kawon/llama3.1-food-finetune_v14_r8
      ensureNotExist:
        models:
        - id: food-review-1
          source: Kawon/llama3.1-food-finetune_v14_r8
```

With this, all requests should be served by the new adapter version.

## **node-update-rollout** {: #node-update-rollout}
Node update roll outs safely migrate inference workloads to new node hardware or accelerator configurations. 
This process happens in a controlled manner without interrupting model service. 
Use node update roll outs to minimize service disruption during hardware upgrades, driver updates, or security issue resolution.

1.  **Create a new `InferencePool`**: deploy an `InferencePool` configured with the
    updated node or hardware specifications.

1.  **Split traffic using an `HTTPRoute`**: configure an `HTTPRoute` to distribute
    traffic between the existing and new `InferencePool` resources. Use the `weight`
    field in `backendRefs` to manage the traffic percentage directed to the new
    nodes.

1.  **Maintain a consistent `InferenceModel`**: retain the existing
    `InferenceModel` configuration to ensure uniform model behavior across both
    node configurations.

1.  **Retain original resources**: keep the original `InferencePool` and nodes
    active during the roll out to enable rollbacks if needed.

For example, you can create a new `InferencePool` named `llm-new`. Configure
this pool with the same model configuration as your existing `llm`
`InferencePool`. Deploy the pool on a new set of nodes within your cluster. Use
an `HTTPRoute` object to split traffic between the original `llm` and the new
`llm-new` `InferencePool`. This technique lets you incrementally update your
model nodes.

See an example here: [replace-inference-pool](replacing-inference-pool.md)

## **base-model-rollout** {: #basemodel-rollout}

Base model updates roll out in phases to a new base LLM, retaining compatibility
with existing LoRA adapters. You can use base model update roll outs to upgrade to
improved model architectures or to address model-specific issues.

1.  **Deploy new infrastructure**: Create new nodes and a new `InferencePool`
    configured with the new base model that you chose.
1.  **Configure traffic distribution**: Use an `HTTPRoute` to split traffic
    between the existing `InferencePool` (which uses the old base model) and the new
    `InferencePool` (using the new base model). The `backendRefs weight` field
    controls the traffic percentage allocated to each pool.
1.  **Maintain `InferenceModel` integrity**: keep your `InferenceModel`
    configuration unchanged. This ensures that the system applies the same LoRA
    adapters consistently across both base model versions.
1.  **Preserve rollback capability**: retain the original nodes and
    `InferencePool` during the roll out to facilitate a rollback if necessary.

See an example here: [replace-inference-pool](replacing-inference-pool.md)