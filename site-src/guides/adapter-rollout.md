# Lora Adapter Rollout 

The goal of this guide is to show you how to perform incremental roll out operations,
which gradually deploy new versions of your inference infrastructure.
You can update LoRA adapters and Inference Pool with minimal service disruption.
This page also provides guidance on traffic splitting and rollbacks to help ensure reliable deployments for LoRA adapters rollout.

LoRA adapter rollouts let you deploy new versions of LoRA adapters in phases,
without altering the underlying base model or infrastructure.
Use LoRA adapter rollouts to test improvements, bug fixes, or new features in your LoRA adapters.

## Example

###  Prerequisites
Follow the steps in the [main guide](index.md)

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
  criticality: 1
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
  criticality: 1
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

