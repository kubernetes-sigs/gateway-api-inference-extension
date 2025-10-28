# InferencePool Rollout
The goal of this guide is to show you how to perform incremental roll out operations,
which gradually deploy new versions of your inference infrastructure.
You can update Inference Pool with minimal service disruption.
This page also provides guidance on traffic splitting and rollbacks to help ensure reliable deployments for InferencePool rollout.

InferencePool rollout is a powerful technique for performing various infrastructure and model updates with minimal disruption and built-in rollback capabilities.
This method allows you to introduce changes incrementally, monitor their impact, and revert to the previous state if necessary.

## Use Cases
Use Cases for InferencePool Rollout:

- Node(compute, accelerator) update roll out
- Base model roll out
- Model server framework rollout

### Node(compute, accelerator) update roll out
Node update roll outs safely migrate inference workloads to new node hardware or accelerator configurations.
This process happens in a controlled manner without interrupting model service.
Use node update roll outs to minimize service disruption during hardware upgrades, driver updates, or security issue resolution.

### Base model roll out
Base model updates roll out in phases to a new base LLM, retaining compatibility with existing LoRA adapters.
You can use base model update roll outs to upgrade to improved model architectures or to address model-specific issues.

### Model server framework rollout
Model server framework rollouts enable the seamless deployment of new versions or entirely different serving frameworks,
like updating from an older vLLM version to a newer one, or even migrating from a custom serving solution to a managed one.
This type of rollout is critical for introducing performance enhancements, new features, or security patches within the serving layer itself,
without requiring changes to the underlying base models or application logic. By incrementally rolling out framework updates,
teams can ensure stability and performance, quickly identifying and reverting any regressions before they impact the entire inference workload.

## How to do InferencePool rollout

1. **Deploy new infrastructure**: Create a new InferencePool configured with the new node(compute/accelerator) / model server / base model that you chose.
1. **Configure traffic splitting**: Use an HTTPRoute to split traffic between the existing InferencePool and the new InferencePool. The `backendRefs.weight` field controls the traffic percentage allocated to each pool.
1. **Preserve rollback capability**: Retain the original nodes and InferencePool during the roll out to facilitate a rollback if necessary.

## Example
This is an example of InferencePool rollout with node(compute, accelerator) update roll out

###  Prerequisites
Follow the steps in the [main guide](index.md)

### Deploy new infrastructure
You start with an existing InferencePool named vllm-llama3-8b-instruct.
To replace the original InferencePool, you create a new InferencePool, configured to select the pods with the `nvidia-h100-80gb` accelerator type.

Assuming the new model servers already exist, simply:
**Create a new helm-managed InferencePool of a different name, with a new selector specified**

### Direct traffic to the new inference pool
By configuring an **HTTPRoute**, as shown below, you can incrementally split traffic between the original `vllm-llama3-8b-instruct` and new `vllm-llama3-8b-instruct-new`.

```bash
kubectl edit httproute llm-route
```

Change the backendRefs list in HTTPRoute to match the following:


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
    - backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: vllm-llama3-8b-instruct
          weight: 90
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: vllm-llama3-8b-instruct-new
          weight: 10
      matches:
        - path:
            type: PathPrefix
            value: /
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


Modify the HTTPRoute to direct 100% of the traffic to the latest version of the InferencePool.

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
    - backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: vllm-llama3-8b-instruct-new
          weight: 100
      matches:
        - path:
            type: PathPrefix
            value: /
```

### Delete old version of InferencePool and Endpoint Picker Extension
```shell
helm uninstall <old-inference-pool-name>
```

With this, all requests should be served by the new Inference Pool.
