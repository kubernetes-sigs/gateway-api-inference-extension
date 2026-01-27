# LoRA Adapter Rollout

The goal of this guide is to show you how to perform incremental roll out operations,
which gradually deploy new versions of your inference infrastructure.
You can update LoRA adapters in an InferencePool with minimal service disruption.

LoRA adapter rollouts let you deploy new versions of LoRA adapters in phases,
without altering the underlying base model or infrastructure.
Use LoRA adapter rollouts to test improvements, bug fixes, or new features in your LoRA adapters.

The [`InferenceModelRewrite`](/api-types/inferencemodelrewrite) resource allows platform administrators and model owners to control how inference requests are routed to specific models within an InferencePool.
This capability is essential for managing model/adapter lifecycles without disrupting client applications.

## Prerequisites & Setup

Follow [getting-started](https://gateway-api-inference-extension.sigs.k8s.io/guides/getting-started-latest/#getting-started-with-an-inference-gateway) to set up the IGW stack.

In this guide, we modify the LoRA adapters ConfigMap to have two qwen-uncensored models to better illustrate the gradual rollout scenario.

The ConfigMap used in this guide is as follows:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-qwen3-32b-adapters
data:
  configmap.yaml: |
    vLLMLoRAConfig:
      name: vllm-qwen3-32b-adapters
      port: 8000
      defaultBaseModel: Qwen/Qwen3-32B
      ensureExist:
        models:
        - id: qwen-uncensored-v1
          source: nicoboss/Qwen3-32B-Uncensored
        - id: qwen-uncensored-v2
          source: nicoboss/Qwen3-32B-Uncensored
```

**Verify Available Models**: You can query the `/v1/models` endpoint to confirm the adapters are loaded:

```bash
curl http://${IP}/v1/models | jq . 
```

## Step 1: Establishing A Baseline (Alias v1)

First, we establish a stable baseline where all requests for `qwen-uncensored` are served by the existing version, `qwen-uncensored-v1`. This decouples the client's request (for "qwen-uncensored") from the specific version running on the backend.

A client requests the model `qwen-uncensored`. We want to ensure this maps strictly to `qwen-uncensored-v1`.

### InferenceModelRewrite

Apply the following `InferenceModelRewrite` CR to map `qwen-uncensored` → `qwen-uncensored-v1`:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModelRewrite
metadata:
  name: qwen-uncensored-rewrite
spec:
  poolRef:
    group: inference.networking.k8s.io
    name: vllm-qwen3-32b
  rules:
    - matches:
        - model:
            type: Exact
            value: qwen-uncensored
      targets:
        - modelRewrite: "qwen-uncensored-v1"
```

When a client requests `"model": "qwen-uncensored"`, the system serves the request using `qwen-uncensored-v1`.

```bash
curl http://${IP}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d 
'{ 
"model": "qwen-uncensored",
"messages": [
  {
    "role": "user",
    "content": "Give me a spicy food challenge list."
  }
],
"max_completion_tokens": 10
}' | jq . 
```

Response:
```json
{
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "Here's a list of spicy foods that can help",
        "reasoning_content": null,
        "role": "assistant",
        "tool_calls": []
      },
      "stop_reason": null
    }
  ],
  "created": 1764786158,
  "id": "chatcmpl-b10d939f-39bc-41ba-85c0-fe9b9d1ed3d9",
  "model": "qwen-uncensored-v1",
  "object": "chat.completion",
  "prompt_logprobs": null,
  "usage": {
    "completion_tokens": 10,
    "prompt_tokens": 43,
    "prompt_tokens_details": null,
    "total_tokens": 53
  }
}
```

## Step 2: Gradual Rollout

Now that `qwen-uncensored-v2` is loaded (from the Prerequisites step), we can begin splitting traffic. Traffic splitting allows you to divide incoming traffic for a single model name across different adapters. This is critical for A/B testing or gradual updates.
You want to direct 90% of `qwen-uncensored` traffic to the stable `qwen-uncensored-v1` and 10% to the new `qwen-uncensored-v2`.

### InferenceModelRewrites (90 / 10 split)

Update the existing `InferenceModelRewrite`:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModelRewrite
metadata:
  name: qwen-uncensored-rewrite
spec:
  poolRef:
    group: inference.networking.k8s.io
    name: vllm-qwen3-32b
  rules:
    - matches:
        - model:
            type: Exact
            value: qwen-uncensored
      targets:
        - modelRewrite: "qwen-uncensored-v1"
          weight: 90
        - modelRewrite: "qwen-uncensored-v2"
          weight: 10
```

Run the [test traffic script](#test-traffic-script) as follows:

```bash
❯ ./test-traffic-splitting.sh
---
Traffic Split Results, total requests: 20
qwen-uncensored-v1: 17 requests
qwen-uncensored-v2: 3 requests
```

### InferenceModelRewrites (50 / 50 split)

To increase traffic to the new model, simply adjust the weights.

```yaml
      targets:
        - modelRewrite: "qwen-uncensored-v1"
          weight: 50
        - modelRewrite: "qwen-uncensored-v2"
          weight: 50
```

Run the [test traffic script](#test-traffic-script) again:

```bash
❯ ./test-traffic-splitting.sh
___
Traffic Split Results, total requests: 20
qwen-uncensored-v1: 10 requests
qwen-uncensored-v2: 10 requests
```

### InferenceModelRewrites (0 / 100 split)

Once the new model is verified, shift all traffic to it.

```yaml
      targets:
        - modelRewrite: "qwen-uncensored-v2"
          weight: 100
```

Run the [test traffic script](#test-traffic-script) one last time:

```bash
❯ ./test-traffic-splitting.sh
------------------------------------------------
Traffic Split Results, total requests: 20
qwen-uncensored-v1: 0 requests
qwen-uncensored-v2: 20 requests
```

## Step 3: Cleanup

Now that 100% of traffic is routed to `qwen-uncensored-v2`, you can safely unload the older version from the servers.

Update the LoRA syncer ConfigMap to list the older version under the `ensureNotExist` list:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-qwen3-32b-adapters
data:
  configmap.yaml: |
    vLLMLoRAConfig:
      name: vllm-qwen3-32b-adapters
      port: 8000
      defaultBaseModel: Qwen/Qwen3-32B
      ensureExist:
        models:
        - id: qwen-uncensored-v2
          source: Kawon/llama3.1-food-finetune_v14_r8
      ensureNotExist:
        models:
        - id: qwen-uncensored-v1
          source: Kawon/llama3.1-food-finetune_v14_r8
```

With this, the old adapter is removed, and the rollout is complete.

## Appendix

### Test Traffic Script

```bash
#!/bin/bash

# --- Configuration ---
# Replace this with your actual IP address or hostname
target_ip="${IP}"
# How many requests you want to send
total_requests=20

# Initialize counters
count_v1=0
count_v2=0

echo "Starting $total_requests requests to http://$target_ip..."
echo "------------------------------------------------"

for ((i=1; i<=total_requests; i++)); do
  # 1. Send the request
  # jq -r '.model': Extracts the raw string of the model name
  model_name=$(curl -s "http://${target_ip}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d 
'{ 
      "model": "qwen-uncensored",
      "messages": [{"role": "user", "content": "test"}],
      "max_completion_tokens": 1
    }' | jq -r '.model')

  # 2. Check the response and update counters
  if [[ "$model_name" == "qwen-uncensored-v1" ]]; then
    ((count_v1++))
    echo "Request $i: Hit qwen-uncensored-v1"
  elif [[ "$model_name" == "qwen-uncensored-v2" ]]; then
    ((count_v2++))
    echo "Request $i: Hit qwen-uncensored-v2"
  else
    echo "Request $i: Received unexpected model: $model_name"
  fi
done

# 3. Print the final report
echo "------------------------------------------------"
echo "Traffic Split Results:"
echo "qwen-uncensored-v1: $count_v1 requests"
echo "qwen-uncensored-v2: $count_v2 requests"
```