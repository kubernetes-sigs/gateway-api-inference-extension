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

In this guide, we modify the LoRA adapters ConfigMap to have two food-review models to better illustrate the gradual rollout scenario.

The ConfigMap used in this guide is as follows:

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
        - id: food-review-v1
          source: Kawon/llama3.1-food-finetune_v14_r8
        - id: food-review-v2
          source: Kawon/llama3.1-food-finetune_v14_r8
```

**Verify Available Models**: You can query the `/v1/models` endpoint to confirm the adapters are loaded:

```bash
curl http://${IP}/v1/models | jq . 
```

## Step 1: Establishing A Baseline (Alias v1)

First, we establish a stable baseline where all requests for `food-review` are served by the existing version, `food-review-v1`. This decouples the client's request (for "food-review") from the specific version running on the backend.

A client requests the model `food-review`. We want to ensure this maps strictly to `food-review-v1`.

### InferenceModelRewrite

Apply the following `InferenceModelRewrite` CR to map `food-review` → `food-review-v1`:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModelRewrite
metadata:
  name: food-review-rewrite
spec:
  poolRef:
    group: inference.networking.k8s.io
    name: vllm-llama3-8b-instruct
  rules:
    - matches:
        - model:
            type: Exact
            value: food-review
      targets:
        - modelRewrite: "food-review-v1"
```

When a client requests `"model": "food-review"`, the system serves the request using `food-review-v1`.

```bash
curl http://${IP}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d 
'{ 
"model": "food-review",
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
  "model": "food-review-v1",
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

Now that `food-review-v2` is loaded (from the Prerequisites step), we can begin splitting traffic. Traffic splitting allows you to divide incoming traffic for a single model name across different adapters. This is critical for A/B testing or gradual updates.
You want to direct 90% of `food-review` traffic to the stable `food-review-v1` and 10% to the new `food-review-v2`.

### InferenceModelRewrites (90 / 10 split)

Update the existing `InferenceModelRewrite`:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModelRewrite
metadata:
  name: food-review-rewrite
spec:
  poolRef:
    group: inference.networking.k8s.io
    name: vllm-llama3-8b-instruct
  rules:
    - matches:
        - model:
            type: Exact
            value: food-review
      targets:
        - modelRewrite: "food-review-v1"
          weight: 90
        - modelRewrite: "food-review-v2"
          weight: 10
```

Run the [test traffic script](#test-traffic-script) as follows:

```bash
❯ ./test-traffic-splitting.sh
---
Traffic Split Results, total requests: 20
food-review-v1: 17 requests
food-review-v2: 3 requests
```

### InferenceModelRewrites (50 / 50 split)

To increase traffic to the new model, simply adjust the weights.

```yaml
      targets:
        - modelRewrite: "food-review-v1"
          weight: 50
        - modelRewrite: "food-review-v2"
          weight: 50
```

Run the [test traffic script](#test-traffic-script) again:

```bash
❯ ./test-traffic-splitting.sh
___
Traffic Split Results, total requests: 20
food-review-v1: 10 requests
food-review-v2: 10 requests
```

### InferenceModelRewrites (0 / 100 split)

Once the new model is verified, shift all traffic to it.

```yaml
      targets:
        - modelRewrite: "food-review-v2"
          weight: 100
```

Run the [test traffic script](#test-traffic-script) one last time:

```bash
❯ ./test-traffic-splitting.sh
------------------------------------------------
Traffic Split Results, total requests: 20
food-review-v1: 0 requests
food-review-v2: 20 requests
```

## Step 3: Cleanup

Now that 100% of traffic is routed to `food-review-v2`, you can safely unload the older version from the servers.

Update the LoRA syncer ConfigMap to list the older version under the `ensureNotExist` list:

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
        - id: food-review-v2
          source: Kawon/llama3.1-food-finetune_v14_r8
      ensureNotExist:
        models:
        - id: food-review-v1
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
      "model": "food-review",
      "messages": [{"role": "user", "content": "test"}],
      "max_completion_tokens": 1
    }' | jq -r '.model')

  # 2. Check the response and update counters
  if [[ "$model_name" == "food-review-v1" ]]; then
    ((count_v1++))
    echo "Request $i: Hit food-review-v1"
  elif [[ "$model_name" == "food-review-v2" ]]; then
    ((count_v2++))
    echo "Request $i: Hit food-review-v2"
  else
    echo "Request $i: Received unexpected model: $model_name"
  fi
done

# 3. Print the final report
echo "------------------------------------------------"
echo "Traffic Split Results:"
echo "food-review-v1: $count_v1 requests"
echo "food-review-v2: $count_v2 requests"
```