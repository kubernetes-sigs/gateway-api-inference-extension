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

In this guide, we modify the LoRA adapters ConfigMap to have two small-segment-lora models to better illustrate the gradual rollout scenario.

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
        - id: small-segment-lora-v1
          source: ttt421/nec119-small-segment-lora
        - id: small-segment-lora-v2
          source: ttt421/nec119-small-segment-lora
```

**Verify Available Models**: You can query the `/v1/models` endpoint to confirm the adapters are loaded:

```bash
curl http://${IP}/v1/models | jq . 
```

## Step 1: Establishing A Baseline (Alias v1)

First, we establish a stable baseline where all requests for `small-segment-lora` are served by the existing version, `small-segment-lora-v1`. This decouples the client's request (for "small-segment-lora") from the specific version running on the backend.

A client requests the model `small-segment-lora`. We want to ensure this maps strictly to `small-segment-lora-v1`.

### InferenceModelRewrite

Apply the following `InferenceModelRewrite` CR to map `small-segment-lora` → `small-segment-lora-v1`:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModelRewrite
metadata:
  name: small-segment-lora-rewrite
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

When a client requests `"model": "small-segment-lora"`, the system serves the request using `small-segment-lora-v1`.

```bash
curl http://${IP}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d 
'{ 
"model": "small-segment-lora",
"messages": [
  {
    "role": "user",
    "content": "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
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
        "content": "I would choose to go to the future because I",
        "reasoning_content": null,
        "role": "assistant",
        "tool_calls": []
      },
      "stop_reason": null
    }
  ],
  "created": 1764786158,
  "id": "chatcmpl-b10d939f-39bc-41ba-85c0-fe9b9d1ed3d9",
  "model": "small-segment-lora-v1",
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

Now that `small-segment-lora-v2` is loaded (from the Prerequisites step), we can begin splitting traffic. Traffic splitting allows you to divide incoming traffic for a single model name across different adapters. This is critical for A/B testing or gradual updates.
You want to direct 90% of `small-segment-lora` traffic to the stable `small-segment-lora-v1` and 10% to the new `small-segment-lora-v2`.

### InferenceModelRewrites (90 / 10 split)

Update the existing `InferenceModelRewrite`:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModelRewrite
metadata:
  name: small-segment-lora-rewrite
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

Run the [test traffic script](#test-traffic-script) as follows:

```bash
❯ ./test-traffic-splitting.sh
---
Traffic Split Results, total requests: 20
small-segment-lora-v1: 17 requests
small-segment-lora-v2: 3 requests
```

### InferenceModelRewrites (50 / 50 split)

To increase traffic to the new model, simply adjust the weights.

```yaml
      targets:
        - modelRewrite: "small-segment-lora-v1"
          weight: 50
        - modelRewrite: "small-segment-lora-v2"
          weight: 50
```

Run the [test traffic script](#test-traffic-script) again:

```bash
❯ ./test-traffic-splitting.sh
___
Traffic Split Results, total requests: 20
small-segment-lora-v1: 10 requests
small-segment-lora-v2: 10 requests
```

### InferenceModelRewrites (0 / 100 split)

Once the new model is verified, shift all traffic to it.

```yaml
      targets:
        - modelRewrite: "small-segment-lora-v2"
          weight: 100
```

Run the [test traffic script](#test-traffic-script) one last time:

```bash
❯ ./test-traffic-splitting.sh
------------------------------------------------
Traffic Split Results, total requests: 20
small-segment-lora-v1: 0 requests
small-segment-lora-v2: 20 requests
```

## Step 3: Cleanup

Now that 100% of traffic is routed to `small-segment-lora-v2`, you can safely unload the older version from the servers.

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
        - id: small-segment-lora-v2
          source: ttt421/nec119-small-segment-lora
      ensureNotExist:
        models:
        - id: small-segment-lora-v1
          source: ttt421/nec119-small-segment-lora
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
      "model": "small-segment-lora",
      "messages": [{"role": "user", "content": "test"}],
      "max_completion_tokens": 1
    }' | jq -r '.model')

  # 2. Check the response and update counters
  if [[ "$model_name" == "small-segment-lora-v1" ]]; then
    ((count_v1++))
    echo "Request $i: Hit small-segment-lora-v1"
  elif [[ "$model_name" == "small-segment-lora-v2" ]]; then
    ((count_v2++))
    echo "Request $i: Hit small-segment-lora-v2"
  else
    echo "Request $i: Received unexpected model: $model_name"
  fi
done

# 3. Print the final report
echo "------------------------------------------------"
echo "Traffic Split Results:"
echo "small-segment-lora-v1: $count_v1 requests"
echo "small-segment-lora-v2: $count_v2 requests"
```