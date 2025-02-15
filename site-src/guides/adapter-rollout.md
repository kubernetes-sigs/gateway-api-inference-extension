# Adapter Rollout

The goal of this guide is to demonstrate how to rollout a new adapter version.

## **Requirements**

Follow the steps in the [main guide](index.md)


## **Safely rollout v2 adapter**

### Load the new adapter version to the model servers

This guide leverages the LoRA syncer sidecar to dynamically manage adapters within a vLLM deployment, enabling users to add or remove them through a shared ConfigMap.


Modify the LoRA syncer ConfigMap to initiate loading of the new adapter version.


```bash
   kubectl edit configmap vllm-llama2-7b-adapters
```

Change the ConfigMap to match the following (note the new entry under models):

```yaml
        apiVersion: v1
        kind: ConfigMap
        metadata:
        name: vllm-llama2-7b-adapters
        data:
        configmap.yaml: |
             vLLMLoRAConfig:
                name: vllm-llama2-7b-adapters
                port: 8000
                ensureExist:
                    models:
                    - base-model: meta-llama/Llama-2-7b-hf
                      id: tweet-summary-1
                      source: vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm
                    - base-model: meta-llama/Llama-2-7b-hf
                      id: tweet-summary-2
                      source: mahimairaja/tweet-summarization-llama-2-finetuned
```

The new adapter version is applied to the model servers live, without requiring a restart.


### Direct traffic to the new adapter version

Modify the InferenceModel to configure a canary rollout with traffic splitting. In this example, 10% of traffic for tweet-summary model will be sent to the new ***tweet-summary-2*** adapter.


```bash
   kubectl edit configmap tweet-summary
```

Change the InferenceModel to match the following:


```yaml
model:
    name: tweet-summary
    targetModels:
    targetModelName: tweet-summary-1
            weight: 90
    targetModelName: tweet-summary-2
            weight: 10
    
```

The above configuration means one in every ten requests should be sent to the new version. Try it out:

1. Get the gateway IP:
```bash
IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}'); PORT=8081
```

2. Send a few requests as follows:
```bash
curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
"model": "tweet-summary",
"prompt": "Write as if you were a critic: San Francisco",
"max_tokens": 100,
"temperature": 0
}'
```

### Finish the rollout


Modify the InferenceModel to direct 100% of the traffic to the latest version of the adapter.

```yaml
model:
    name: tweet-summary
    targetModels:
    targetModelName: tweet-summary-2
            weight: 100
```

Unload the older versions from the servers by updating the LoRA syncer ConfigMap to list the older version under the `ensureNotExist` list:

```yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
    name: dynamic-lora-config
    data:
    configmap.yaml: |
            vLLMLoRAConfig:
                name: sql-loras-llama
                port: 8000
                ensureExist:
                    models:
                    - base-model: meta-llama/Llama-2-7b-hf
                      id: tweet-summary-2
                      source: mahimairaja/tweet-summarization-llama-2-finetuned
                ensureNotExist:
                    models:
                    - base-model: meta-llama/Llama-2-7b-hf
                      id: tweet-summary-1
                      source: vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm
```

With this, all requests should be served by the new adapter version.
