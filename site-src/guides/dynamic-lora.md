# Getting started with Gateway API Inference Extension with Dynamic lora updates on vllm

The goal of this guide is to get a single InferencePool running with VLLM and demonstrate use of dynamic lora updating ! 

### Requirements
 - Envoy Gateway [v1.2.1](https://gateway.envoyproxy.io/docs/install/install-yaml/#install-with-yaml) or higher
 - A cluster with:
   - Support for Services of type `LoadBalancer`. (This can be validated by ensuring your Envoy Gateway is up and running). For example, with Kind,
     you can follow [these steps](https://kind.sigs.k8s.io/docs/user/loadbalancer).
   - 3 GPUs to run the sample model server. Adjust the number of replicas in `./manifests/vllm/deployment.yaml` as needed.

### Steps

1. **Deploy Sample VLLM Model Server with dynamic lora update enabled and dynamic lora syncer sidecar **
    [Deploy sample vllm deployment with Dynamic lora adapter enabled and Lora syncer sidecar and configmap](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/manifests/vllm/dynamic-lora-sidecar/deployment.yaml)

Rest of the steps are same as [general setup](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/site-src/guides/index.md)


### Safely rollout v2 adapter
    
1. Update lora configmap

``` yaml

        apiVersion: v1
        kind: ConfigMap
        metadata:
        name: dynamic-lora-config
        data:
        configmap.yaml: |
            vLLMLoRAConfig:
            ensureExist:   
                models:
                - id: tweet-summary-v1
                    source: tweet-summary-1=/adapters/vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm_1
                - id: tweet-summary-v2
                    source: tweet-summary-2=/adapters/vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm_2
    ```

2. Configure a canary rollout with traffic split using InferenceModel. In this example, 10% of traffic to the chatbot model will be sent to `tweet-summary-3`.

``` yaml
model:
    name: chatbot
    targetModels:
    targetModelName: chatbot-v1
            weight: 90
    targetModelName: chatbot-v2
            weight: 10
```
            
3. Finish rollout by setting the traffic to the new version 100%.
```yaml
model:
    name: chatbot
    targetModels:
    targetModelName: chatbot-v2
            weight: 100
```
    
4. Remove v1 from dynamic lora configmap.
```yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
    name: dynamic-lora-config
    data:
    configmap.yaml: |
            vLLMLoRAConfig:
            ensureExist:
            models:
            - id: chatbot-v2
                source: gs://[TEAM-A-MODELS-BUCKET]/chatbot-v2
            ensureNotExist: # Explicitly unregisters the adapter from  model servers
            models:
            - id: chatbot-v1
                source: gs://[TEAM-A-MODELS-BUCKET]/chatbot-v1
```
