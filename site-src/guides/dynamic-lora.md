# Getting started with Gateway API Inference Extension with Dynamic lora updates on vllm

The goal of this guide is to get a single InferencePool running with vLLM and demonstrate use of dynamic lora updating! 

### Requirements
 - Envoy Gateway [v1.2.1](https://gateway.envoyproxy.io/docs/install/install-yaml/#install-with-yaml) or higher
 - A cluster with:
   - Support for Services of type `LoadBalancer`. (This can be validated by ensuring your Envoy Gateway is up and running). For example, with Kind,
     you can follow [these steps](https://kind.sigs.k8s.io/docs/user/loadbalancer).
   - 3 GPUs to run the sample model server. Adjust the number of replicas in `./manifests/vllm/deployment.yaml` as needed.

### Steps

1. **Deploy Sample VLLM Model Server with dynamic lora update enabled and dynamic lora syncer sidecar **
    [Redeploy the vLLM deployment with Dynamic lora adapter enabled and Lora syncer sidecar and configmap](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/manifests/vllm/dynamic-lora-sidecar/deployment.yaml)

Rest of the steps are same as [general setup](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/site-src/guides/index.md)


### Safely rollout v2 adapter
    
1. Update the LoRA syncer ConfigMap to make the new adapter version available on the model servers.

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
                      id: tweet-summary-0
                      source: vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm
                    - base-model: meta-llama/Llama-2-7b-hf
                      id: tweet-summary-1
                      source: vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm
                    - base-model: meta-llama/Llama-2-7b-hf
                      id: tweet-summary-2
                      source: vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm
2. Configure a canary rollout with traffic split using LLMService. In this example, 40% of traffic for tweet-summary model will be sent to the ***tweet-summary-2*** adapter .

```yaml
model:
    name: tweet-summary
    targetModels:
    targetModelName: tweet-summary-0
            weight: 20
    targetModelName: tweet-summary-1
            weight: 40
    targetModelName: tweet-summary-2
            weight: 40
    
```
            
3. Finish rollout by setting the traffic to the new version 100%.
```yaml
model:
    name: tweet-summary
    targetModels:
    targetModelName: tweet-summary-2
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
                name: sql-loras-llama
                port: 8000
                ensureExist:
                    models:
                    - base-model: meta-llama/Llama-2-7b-hf
                      id: tweet-summary-2
                      source: vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm
                ensureNotExist:
                    models:
                    - base-model: meta-llama/Llama-2-7b-hf
                      id: tweet-summary-1
                      source: gs://[HUGGING FACE PATH]
                    - base-model: meta-llama/Llama-2-7b-hf
                      id: tweet-summary-0
                      source: gs://[HUGGING FACE PATH]
```
