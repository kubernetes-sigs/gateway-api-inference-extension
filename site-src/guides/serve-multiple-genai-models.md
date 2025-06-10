# Serve multiple generative AI models
A company wants to deploy multiple large language models (LLMs) to serve different workloads. 
For example, they might want to deploy a Gemma3 model for a chatbot interface and a Deepseek model for a recommendation application. 
The company needs to ensure optimal serving performance for these LLMs.
Using Gateway API Inference Extension, you can deploy these LLMs on your cluster with your chosen accelerator configuration in an `InferencePool`. 
You can then route requests based on the model name (such as "chatbot" and "recommender") and the `Criticality` property.

## How
The following diagram illustrates how Gateway API Inference Extension routes requests to different models based on the model name.
The model name is extarcted by [Body-Based routing](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/bbr/README.md)
 from the request body to the header. The header is then matched to dispatch
 requests to different `InferencePool` (and their EPPs) instances.
![Serving multiple generative AI models](../images/serve-mul-gen-AI-models.png)

This example illustrates a conceptual example regarding how to use the `HTTPRoute` object to route based on model name like “chatbot” or “recommender” to `InferencePool`.
```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: routes-to-llms
spec:
  parentRefs:
  - name: inference-gateway
  rules:
  - matches:
    - headers:
      - type: Exact
        #Body-Based routing(https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/bbr/README.md) is being used to copy the model name from the request body to the header.
        name: X-Gateway-Model-Name
        value: chatbot
      path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: gemma3
      kind: InferencePool
  - matches:
    - headers:
      - type: Exact
        #Body-Based routing(https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/bbr/README.md) is being used to copy the model name from the request body to the header.
        name: X-Gateway-Model-Name
        value: recommender
      path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: deepseek-r1
      kind: InferencePool     
```

## Try it out

1. Get the gateway IP:
```bash
IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}'); PORT=80
```
2. Send a few requests to model "chatbot" as follows:
```bash
curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
"model": "chatbot",
"prompt": "What is the color of the sky",
"max_tokens": 100,
"temperature": 0
}'
```
3. Send a few requests to model "recommender" as follows:
```bash
curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
"model": "recommender",
"prompt": "Give me restaurant recommendations in Paris",
"max_tokens": 100,
"temperature": 0
}'
```
