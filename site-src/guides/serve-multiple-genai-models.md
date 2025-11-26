# Serve multiple generative AI models and multiple LoRAs for the base AI models

A company wants to deploy multiple large language models (LLMs) to a cluster to serve different workloads.
For example, they might want to deploy a Gemma3 model for a chatbot interface and a DeepSeek model for a recommendation application (or as in the example in this guide, a combination of a Llama3 model and a smaller Phi4 model).. You may choose to locate these 2 models at 2 different L7 url paths and follow the steps described in the [`Getting started`](index.md) guide for each such model as already described. However you may also need to serve multiple models located at the same L7 url path and rely on parsing information such as
the Model name in the LLM prompt requests as defined in the OpenAI API format which is commonly used by most models. For such Model-aware routing, you can use the Body-Based Routing feature as described in this guide.

In addition, for each base AI model multiple [Low Rank Adaptaions (LoRAs)](https://www.ibm.com/think/topics/lora) can be defined. LoRAs defined for the same base AI model are served from the same backend inference server that serves the base model. A LoRA name is specified as the Model name in the body of LLM prompt requests. LoRA naming is not standardised. Therefore, it cannot be expected that the base model name can be inferred from the LoRA name.

## How

The following diagram illustrates how an Inference Gateway routes requests to different models based on the model name.
The model name is extracted by [Body-Based routing](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/bbr/README.md) (BBR)
 from the request body to the header. The header is then matched to dispatch
 requests to different `InferencePool` (and their EPPs) instances.

### Example Model-Aware Routing using Body-Based Routing (BBR)

This guide assumes you have already setup the cluster for basic model serving as described in the [`Getting started`](index.md) guide and this guide describes the additional steps needed from that point onwards in order to deploy and exercise an example of routing across multiple models and multiple LoRAs with many to one relationship of LoRAs to the base model.

### Deploy Body-Based Routing Extension

To enable body-based routing, you need to deploy the Body-Based Routing ExtProc server using Helm. This is a separate ExtProc server from the EndPoint Picker and when installed, is automatically inserted at the start of the gateway's ExtProc chain ahead of other EtxProc servers such as EPP.  

First install this server. Depending on your Gateway provider, you can use one of the following commands:

=== "GKE"

      ```bash
      helm install body-based-router \
      --set provider.name= \
      --version v1.0.0 \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/body-based-routing
      ```

=== "Istio"

      ```bash
      helm install body-based-router \
      --set provider.name=istio \
      --version v1.0.0 \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/body-based-routing
      ```

=== "Kgateway"

    Kgateway does not require the Body-Based Routing Extension, and instead natively implements Body-Based Routing.
    To use Body Based Routing, apply an `AgentgatewayPolicy`:

    ```yaml
    apiVersion: gateway.kgateway.dev/v1alpha1
    kind: AgentgatewayPolicy
    metadata:
      name: bbr
    spec:
      targetRefs:
      - group: gateway.networking.k8s.io
        kind: Gateway
        name: inference-gateway
      traffic:
        phase: PreRouting
        transformation:
          request:
            set:
            - name: X-Gateway-Model-Name
              value: 'json(request.body).model'
    ```

=== "Other"

      ```bash
      helm install body-based-router \
      --version v1.0.0 \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/body-based-routing
      ```

Once this is installed, verify that the BBR pod is running without errors using the command `kubectl get pods`.

### Serving a Second Base Model

Next deploy the second base model that will be served from the same L7 path (which is `/`) as the `meta-llama/Llama-3.1-8B-Instruct` model already being served after following the steps from the [`Getting started`](index.md) guide. In this example the 2nd model is `microsoft/Phi-4-mini-instruct` a relatively small model ( about 3B parameters) from HuggingFace. Note that for this exercise, there need to be atleast 2 GPUs available on the system one each for the two models being served. Serve the second model via the following command.

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/heads/main/config/manifests/bbr-example/vllm-phi4-mini.yaml
```

Once this is installed, and after allowing for model download and startup time which can last several minutes, verify that the pod with this 2nd LLM phi4-mini, is running without errors using the command `kubectl get pods`.

### Deploy the 2nd InferencePool and Endpoint Picker Extension

We also want to use an InferencePool and EndPoint Picker for this second model in addition to the Body Based Router in order to be able to schedule across multiple endpoints.

===


=== "GKE"

      ```bash
      export GATEWAY_PROVIDER=gke
      helm install vllm-phi4-mini-instruct \
      --set inferencePool.modelServers.matchLabels.app=phi4-mini \
      --set provider.name=$GATEWAY_PROVIDER \
      --version v1.0.0 \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Istio"

      ```bash
      export GATEWAY_PROVIDER=istio
      helm install vllm-phi4-mini-instruct \
      --set inferencePool.modelServers.matchLabels.app=phi4-mini \
      --set provider.name=$GATEWAY_PROVIDER \
      --version v1.0.0 \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

After executing this, verify that you see two InferencePools and two EPP pods, one per base model type, running without errors, using the CLIs `kubectl get inferencepools` and `kubectl get pods`.

### Configure HTTPRoute

Before configuring the httproutes for the models, we need to delete the prior httproute created for the vllm-llama3-8b-instruct model because we will alter the routing to now also match on the model name as determined by the `X-Gateway-Model-Name` http header that will get inserted by the BBR extension after parsing the Model name from the body of the LLM request message.

```bash
kubectl delete httproute llm-route
```

Now configure new HTTPRoutes, one per each model we want to serve via BBR using the following command which configures both routes. Also examine this manifest file, to see how the `X-Gateway-Model-Name` is used for a header match in the Gateway's rules to route requests to the correct Backend based on model name. For convenience the manifest is also listed below in order to view this routing configuration.

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/bbr-example/httproute_bbr.yaml
```

```yaml
---   
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-llama-route
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
    matches:
    - path:
        type: PathPrefix
        value: /
      headers:
        - type: Exact
          #Body-Based routing(https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/bbr/README.md) is being used to copy the model name from the request body to the header.
          name: X-Gateway-Model-Name # (1)!
          value: 'meta-llama/Llama-3.1-8B-Instruct'
    timeouts:
      request: 300s
---   
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-phi4-route
spec:
  parentRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: inference-gateway
  rules:
  - backendRefs:
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: vllm-phi4-mini-instruct
    matches:
    - path:
        type: PathPrefix
        value: /
      headers:
        - type: Exact
          #Body-Based routing(https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/bbr/README.md) is being used to copy the model name from the request body to the header.
          name: X-Gateway-Model-Name
          value: 'microsoft/Phi-4-mini-instruct'
    timeouts:
      request: 300s
---   
```

Before testing the setup, confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True` for both routes using the following commands.

```bash
kubectl get httproute llm-llama-route -o yaml
```

```bash
kubectl get httproute llm-phi4-route -o yaml
```

## Try it out

1. Get the gateway IP:

```bash
IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}'); PORT=80
```

=== "Chat Completions API"

       1. Send a few requests to Llama model as follows:

          ```bash
          curl -X POST -i ${IP}:${PORT}/v1/chat/completions \
               -H "Content-Type: application/json" \
               -d '{
                     "model": "meta-llama/Llama-3.1-8B-Instruct",
                     "max_tokens": 100,
                     "temperature": 0,
                     "messages": [
                       {
                          "role": "developer",
                          "content": "You are a helpful assistant."
                       },
                       {
                          "role": "user",
                          "content": "Linux is said to be an open source kernel because "
                       }
                     ]
                }'
          ```

       1. Send a few requests to the Phi4 as follows:

          ```bash
          curl -X POST -i ${IP}:${PORT}/v1/chat/completions \
               -H "Content-Type: application/json" \
               -d '{
                      "model": "microsoft/Phi-4-mini-instruct",
                      "max_tokens": 100,
                      "temperature": 0,
                      "messages": [
                      {
                         "role": "developer",
                         "content": "You are a helpful assistant."
                      },
                      {
                         "role": "user",
                         "content": "2+2 is "
                      }
                     ]
                }'
          ```

=== "Completions API"

       1. Send a few requests to Llama model as follows:

          ```bash
          curl -X POST -i ${IP}:${PORT}/v1/completions \
               -H "Content-Type: application/json" \
               -d '{
                      "model": "meta-llama/Llama-3.1-8B-Instruct",
                      "prompt": "Linux is said to be an open source kernel because ",
                      "max_tokens": 100,
                      "temperature": 0
               }'
          ```

       1. Send a few requests to the Phi4 as follows:

          ```bash
          curl -X POST -i ${IP}:${PORT}/v1/completions \
               -H "Content-Type: application/json" \
               -d '{
                      "model": "microsoft/Phi-4-mini-instruct",
                      "prompt": "2+2 is ",
                      "max_tokens": 20,
                      "temperature": 0
                }'
          ```

### Serving multiple LoRAs per base AI model

⚠️ **Requirement**: LoRA names must be unique across the base AI models (i.e., across the backend inference server deployments)

Deploy the third base model that is used to demonstrate multiple LoRA configuration per base model. The example uses a vLLM simulator since this is the least common denominator configuration that can be run in every environment. The model, `deepseek/vllm-deepseek-r1`, will be served from the same `/` L7 path, as in the previous examples.

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/sim-deployment-1.yaml
```

Once this is installed, verify that the BBR pod is running without errors using the command `kubectl get pods`

### Deploy the 3rd InferencePool and Endpoint Picker Extension

We also want to use an InferencePool and EndPoint Picker for this third model.

=== "GKE"

      ```bash
      export GATEWAY_PROVIDER=gke
      helm install vllm-deepseek-r1 \
      --set inferencePool.modelServers.matchLabels.app=vllm-deepseek-r1 \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Istio"

      ```bash
      export GATEWAY_PROVIDER=istio
      helm install vllm-deepseek-r1 \
      --set inferencePool.modelServers.matchLabels.app=vllm-deepseek-r1 \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

### Configure HTTPRoutes

Now configure new HTTPRoutes for the two simulated models and their LoRAs that we want to serve via BBR using the following command which configures both routes.

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/bbr-example/httproute_bbr_lora.yaml
```

Also examine this manifest file (see the yaml below), to see how the `X-Gateway-Model-Name` is used for a header match in the Gateway's rules to route requests to the correct Backend based on model name. For convenience the manifest is also listed below in order to view this routing configuration. Note that the manifest file uses two different ways of defining the routes to LoRAs: (1) via adding match clauses on the same base AI model HTTPRoute or by (2) defining a separate HTTPRoutes. There is no functional diffeence between the two methods, except for the limitation on the number of matchers per route imposed by the API Gateway

⚠️ **Known Issue** :
[Kubernetes API Gateway limits the total number of matchers per HTTPRoute to be less than 128](https://github.com/kubernetes-sigs/gateway-api/blob/df8c96c254e1ac6d5f5e0d70617f36143723d479/apis/v1/httproute_types.go#L128).

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-llama-route
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
    matches:
    - path:
        type: PathPrefix
        value: /
      headers:
        - type: Exact
          name: X-Gateway-Model-Name 
          value: 'meta-llama/Llama-3.1-8B-Instruct'
    timeouts:
      request: 300s
---   
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-deepseek-route #give this HTTPRoute any name that helps you to group and track the matchers
spec:
  parentRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: inference-gateway
  rules:
  - backendRefs:
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: vllm-deepseek-r1
    matches:
    - path:
        type: PathPrefix
        value: /
      headers:
        - type: Exact
          name: X-Gateway-Model-Name
          value: 'deepseek/vllm-deepseek-r1'
    - path:
        type: PathPrefix
        value: /
      headers:
        - type: Exact
          name: X-Gateway-Model-Name
          value: 'food-review'
    - path:
        type: PathPrefix
        value: /
      headers:
        - type: Exact
          name: X-Gateway-Model-Name
          value: 'movie-critique'
    timeouts:
      request: 300s
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: vllm-llama3-8b-instruct-lora-food-review-1 #give this HTTPRoute any name that helps you to group and track the routes
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
    matches:
    - path:
        type: PathPrefix
        value: /
      headers:
        - type: Exact
          name: X-Gateway-Model-Name 
          value: 'food-review-1'   #this is the name of LoRA as defined in vLLM deployment
    timeouts:
      request: 300s
---
```

Before testing the setup, confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True` for both routes using the following commands.

```bash
kubectl get httproute llm-llama-route -o yaml
```

```bash
kubectl get httproute llm-deepseek-route -o yaml
```

```bash
kubectl get httproute vllm-llama3-8b-instruct-lora-food-review-1 -o yaml
```

### Try the setup

=== "Chat Completions API"

      1. Send a few requests to Llama model to test that it works as before, as follows:

          ```bash
          curl -X POST -i ${IP}:${PORT}/v1/chat/completions \
               -H "Content-Type: application/json" \
               -d '{
                      "model": "meta-llama/Llama-3.1-8B-Instruct",
                      "max_tokens": 100,
                      "temperature": 0,
                      "messages": [
                          {
                             "role": "developer",
                             "content": "You are a helpful assistant."
                          },
                          {
                              "role": "user",
                              "content": "Linux is said to be an open source kernel because "
                          }
                      ]
                   }'
          ```

      1. Send a few requests to the LoRA of the Llama model as follows:

          ```bash
          curl -X POST -i ${IP}:${PORT}/v1/chat/completions \
               -H "Content-Type: application/json" \
               -d '{
                      "model": "food-review-1",
                      "max_tokens": 100,
                      "temperature": 0,
                      "messages": [
                          {
                             "role": "reviewer",
                             "content": "You are a helpful assistant."
                          },
                          {
                              "role": "user",
                              "content": "Write a review of the best restaurans in San-Francisco"
                          }
                      ]
                }'
          ```

      1. Send a few requests to one LoRA of the Deepseek model as follows:

          ```bash
          curl -X POST -i ${IP}:${PORT}/v1/chat/completions \
               -H "Content-Type: application/json" \
               -d '{
                      "model": "movie-critique",
                      "max_tokens": 100,
                      "temperature": 0,
                      "messages": [
                          {
                             "role": "reviewer",
                             "content": "You are a helpful assistant."
                          },
                          {
                             "role": "user",
                             "content": "The best movies of 2025 are"
                          }
                      ]
                }'
          ```

      1. Send a few requests to another LoRA of the Deepseek model as follows:

          ```bash
          curl -X POST -i ${IP}:${PORT}/v1/chat/completions \
               -H "Content-Type: application/json" \
               -d '{
                      "model": "ski-resorts",
                      "max_tokens": 100,
                      "temperature": 0,
                      "messages": [
                          {
                             "role": "reviewer",
                             "content": "You are a helpful assistant."
                           },
                           {
                             "role": "user",
                             "content": "The best movies of 2025 are"
                            }
                       ]
                }'
          ```

=== "Completions API"

      1. Send a few requests to Llama model's LoRA as follows:

         ```bash
         curl -X POST -i ${IP}:${PORT}/v1/completions \
              -H "Content-Type: application/json" \
              -d '{
                    "model": "food-review-1",
                    "prompt": "Linux is said to be an open source kernel because ",
                    "max_tokens": 100,
                    "temperature": 0
             }'
         ```

      1. Send a few requests to the first Deepseek LoRA as follows:

           ```bash
           curl -X POST -i ${IP}:${PORT}/v1/completions \
                -H "Content-Type: application/json" \
                -d '{
                       "model": "ski-resorts",
                       "prompt": "What is the best ski resort in Austria?",
                       "max_tokens": 20,
                       "temperature": 0
                }'
           ```

      1. Send a few requests to the second Deepseek LoRA as follows:

           ```bash
           curl -X POST -i ${IP}:${PORT}/v1/completions \
                -H "Content-Type: application/json" \
                -d '{
                       "model": "movie-critique",
                       "prompt": "Write as if you were a movie critique",
                       "max_tokens": 20,
                       "temperature": 0
                }'
           ```
