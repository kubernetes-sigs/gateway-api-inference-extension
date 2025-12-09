# Serve multiple generative AI models and multiple LoRAs for the base AI models

A company may need to deploy multiple large language models (LLMs) in a cluster to support different workloads. For example, a Llama model could power a chatbot interface, while a DeepSeek model might serve a recommendation application. One approach is to expose these models on separate URL paths and follow the steps in the [`Getting Started (Latest/Main)`](getting-started-latest.md) guide for each model.

However, one may also need to serve multiple models from the same URL path. To achieve this, the system needs to extract information (such as the model name) from the request body (i.e., the LLM prompt). This pattern of serving multiple models behind a single endpoint is common among providers and is generally expected by clients. The OpenAI API format requires the model name to be specified in the request body. For such model-aware routing, use the Body-Based Routing (BBR) feature described in this guide.

Additionally, each base AI model can have multiple Low-Rank Adaptations ([LoRAs](https://www.ibm.com/think/topics/lora)). LoRAs associated with the same base model are served by the same backend inference server that hosts the base model. A LoRA name is also provided as the model name in the request body.

## How

The BBR extracts the model name from the request body and adds it to the `X-Gateway-Model-Name` header. This header is then used for matching and routing the request to the appropriate `InferencePool` and its associated Endpoint Picker Extension (EPP) instances.

### Example Model-Aware Routing using Body-Based Routing (BBR)

This guide assumes you have already setup the cluster for basic model serving as described in the [`Getting started (Latest/Main)`](getting-started-latest.md) guide. In what follows, this guide describes the additional steps required to deploy and test routing across multiple models and multiple LoRAs, where several LoRAs may be associated with a single base model.

### Deploy Body-Based Routing Extension

To enable body-based routing, deploy the BBR server using Helm. This server runs as a gateway extension and is independent of the EPP. Once installed, it is automatically added as the first filter in the gateway’s filter chain, ahead of other gateway extension servers such as the EPP.

Select an appropriate tab depending on your Gateway provider:

=== "GKE"

      ```bash
      helm install body-based-router \
      --set provider.name=gke \
      --version v0 \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/body-based-routing
      ```

=== "Istio"

      ```bash
      helm install body-based-router \
      --set provider.name=istio \
      --version v0 \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/body-based-routing
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
      --version v0 \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/body-based-routing
      ```

After the installation, verify that the BBR pod is running without errors:

```bash
kubectl get pods
```

### Serving a Second Base Model

The example uses a vLLM simulator since this is the least common denominator configuration that can be run in every environment. The model, `deepseek/vllm-deepseek-r1`, will be served from the same URL path, as in the previous example from the [Getting Started (Latest/Main)](getting-started-latest.md) guide.

Deploy the second base model:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/sim-deployment-1.yaml
```

The overall setup is as follows. Two base models are deployed: `meta-llama/Llama-3.1-8B-Instruct` and `deepseek/vllm-deepseek-r1`. Additionally, the `food-review-1` LoRA is associated with `meta-llama/Llama-3.1-8B-Instruct`, while the `ski-resorts` and `movie-critique` LoRAs are associated with `deepseek/vllm-deepseek-r1`.

⚠️ **Note**: LoRA names must be unique across the base AI models (i.e., across the backend inference server deployments)

Review the YAML definition.

```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-deepseek-r1
      spec:
      replicas: 1 
      selector:
        matchLabels:
          app: vllm-deepseek-r1
      template:
        metadata:
          labels:
            app: vllm-deepseek-r1
        spec:
          containers:
          - name: vllm-sim
            image: ghcr.io/llm-d/llm-d-inference-sim:v0.4.0
            imagePullPolicy: Always
            args:
            - --model
            - deepseek/vllm-deepseek-r1
            - --port
            - "8000"
            - --max-loras
            - "2"
            - --lora-modules
            - '{"name": "ski-resorts"}'
            - '{"name": "movie-critique"}'
            env:
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
            ports:
            - containerPort: 8000
              name: http
              protocol: TCP
            resources:
              requests:
                scpu: 10m
```

Verify that the second base model pod is running without errors:

```bash
kubectl get pods
```

### Deploy the 2nd InferencePool and Endpoint Picker Extension

Set the Helm chart version (unless already set).

   ```bash
   export IGW_CHART_VERSION=v0
   ```

Select a tab to follow the provider-specific instructions.

=== "GKE"

      ```bash
      export GATEWAY_PROVIDER=gke
      helm install vllm-deepseek-r1 \
      --set inferencePool.modelServers.matchLabels.app=vllm-deepseek-r1 \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Istio"

      ```bash
      export GATEWAY_PROVIDER=istio
      helm install vllm-deepseek-r1 \
      --set inferencePool.modelServers.matchLabels.app=vllm-deepseek-r1 \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```
=== "Kgateway"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install vllm-deepseek-r1 \
      --set inferencePool.modelServers.matchLabels.app=vllm-deepseek-r1 \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```

=== "NGINX Gateway Fabric"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install vllm-deepseek-r1 \
      --set inferencePool.modelServers.matchLabels.app=vllm-deepseek-r1 \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```

After the installation, verify that you have two `InferencePools` and two EPP pods, one per base model type, running without errors

```bash
kubectl get inferencepools
```

```bash
kubectl get pods
```

### Configure HTTPRoutes

Before configuring the HTTPRoutes for the models and their LoRAs, delete the existing HTTPRoute for the `meta-llama/Llama-3.1-8B-Instruct` model. The new routes will match the model name in the `X-Gateway-Model-Name` HTTP header, which is inserted by the BBR extension after parsing the model name from the LLM request body.

```bash
kubectl delete httproute llm-route
```

Now configure new HTTPRoutes for the two simulated models and their LoRAs that we want to serve via BBR using the following command which configures both routes.

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/bbr-example/httproute_bbr_lora.yaml
```

Also examine the manifest file (see the yaml below), to see how the `X-Gateway-Model-Name` is used for a header match in the Gateway's rules to route requests to the correct Backend based on the model name.

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
    - path:
        type: PathPrefix
        value: /
      headers:
        - type: Exact
          name: X-Gateway-Model-Name
          value: 'food-review-1'  
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
          value: 'ski-resorts'
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
```

⚠️ **Note** :
[Kubernetes API Gateway limits the total number of matchers per HTTPRoute to be less than 128](https://github.com/kubernetes-sigs/gateway-api/blob/df8c96c254e1ac6d5f5e0d70617f36143723d479/apis/v1/httproute_types.go#L128).

Before testing the setup, confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True` for both routes using the following commands.

```bash
kubectl get httproute llm-llama-route -o yaml
```

```bash
kubectl get httproute llm-deepseek-route -o yaml
```

### Try the setup

First, make sure that the setup works as before by sending a request to the LoRA of the first model set up in the [`Getting started (Latest/Main)`](getting-started-latest.md) guide.

--8<-- "site-src/_includes/test.md"

=== "Chat Completions API"

      1. Send a few requests to the Llama model directly:

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

      1. Send a few requests to Deepseek model to test that it works, as follows:

          ```bash
          curl -X POST -i ${IP}:${PORT}/v1/chat/completions \
               -H "Content-Type: application/json" \
               -d '{
                      "model": "deepseek/vllm-deepseek-r1",
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
                             "content": "What are the best movies of 2025?"
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
                             "content": "Tell mne about ski deals"
                            }
                       ]
                }'
          ```

=== "Completions API"

      1. Send a few requests to the Deepseek model:

           ```bash
           curl -X POST -i ${IP}:${PORT}/v1/completions \
                -H "Content-Type: application/json" \
                -d '{
                       "model": "deepseek/vllm-deepseek-r1",
                       "prompt": "What is the best ski resort in Austria?",
                       "max_tokens": 20,
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
                       "prompt": "Tell me about movies",
                       "max_tokens": 20,
                       "temperature": 0
                }'
           ```
