# Metrics & Observability

This guide describes how to setup multiple InferencePools within a gateway, it assumes familiarity with the concepts and resources covered in the [Getting Started](index.md) guide.

## **Prerequisites**

- A fork of: https://github.com/kubernetes-sigs/gateway-api-inference-extension

## **Steps**

In this guide we will create 2 Inference Pools, but this applies to any number.

1. Create 2 model server deployments with _distinct_ labels. If aquiring adequate accelerators is a blocker, consider the lightweight [sim server](https://github.com/llm-d/llm-d-inference-sim).

2. Using helm, we will install the chart *twice*:

**NOTE:** To confirm requests are sent to each pool, it is reccomended to use the [values](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/config/charts/inferencepool/values.yaml) file, and set the log verbosity flag to 3.

      ```bash
      helm install <NAME-OF-POOL-1> \
      --set inferencePool.modelServers.matchLabels.<label-key-for-first-deployment>=<label-value-for-first-deployment> \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

      ```bash
      helm install <NAME-OF-POOL-2> \
      --set inferencePool.modelServers.matchLabels.<label-key-for-second-deployment>=<label-value-for-second-deployment> \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

3. Using HTTPRoute, we will create 2 match rules each corresponding to an inference pool.

**NOTE:** You could create an httpRoute per Inference Pool. In this example, a single HTTPRoute was used. This is not complete yaml, but an example of how 2 pools can be routed to. In this example, the pools use the same path, but for pool 2, require that a header of `pool-name: <NAME-OF-POOL-2>` be present. 

This allows catch-all behavior for pool 1.

    ```yaml
    apiVersion: gateway.networking.k8s.io/v1
    kind: HTTPRoute

    ...

    rules:
    - backendRefs:
        - group: inference.networking.k8s.io
        kind: InferencePool
        name: <NAME-OF-POOL-1>
        matches:
        - path:
            type: PathPrefix
            value: /
    - backendRefs:
        - group: inference.networking.k8s.io
        kind: InferencePool
        name: <NAME-OF-POOL-2>
        matches:
        - path:
            type: PathPrefix
            value: /
        headers:
        - type: Exact
            name: pool-name
            value: <NAME-OF-POOL-2>
    ```

4. Run requests to each pool

Each inferencePool should be able to serve traffic, to validate we can run the following curl commands:

```bash
curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
"model": "meta-llama/Llama-3.1-8B-Instruct",
"prompt": "Write as if you were a critic: San Francisco",
"max_tokens": 100,
"temperature": 0
}'
```

```bash
curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -H 'pool-name: <NAME-OF-POOL-2>' -d '{
"model": "meta-llama/Llama-3.1-8B-Instruct",
"prompt": "Write as if you were a critic: San Francisco",
"max_tokens": 100,
"temperature": 0
}'
```

Each of these requests should have gone to a different pool, and this can be validated by checking the logs of each EPP that was deployed.