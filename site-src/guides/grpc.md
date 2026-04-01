# End-to-End gRPC Guide for GKE Gateway

This guide covers how to use gRPC with the Inference Extension (EPP) and GKE Gateway. The setup ensures that gRPC traffic is routed through the EPP, which can parse request payloads and make routing decisions based on model metrics (e.g., queue size, KV cache usage).

## Prerequisites
*   A running GKE cluster with Gateway API enabled.
*   Helm v3 installed.
*   A model server (such as vLLM) that supports gRPC and is listening for requests.

## Configuration

To deploy the `inferencepool` chart with gRPC support, configure your `values.yaml` file as follows.

### Example `values.yaml`
```yaml
inferencePool:
  targetPorts:
    - number: 8000
  modelServerType: vllm                      # vllm is the default
  modelServerProtocol: grpc                 # Explicitly set to grpc to use vllmgrpc-parser
```

### How it Works
When you set `modelServerProtocol: grpc` (and type is `vllm`), the `epplib` chart automatically configures the `EndpointPickerConfig` with the `vllmgrpc-parser`. This parser treats incoming request bodies as `vllm.GenerateRequest` and can extract prompt tokens, weights, etc.

Here is the generated `EndpointPickerConfig` inside the configMap:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: maxScore
  type: max-score-picker
- name: vllmgrpcParser
  type: vllmgrpc-parser
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
parser:
  pluginRef: vllmgrpcParser
```

## Deployment

Install the `inferencepool` chart using Helm:

```bash
helm install vllm-qwen3-32b ./config/charts/inferencepool \
  --set inferencePool.modelServers.matchLabels.app=vllm-qwen3-32b \
  --set inferencePool.modelServerProtocol=grpc
```

## Traffic Routing with GKE Gateway

The `inferencepool` chart can deploy a custom `HTTPRoute` for routing traffic to the pool. GKE Gateway typically uses `HTTPRoute` for gRPC traffic as well, by matching on paths or headers.

Here is an example of the deployed `HTTPRoute`:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: vllm-qwen3-32b
spec:
  parentRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: inference-gateway
  rules:
  - backendRefs:
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: vllm-qwen3-32b
    matches:
    - path:
        type: PathPrefix
        value: /
```

## Verification

To verify that the setup is working, you can use `grpcurl` or create a python gRPC client to send requests.

### Using `grpcurl`

If you have `grpcurl` installed, you can send a request to the Gateway IP:

```bash
grpcurl -plaintext -d '{"input": {"text": "Hello, world!"}}' \
  <GATEWAY_IP>:<GATEWAY_PORT> vllm.VllmService/Generate
```

### Automated Testing
For reference, there are integration tests in the repository that exercise gRPC parsing and routing in the EPP. Check [grpc_test.go](file:///usr/local/google/home/bobzetian/projects/modelrewriteuserguide/test/integration/epp/grpc_test.go) for examples of how to verify using Go testing framework.
