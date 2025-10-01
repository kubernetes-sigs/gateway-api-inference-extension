# Inference Gateway: Migrating from v1alpha2 to v1 API

## Introduction

This guide provides a comprehensive walkthrough for migrating your Inference Gateway setup from the alpha `v1alpha2` API to the generally available `v1` API.
This document is intended for platform administrators and networking specialists
who are currently using the `v1alpha2` version of the Inference Gateway and
want to upgrade to the `v1` version to leverage the latest features and improvements.

Before you start the migration, ensure you are familiar with the concepts and deployment of the Inference Gateway.

***

## Before you begin

Before starting the migration, it's important to determine if this guide is necessary for your setup.

### Checking for Existing v1alpha2 APIs

To check if you are actively using the `v1alpha2` Inference Gateway APIs, run the following command:

```bash
kubectl get inferencepools.inference.networking.x-k8s.io --all-namespaces
```

* If this command returns one or more `InferencePool` resources, you are using the `v1alpha2` API and should proceed with this migration guide.
* If the command returns `No resources found`, you are not using the `v1alpha2` `InferencePool` and do not need to follow this migration guide. You can proceed with a fresh installation of the `v1` Inference Gateway.

***

## Migration Paths

There are two paths for migrating from `v1alpha2` to `v1`:

1.  **Simple Migration (with downtime):** This path is for users who can afford a short period of downtime. It involves deleting the old `v1alpha2` resources and CRDs before installing the new `v1` versions.
2.  **Zero-Downtime Migration:** This path is for users who need to migrate without any service interruption. It involves running both `v1alpha2` and `v1` stacks side-by-side and gradually shifting traffic.

***

## Simple Migration (with downtime)

This approach is faster and simpler but will result in a brief period of downtime while the resources are being updated. It is the recommended path if you do not require a zero-downtime migration.

### 1. Delete Existing v1alpha2 Resources

**Option a: Uninstall using Helm.**

```bash
helm uninstall <helm_alpha_inferencepool_name>
```

**Option b: Manually delete alpha `InferencePool` resources.**

If you are not using Helm, you will need to manually delete all resources associated with your `v1alpha2` deployment. The key is to remove the `HTTPRoute`'s reference to the old `InferencePool` and then delete the `v1alpha2` resources themselves.

1.  **Update or Delete the `HTTPRoute`**: Modify the `HTTPRoute` to remove the `backendRef` that points to the `v1alpha2` `InferencePool`.
2.  **Delete the `InferencePool` and associated resources**: You must delete the `v1alpha2` `InferencePool`, any `InferenceModel` (or 'InferenceObjective') resources that point to it, and the corresponding Endpoint Picker (EPP) Deployment and Service.
3.  **Delete the `v1alpha2` CRDs**: Once all `v1alpha2` custom resources are deleted, you can remove the CRD definitions from your cluster.
    ```bash
    # You can change the version to the one you installed `v1alpha2` CRDs
    export VERSION="v0.3.0" 
    kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${VERSION}/manifests.yaml
    ```

### 2. Install v1 Resources

After cleaning up the old resources, you can proceed with a fresh installation of the `v1` Inference Gateway. 
This involves deploying a new EPP image compatible with the `v1` API and installing the new `v1` CRDs. 
You can then create a new v1 InferencePool with its corresponding InferenceObjective resources, and a new HTTPRoute that directs traffic to your new `v1` InferencePool.


### 3. Verify the Deployment

After a few minutes, verify that your new `v1` stack is correctly serving traffic. You should have a **`PROGRAMMED`** gateway.

```bash
❯ kubectl get gateway -o wide
NAME                               CLASS            ADDRESS         PROGRAMMED   AGE
<YOUR_INFERENCE_GATEWAY_NAME>   inference-gateway   <IP_ADDRESS>    True         10m
```

Curl the endpoint to make sure you are getting a successful response with a **200** response code.

```bash
IP=$(kubectl get gateway/<YOUR_INFERENCE_GATEWAY_NAME> -o jsonpath='{.status.addresses[0].value}')
PORT=80

curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
"model": "<your_model>",
"prompt": "<your_prompt>",
"max_tokens": 100,
"temperature": 0
}'
```

***

## Zero-Downtime Migration

This migration path is designed for users who cannot afford any service interruption. Assuming you already have the following stack shown in the diagram

<img src="/images/alpha-stage.png" alt="Inference Gateway Alpha Stage" class="center" />

### A Note on Interacting with Multiple API Versions

During the zero-downtime migration, both `v1alpha2` and `v1` CRDs will be installed on your cluster. This can create ambiguity when using `kubectl` to query for `InferencePool` resources. To ensure you are interacting with the correct version, you **must** use the full resource name:

* **For v1alpha2**: `kubectl get inferencepools.inference.networking.x-k8s.io`
* **For v1**: `kubectl get inferencepools.inference.networking.k8s.io`

The `v1` API also provides a convenient short name, `infpool`, which can be used to query `v1` resources specifically:

```bash
kubectl get infpool
```

This guide will use these full names or the short name for `v1` to avoid ambiguity.

***

### Stage 1: Side-by-side v1 Deployment

In this stage, you will deploy the new `v1` `InferencePool` stack alongside the existing `v1alpha2` stack. This allows for a safe, gradual migration.

After finishing all the steps in this stage, you’ll have the following infrastructure shown in the following diagram

<img src="/images/migration-stage.png" alt="Inference Gateway Migration Stage" class="center" />

**1. Install v1 CRDs**

```bash
RELEASE=v1.0.0
kubectl apply -f [https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/$RELEASE/config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml](https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/$RELEASE/config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml)
```

**2. Install the v1 `InferencePool`**

Use Helm to install a new `v1` `InferencePool` with a distinct release name (e.g., `vllm-llama3-8b-instruct-ga`).

```bash
helm install vllm-llama3-8b-instruct-ga \
  --set inferencePool.modelServers.matchLabels.app=<the_label_you_used_for_the_model_server_deployment> \
  --set provider.name=<YOUR_PROVIDER> \
  --version $RELEASE \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
```

**3. Create the v1 `InferenceObjective`**

The `v1` API replaces `InferenceModel` with `InferenceObjective`. Create the new resources, referencing the new `v1` `InferencePool`.

```yaml
kubectl apply -f - <<EOF
---
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceObjective
metadata:
  name: food-review
spec:
  priority: 1
  poolRef:
    group: inference.networking.k8s.io
    name: vllm-llama3-8b-instruct-ga
---
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceObjective
metadata:
  name: base-model
spec:
  priority: 2
  poolRef:
    group: inference.networking.k8s.io
    name: vllm-llama3-8b-instruct-ga
---
EOF
```

***

### Stage 2: Traffic Shifting

With both stacks running, you can start shifting traffic from `v1alpha2` to `v1` by updating the `HTTPRoute` to split traffic. This example shows a 50/50 split.

**1. Update `HTTPRoute` for Traffic Splitting**

```yaml
kubectl apply -f - <<EOF
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-route
spec:
  parentRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: inference-gateway
  rules:
  - backendRefs:
    - group: inference.networking.x-k8s.io
      kind: InferencePool
      name: vllm-llama3-8b-instruct-alpha
      weight: 50
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: vllm-llama3-8b-instruct-ga
      weight: 50
---
EOF
```

**2. Verify and Monitor**

After applying the changes, monitor the performance and stability of the new `v1` stack. Make sure the `inference-gateway` status `PROGRAMMED` is `True`.

***

### Stage 3: Finalization and Cleanup

Once you have verified that the `v1` `InferencePool` is stable, you can direct all traffic to it and decommission the old `v1alpha2` resources.

**1. Shift 100% of Traffic to the v1 `InferencePool`**

Update the `HTTPRoute` to send all traffic to the `v1` pool.

```yaml
kubectl apply -f - <<EOF
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-route
spec:
  parentRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: inference-gateway
  rules:
  - backendRefs:
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: vllm-llama3-8b-instruct-ga
      weight: 100
EOF
```

**2. Final Verification**

Send test requests to ensure your `v1` stack is handling all traffic as expected.

<img src="/images/ga-stage.png" alt="Inference Gateway GA Stage" class="center" />

You should have a **`PROGRAMMED`** gateway:
```bash
❯ kubectl get gateway -o wide
NAME                  CLASS                                ADDRESS         PROGRAMMED   AGE
inference-gateway   inference-gateway   <IP_ADDRESS>    True         10m
```

Curl the endpoint and verify a **200** response code:
```bash
IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
PORT=80

curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
"model": "<your_model>",
"prompt": "<your_prompt>",
"max_tokens": 100,
"temperature": 0
}'
```

**3. Clean Up v1alpha2 Resources**

After confirming the `v1` stack is fully operational, safely remove the old `v1alpha2` resources.
