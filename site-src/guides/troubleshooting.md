# Troubleshooting Guide

This guide provides troubleshooting steps and solutions for common issues encountered with the Gateway API inference extension.

## 400 Bad Request

### `model not found in request body` or `prompt not found in request`
If the OpenAI API endpoint you're using isn't working as expected, the issue might be related to the request body format. The endpoint picker (EPP) assumes that if a request is a POST, its body must contain the `model` field and the `prompt` (or `messages`) field. This is because the gateway currently assumes the requests are for Large Language Models (LLMs).

**Solution**: Make sure your request body contains the missing field.

## 404 Not Found
This is a default gateway error, meaning the request never reached a backend service. This usually means that there is no HTTPRoute configured to match the request path (e.g. /v1/completions). The gateway doesn't know where to send the traffic.

**Solution**: Ensure you have an HTTPRoute resource deployed that specifies the correct host, path, and backendRef to your InferencePool.

## 429 Too Many Requests
### `system saturated, sheddable request dropped`
This error indicates that the entire inference pool has exceeded its saturation thresholds. This means the system is under heavy load and is shedding low priority requests. To address this, check the following:

* gateway-api-inference-extension version:
    * **v0.5.1 and earlier**: Verify you're using an `InferenceModel` and that its `criticality` is set to `Critical`. This ensures requests are queued on the model servers instead of being dropped.
    * **v1.0.0 and later**: Ensure the `InferenceObjective` you're using has a `priority` greater than or equal to 0. A negative priority can cause requests to be dropped.

* **Pool Thresholds (Saturation Detector)**: The Gateway uses a pluggable Saturation Detector to assess if the pool is overloaded. By default, it looks at the queue depth and KV cache utilization of your endpoints. If these metrics exceed configured safety limits, the system sheds load. If you are seeing unexpected 429s, your thresholds may be configured too tightly for your workload.

    > **Note:**
    > To learn how to tune these thresholds, change the detector type, or understand how priority interacts with capacity, see our standard guides:
    > - [Saturation Detector Configuration](./epp-configuration/config-text.md#saturation-detector-plugins)
    > - [Priority and Capacity Concepts](../concepts/priority-and-capacity.md)
    > - [Flow Control Tuning Guidance](./flow-control.md#2-tuning-the-saturation-detector-the-healthy-buffer)

## 500 Internal Server Error
### `fault filter abort`
This internal error suggests a misconfiguration in the gateway's backend routing. Your HTTPRoute is configured to point to an InferencePool that does not exist or cannot be found by the gateway. The gateway recognizes the route but fails when trying to send traffic to the non-existent backend.

**Solution**: Verify that the backendRef in your HTTPRoute correctly names an InferencePool resource that is deployed and accessible in the same namespace. If you wish to route to an InferencePool in a different namespace, you can create a `ReferenceGrant` like below:

```
apiVersion: gateway.networking.k8s.io/v1beta1
kind: ReferenceGrant
metadata:
  name: ref-grant
  namespace: ns2
spec:
  from:
    - group: gateway.networking.k8s.io
      kind: HTTPRoute
      namespace: ns1
  to:
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: my-inference-pool
```

## 502 Bad Gateway or 503 Service Unavailable
### `upstream connect error or disconnect/reset ...`
The gateway can return an error when it cannot connect to its backends. This error indicates that the gateway successfully identified the correct model server pod but failed to establish a connection to it. This is likely caused by the port number specified in the InferencePool's configuration doesn't match the port your model server is listening on. The gateway tries to connect to the wrong port and is refused.

**Solution**: Verify the port specified in your InferencePool matches the port number exposed by your model server container, and update your InferencePool accordingly.

## 503 Service Unavailable
### `no healthy upstream`
This error indicates that the HTTPRoute and InferencePool are correctly configured, but there are no healthy pods in the pool to route traffic to. This can happen if the pods are crashing, still starting up, or failing their health checks.

**Solution**: Check the status of your model server pods. Investigate the pod logs for any startup errors or health check failures. Ensure your model server is running and listening on the correct port and that any configured healthchecks / readiness probes are succeeding.

## The endpoint picker (EPP) Crashlooping
When EPP is crashlooping, check the logs of your EPP pod. Some common errors include:

### `failed to list <InferencePool or InferenceObjective or Pod>: … is forbidden`
The EPP needs to watch the InferencePool, InferenceObjectives and Pods that belong to them. This constant watching and reconciliation allows the EPP to maintain an up-to-date view of the environment, enabling it to make dynamic decisions. This particular error indicates that the service account used by the EPP doesn't have the necessary permissions to list the resources it’s watching.

**Solution**: Create or update the RBAC configuration to grant the [required permissions](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/a3f25c07231a47945d52518e979aa7ee386907d3/config/charts/inferencepool/templates/rbac.yaml#L41) to the EPP service account.

### `Pool is not initialized, skipping refreshing metrics`
This error indicates that the Inference Pool pods are not initialized.

**Solution**: Check the EPP start up argument `--pool-name` has the correct InferencePool name specified and the InferencePool exists.

## Unexpected Routing Behaviors
The EPP's core function is to intelligently route requests to the most optimal model server pod in a pool. It uses a score-based algorithm that considers several metrics (such as queue depth, KV cache utilization, etc.) to choose the best pod for each request.

For unexpected routing behaviors:

* Verify the expected metrics are being emitted from the model server. Some model servers aren't fully compatible with the default expected metrics, vLLM is generally the most up-to-date in this regard. See [Support Model Servers](https://gateway-api-inference-extension.sigs.k8s.io/implementations/model-servers/).
* Check your [plugins](https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/config-text/) configuration, especially the weights of the scorer plugins. If weight is omitted, a default weight of 1 will be used.

## Poor Performance under High Concurrency
For more information, check out [EPP scale testing](https://docs.google.com/document/d/1TDD_wvuTO5hhm1Byl8K7TZkNnn8sVQ1ZrkZM_u0gJvw/edit?tab=t.0#heading=h.mtff4cnithxf).

When performance degrades under high load (for example high-latency tail or significantly lower-than-expected successful QPS) with underutilized resources, the issue may be related to excessive logging in the endpoint picker (EPP). Higher verbosity levels (e.g., `--v=2` or greater) generate a large volume of logs. This floods the log buffer and standard output, leading to heavy writelock contention. In extreme cases, this can cause the kubelet to kill the pod due to health check timeouts, leading to a restart cycle.

**Solution**: Ensure log level for the EPP is set to `--v=1`.

## TTFT Latency Spikes Under Low Concurrency

Experiencing high spikes in Time to First Token (TTFT) under low concurrency, despite having underutilized resources, often points to a mismatch between the EPP's `prefix-cache-scorer` configuration and your model server's actual state.

This mismatch causes the EPP to fundamentally misunderstand the caching behavior of your backend:

*   **Overpredicting Cache Hits (Hotspots):** If EPP believes a pod has a prefix match when it doesn't (e.g., due to an incorrect `blockSize` or `lruCapacityPerServer` that is too large), it will aggressively funnel requests to that single pod. The pod effectively experiences a cache miss for every request, recalculating the full prompt while other pods sit idle.
*   **Underpredicting Cache Hits (Missed Opportunities):** Conversely, if `lruCapacityPerServer` is too small, EPP may believe a pod has evicted a prefix that is actually still present. This leads to inefficient routing where requests are sent to suboptimal pods, missing out on "free" acceleration.

**Solution**:
*   **v1.2+**: We recommend using the auto-tuning capability, which automatically syncs configuration with the model server. This feature is currently only supported for model servers that expose the required memory block metrics (e.g., vLLM). It is not currently supported for `sglang` because the necessary metrics regarding memory blocks (`Block Size` and `Number of GPU Blocks`) are not yet exposed.
*   **v1.1 and earlier**: Ensure that your prefix cache scorer parameters are manually tuned to match your model server's settings. You can override these in your `values.yaml` by providing a custom plugin configuration that matches your existing profile structure:

#### Example: Classic Routing (Latency Predictor Disabled)
Use this if you are using the default queue and cache-based scorers.

```yaml
inferenceExtension:
  pluginsConfigFile: "custom-plugins.yaml"
  pluginsCustomConfig:
    custom-plugins.yaml: |
      apiVersion: inference.networking.x-k8s.io/v1alpha1
      kind: EndpointPickerConfig
      plugins:
      - type: queue-scorer
      - type: kv-cache-utilization-scorer
      - type: prefix-cache-scorer
        parameters:
          blockSize: 64 # Match your setup
          maxPrefixBlocksToMatch: 512 # Match your setup
          lruCapacityPerServer: 31250 # Match your setup
      schedulingProfiles:
      - name: default
        plugins:
        - pluginRef: queue-scorer
          weight: 2
        - pluginRef: kv-cache-utilization-scorer
          weight: 2
        - pluginRef: prefix-cache-scorer
          weight: 3
```

#### Example: Latency-Based Routing Enabled
Use this if `inferenceExtension.latencyPredictor.enabled` is set to `true`. See the [Latency-Based Request Scheduling guide](./latency-based-predictor.md) for details on the plugins used below.

```yaml
inferenceExtension:
  pluginsConfigFile: "custom-plugins.yaml"
  pluginsCustomConfig:
    custom-plugins.yaml: |
      apiVersion: inference.networking.x-k8s.io/v1alpha1
      kind: EndpointPickerConfig
      plugins:
      - type: predicted-latency-producer
      - type: prefix-cache-affinity-filter
      - type: latency-scorer
      - type: weighted-random-picker
      schedulingProfiles:
      - name: default
        plugins:
        - pluginRef: predicted-latency-producer
        - pluginRef: prefix-cache-affinity-filter
        - pluginRef: latency-scorer
        - pluginRef: weighted-random-picker
```

See the [Prefix Cache Aware Plugin Customization](https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/prefix-aware/#customize-the-prefix-cache-plugin) for a detailed explanation of each parameter.
