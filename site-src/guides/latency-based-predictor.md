# Latency-Based Request Scheduling

> For deployment instructions, jump to [Deploying with Latency-Based Request Scheduling](#deploying-with-latency-based-request-scheduling).

Latency-based request scheduling is a feature of the Inference Gateway that enables intelligent scheduling of inference requests using latency predictions. It uses a latency predictor to estimate the Time to First Token (TTFT) and Time Per Output Token (TPOT) for each request on each available model server. This allows the gateway to schedule requests to the server with the lowest predicted latency, and optionally enforce Service Level Objectives (SLOs).

For a deep dive into the design and motivation, see the blog post: [Predicted Latency-Based Scheduling for LLMs](https://llm-d.ai/blog/predicted-latency-based-scheduling-for-llms).

## How it Works

Latency-based request scheduling is implemented as a pipeline of composable plugins for the Endpoint Picker (EPP):

1.  **Latency Prediction** (`predicted-latency-producer`): Calls the latency predictor sidecar to predict TTFT and TPOT for the request on each available model server. Predictions are based on each server's current KV cache utilization, queue depth, and prefix cache match score. If SLO headers are present (`x-slo-ttft-ms`, `x-slo-tpot-ms`), headroom is computed as `SLO - predicted`.

2.  **Prefix Cache Affinity Filtering** (`prefix-cache-affinity-filter`): If any endpoint has a prefix cache match score above the affinity threshold (default 0.80), the filter narrows the candidate set to those endpoints. This improves cache hit rates without hot-spotting.

3.  **Scoring** (`latency-scorer`): Scores endpoints based on predicted latency. In the absence of SLOs, `least` and `most` behave the same way — endpoints with the lowest predicted latency are scored higher. When SLO headers are present, scoring is based on headroom (`SLO - predicted`) and the strategy determines how headroom is used (see [Scoring Strategy](#scoring-strategy)).

4.  **Selection** (`weighted-random-picker`): Selects an endpoint using weighted random selection based on the scores. This provides load spreading while still favoring better-scoring endpoints.

5.  **Fallback**: If the latency predictor is not available or fails to make a prediction, the scorer falls back to a composite scoring mechanism using KV cache utilization, queue depth, and prefix cache scores.

### SLO enforcement

Two additional plugins are included in the default pipeline. They are noop when requests do not include SLO headers, and activate automatically when SLO headers are present:

-   `slo-headroom-tier-filter`: Splits endpoints into positive (meets SLO) and negative (violates SLO) tiers. Probabilistically explores the negative tier to let recovering pods get traffic.
-   `latency-slo-admitter`: Rejects [sheddable](../concepts/priority-and-capacity.md) requests when no endpoint can meet SLO constraints.

#### Scoring Strategy

The `latency-scorer` plugin supports the following strategies for selecting a model server based on predicted latency headroom. These strategies only affect scoring when SLO headers are present; without SLOs, endpoints are always scored by lowest predicted latency regardless of strategy.

-   `least`: (Default) Prefers the endpoint closest to the SLO boundary in either direction — the smallest positive headroom (just meets SLO) or the smallest negative deficit (least overloaded). This strategy is good for bin-packing and maximizing utilization.
-   `most`: Prefers the endpoint with the most headroom. This strategy is more conservative and leaves more room for unexpected latency spikes. Only applies to positive headroom; for negative headroom, `least` is always used.

The strategy can be configured via the `headroomSelectionStrategy` parameter on the `latency-scorer` plugin.

## Request Headers

The following headers can optionally be included in inference requests for SLO-based scheduling:

-   `x-slo-ttft-ms`: The Time to First Token SLO in milliseconds.
-   `x-slo-tpot-ms`: The Time Per Output Token SLO in milliseconds.

When SLO headers are omitted, latency-based request scheduling still works — the scorer schedules to the endpoint with the lowest predicted latency.

## Streaming Mode

The `predicted-latency-producer` plugin has a `streamingMode` parameter (default: `false`) that controls how TTFT and TPOT are measured and trained.

-   **`streamingMode: false`** (default): The system trains on end-to-end request latency. TTFT is recorded at the end of the response and represents the full e2e latency. TPOT is not trained. This is the right default when the workload has a mix of streaming and non-streaming requests, or when no SLO headers are used.

-   **`streamingMode: true`**: The system trains separate TTFT (time to first token) and TPOT (time per output token) models. TTFT is recorded on the first streaming chunk, and TPOT is sampled across subsequent tokens. **Set this when using TTFT/TPOT SLO headers** (`x-slo-ttft-ms`, `x-slo-tpot-ms`) and the workload is streaming. If the workload is a mix of streaming and non-streaming requests, SLO enforcement will not work correctly because TTFT and TPOT cannot be measured for non-streamed responses.

To enable streaming mode, set the `streamingMode` parameter on the `predicted-latency-producer` plugin via `inferenceExtension.pluginsCustomConfig` in your `values.yaml`.

## Deploying with Latency-Based Request Scheduling

### Prerequisites

Before you begin, ensure you have a functional Inference Gateway with at least one model server deployed. If you haven't set this up yet, please follow the [Getting Started Guide](getting-started-latest.md).

### Deployment

To enable latency-based request scheduling, enable the latency predictor in the Helm chart. When enabled, the full plugin pipeline is automatically configured including latency prediction, prefix cache affinity filtering, SLO-aware tier gating, latency scoring, and admission control. The SLO plugins are noop when requests do not include SLO headers.

#### Steps:

1. Deploy the chart with the latency predictor enabled by setting `inferenceExtension.latencyPredictor.enabled` to `true` in your `values.yaml` file, or by using the `--set` flag on the command line:

```txt
helm install vllm-qwen3-32b . \
  --set inferencePool.modelServers.matchLabels.app=vllm-qwen3-32b \
  --set inferenceExtension.monitoring.prometheus.enabled=true \
  --set inferenceExtension.latencyPredictor.enabled=true \
  --set provider.name=gke \
  -f values.yaml
```

After these steps, Inference Gateway will be prepared to predict, train, and schedule requests based on predicted latency. SLO headers are optional — see [Request Headers](#request-headers).

Each plugin accepts its own parameters. For the full list of per-plugin configuration options, refer to the README in each plugin's package directory under `pkg/epp/framework/plugins/`. To override the default plugin configuration, provide a full custom config via `inferenceExtension.pluginsCustomConfig` in your `values.yaml`.

### Sending Requests

Latency-based request scheduling works out of the box without any special request headers. To additionally enforce SLO constraints, include the SLO headers as described in [Request Headers](#request-headers).

If you have a standard setup via using the [Getting Started Guide](getting-started-latest.md) and then followed the steps outlined above, below is an example inference request with SLOs specified:

```txt
export GW_IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}'):80

curl -v $GW_IP/v1/completions -H 'Content-Type: application/json' -H 'x-slo-ttft-ms: 100' -H 'x-slo-tpot-ms: 100' -d '{
"model": "Qwen/Qwen3-32B",
"prompt": "Write as if you were a critic: San Francisco",
"max_tokens": 100,
"temperature": 0, "stream_options": {"include_usage": "true"}, "stream" : "true"
}'
```

## Monitoring

When latency-based request scheduling is enabled, a number of Prometheus metrics are exposed to allow for monitoring and observability of the feature. These metrics provide insight into the performance of the latency predictor and the effectiveness of the SLO-based scheduling.

Key categories of metrics include:

-   **Actual vs. Predicted Latency**: Metrics for both actual and predicted Time to First Token (TTFT) and Time Per Output Token (TPOT) are available. This allows you to compare the accuracy of the latency predictor.
-   **Prediction Duration**: The time it takes for the latency predictor to generate a prediction is also measured.
-   **SLO Violations**: Counters and gauges are available to track when SLOs are violated. This can be used to alert on SLO breaches.
-   **SLO Thresholds**: The current SLO thresholds for TTFT and TPOT are also exposed as metrics.

The following is a comprehensive list of the Prometheus metrics exposed:

| Metric Name                                                | Description                                                                                                      |
| :--------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| `inference_objective_request_ttft_seconds`                 | Inference model TTFT distribution in seconds for each model and target model.                                    |
| `inference_objective_request_ttft_seconds_gauge`           | Inference model TTFT gauge in seconds for each model and target model.                                           |
| `inference_objective_request_predicted_ttft_seconds`       | Inference model Predicted TTFT distribution in seconds for each model and target model.                          |
| `inference_objective_request_predicted_ttft_seconds_gauge` | Inference model Predicted TTFT gauge in seconds for each model and target model.                                 |
| `inference_objective_request_ttft_prediction_duration_seconds` | Duration taken to generate TTFT predictions in seconds for each model and target model.                          |
| `inference_objective_request_ttft_prediction_duration_seconds_gauge` | Latest duration taken to generate TTFT predictions in seconds for each model and target model.                     |
| `inference_objective_request_tpot_seconds`                 | Inference model TPOT distribution in seconds for each model and target model.                                    |
| `inference_objective_request_tpot_seconds_gauge`           | Inference model TPOT gauge in seconds for each model and target model.                                           |
| `inference_objective_request_predicted_tpot_seconds`       | Inference model Predicted TPOT distribution in seconds for each model and target model.                          |
| `inference_objective_request_predicted_tpot_seconds_gauge` | Inference model Predicted TPOT gauge in seconds for each model and target model.                                 |
| `inference_objective_request_tpot_prediction_duration_seconds` | Duration taken to generate TPOT predictions in seconds for each model and target model.                          |
| `inference_objective_request_tpot_prediction_duration_seconds_gauge` | Latest duration taken to generate TPOT predictions in seconds for each model and target model.                     |
| `inference_objective_request_ttft_slo_violation`           | Boolean indicator (0 or 1) of whether the last TTFT measurement violated the SLO threshold for each model and target model. |
| `inference_objective_request_ttft_slo_violation_total`     | Counter of TTFT SLO violations for each model and target model.                                                  |
| `inference_objective_request_tpot_slo_violation`           | Boolean indicator (0 or 1) of whether the last TPOT measurement violated the SLO threshold for each model and target model. |
| `inference_objective_request_tpot_slo_violation_total`     | Counter of TPOT SLO violations for each model and target model.                                                  |
