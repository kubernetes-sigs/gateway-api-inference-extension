# Latency-Based Routing

> For deployment instructions, jump to [Deploying with Latency-Based Routing](#deploying-with-latency-based-routing).

Latency-based routing is a feature of the Inference Gateway that enables intelligent routing of inference requests based on Service Level Objectives (SLOs) using latency predictions. It uses a latency predictor to estimate the Time to First Token (TTFT) and Time Per Output Token (TPOT) for each request on each available model server. This allows the gateway to select the optimal server that can meet the request's SLOs, while also considering the overall health and utilization of the model servers.

## How it Works

The latency-based routing feature is implemented as a plugin for the Endpoint Picker (EPP). When a request is received, the plugin performs the following steps:

1.  **SLO Extraction**: The plugin extracts the TTFT and TPOT SLOs from the request headers (`x-slo-ttft-ms` and `x-slo-tpot-ms`). It also checks for the `x-prediction-based-scheduling-off` header to determine if latency-based routing should be used for this request.

2.  **Latency Prediction**: The plugin uses a latency predictor, deployed as a set of sidecar containers to the EPP, to predict the TTFT and TPOT for the request on each of the available model servers. The prediction is based on the current state of the server, including its KV cache utilization, and the number of running and waiting requests.

3.  **Headroom Calculation**: For each model server, the plugin calculates the "headroom", which is the difference between the predicted latency and the SLO. A positive headroom means the server is expected to meet the SLO, while a negative headroom means it is not.

4.  **Pod Selection**: The plugin selects a model server based on the calculated headrooms and a configurable selection strategy. The goal is to pick a server that can meet the SLOs without being overloaded.

5.  **Fallback**: If the latency predictor is not available or fails to make a prediction, the plugin falls back to a "composite scoring" mechanism. This mechanism uses a combination of metrics, including prefix cache scores and queue sizes, to make a routing decision.

## Request Headers

To use latency-based routing, you need to include the following headers in your inference requests:

-   `x-prediction-based-scheduling-off`: Include this header to disable predictive routing for that specific request. If omitted, predictive routing is enabled by default.
-   `x-slo-ttft-ms`: The Time to First Token SLO in milliseconds.
-   `x-slo-tpot-ms`: The Time Per Output Token SLO in milliseconds (this is vLLMs equivalent of ITL, is it **not** NTPOT).

## Headroom Selection Strategies

The latency-based routing plugin provides several strategies for selecting a model server based on the calculated headrooms:

-   `least`: (Default) Prefers the pod with the least positive headroom. This strategy is good for packing pods tightly and maximizing utilization.
-   `most`: Prefers the pod with the most positive headroom. This strategy is more conservative and leaves more room for unexpected latency spikes.
-   `composite-least`: A strategy that considers a composite score of various metrics, and prefers the pod with the lowest score.
-   `composite-most`: A strategy that considers a composite score of various metrics, and prefers the pod with the highest score.
-   `composite-only`: This strategy only uses the composite score and ignores latency predictions.

The selection strategy can be configured via the `headroomSelectionStrategy` plugin config variable in the EPP helm chart (see deployment details below).

## Deploying with Latency-Based Routing

### Prerequisites

Before you begin, ensure you have a functional Inference Gateway with at least one model server deployed. If you haven't set this up yet, please follow the [Getting Started Guide](getting-started-latest.md).

### Deployment

To enable latency-based routing, you must enable the latency predictor in the chart and have built the images for the training/prediction sidecars, which are then deployed as containers alongside the Endpoint Picker. When the latency predictor is enabled, the `predicted-latency-scorer` and `predicted-latency-profile-handler` plugins are automatically configured.

#### Steps:

1. Build the predictor and sidecar images from inside the `latencypredictor` package. See the [Latency Predictor - Build Guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/latencypredictor/README.md) for instructions.

2. Set your Docker repository path by replacing the placeholders in Helm chart [values.yaml](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/config/charts/inferencepool/values.yaml) in the format `us-docker.pkg.dev/PROJECT_ID/REPOSITORY` based on what you used to build the sidecars in the Build Guide from step 1.

3. Deploy the chart with the latency predictor enabled by setting `inferenceExtension.latencyPredictor.enabled` to `true` in your `values.yaml` file, or by using the `--set` flag on the command line:

```txt
helm install vllm-llama3-8b-instruct . \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
  --set inferenceExtension.monitoring.gke.enabled=true \
  --set inferenceExtension.latencyPredictor.enabled=true \
  --set provider.name=gke \
  -f values.yaml
```

After these steps, Inference Gateway will be prepared to predict, train, and route requests based on their SLOs.

For details on specific plugin config variables for latency-based routing, refer to the [InferencePool Helm Chart README](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/config/charts/inferencepool/README.md#latency-based-router-configuration).

### Sending Requests

To send a request with Latency-Based Routing, you will need to specify the request SLOs and whether to route or not in the request header. See [Request Headers](#request-headers) section above.

If you have a standard setup via using the [Getting Started Guide](getting-started-latest.md) and then followed the steps outlined above, below is an example inference request with SLOs specified and routing enabled:

```txt
export GW_IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}'):80

curl -v $GW_IP/v1/completions -H 'Content-Type: application/json' -H 'x-slo-ttft-ms: 100' -H 'x-slo-tpot-ms: 100' -d '{
"model": "meta-llama/Llama-3.1-8B-Instruct",
"prompt": "Write as if you were a critic: San Francisco where the ",
"max_tokens": 100,
"temperature": 0, "stream_options": {"include_usage": "true"}, "stream" : "true"
}'
```

## Monitoring

When latency-based routing is enabled, a number of Prometheus metrics are exposed to allow for monitoring and observability of the feature. These metrics provide insight into the performance of the latency predictor and the effectiveness of the SLO-based routing.

Key categories of metrics include:

-   **Actual vs. Predicted Latency**: Metrics for both actual and predicted Time to First Token (TTFT) and Time Per Output Token (TPOT) are available. This allows you to compare the accuracy of the latency predictor.
-   **Prediction Duration**: The time it takes for the latency predictor to generate a prediction is also measured.
-   **SLO Violations**: Counters and gauges are available to track when SLOs are violated. This can be used to alert on SLO breaches.
-   **SLO Thresholds**: The current SLO thresholds for TTFT and TPOT are also exposed as metrics.

NOTE: TPOT is equivalen to vLLM's **ITL** (Inter Token Latency), as vLLM defines TPOT as the average time per output token *including the TTFT*. This is commonly known as NTPOT in other contexts, and we don't capture that metric here.

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
