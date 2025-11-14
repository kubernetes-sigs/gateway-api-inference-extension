# SLO-Aware Routing

> For deployment instructions, jump to [Deploying with SLO-Aware Routing](#deploying-with-slo-aware-routing).

SLO-aware routing is a feature of the Inference Gateway that enables intelligent routing of inference requests based on Service Level Objectives (SLOs). It uses a latency predictor to estimate the Time to First Token (TTFT) and Time Per Output Token (TPOT) for each request on each available model server. This allows the gateway to select the optimal server that can meet the request's SLOs, while also considering the overall health and utilization of the model servers.

## How it Works

The SLO-aware routing feature is implemented as a plugin for the Endpoint Picker (EPP). When a request is received, the plugin performs the following steps:

1.  **SLO Extraction**: The plugin extracts the TTFT and TPOT SLOs from the request headers (`x-slo-ttft-ms` and `x-slo-tpot-ms`). It also checks for the `x-prediction-based-scheduling` header to determine if SLO-aware routing should be used for this request.

2.  **Latency Prediction**: The plugin uses a latency predictor, deployed as a set of sidecar containers to the EPP, to predict the TTFT and TPOT for the request on each of the available model servers. The prediction is based on the current state of the server, including its KV cache utilization, and the number of running and waiting requests.

3.  **Headroom Calculation**: For each model server, the plugin calculates the "headroom", which is the difference between the predicted latency and the SLO. A positive headroom means the server is expected to meet the SLO, while a negative headroom means it is not.

4.  **Pod Selection**: The plugin selects a model server based on the calculated headrooms and a configurable selection strategy. The goal is to pick a server that can meet the SLOs without being overloaded.

5.  **Fallback**: If the latency predictor is not available or fails to make a prediction, the plugin falls back to a "composite scoring" mechanism. This mechanism uses a combination of metrics, including prefix cache scores and queue sizes, to make a routing decision.

## Request Headers

To use SLO-aware routing, you need to include the following headers in your inference requests:

-   `x-prediction-based-scheduling`: Set to `true` to enable SLO-aware routing for the request.
-   `x-slo-ttft-ms`: The Time to First Token SLO in milliseconds.
-   `x-slo-tpot-ms`: The Time Per Output Token SLO in milliseconds.

## Headroom Selection Strategies

The SLO-aware routing plugin provides several strategies for selecting a model server based on the calculated headrooms:

-   `least`: (Default) Prefers the pod with the least positive headroom. This strategy is good for packing pods tightly and maximizing utilization.
-   `most`: Prefers the pod with the most positive headroom. This strategy is more conservative and leaves more room for unexpected latency spikes.
-   `composite-least`: A strategy that considers a composite score of various metrics, and prefers the pod with the lowest score.
-   `composite-most`: A strategy that considers a composite score of various metrics, and prefers the pod with the highest score.
-   `composite-only`: This strategy only uses the composite score and ignores latency predictions.

The selection strategy can be configured via the `HEADROOM_SELECTION_STRATEGY` environment variable in the Endpoint Picker deployment.

## Deploying with SLO-Aware Routing

### Prerequisites

Before you begin, ensure you have a functional Inference Gateway with at least one model server deployed. If you haven't set this up yet, please follow the [Getting started guide](../index.md).

### Deployment

To use SLO-aware routing, you must deploy the Endpoint Picker with the latency predictor sidecars. This can be done via the Helm chart by setting the `inferenceExtension.latencyPredictor.enabled` flag to `true`. When this flag is set, the necessary `slo-aware-routing` and `slo-aware-profile-handler` plugins are automatically configured.

For specific deployment instructions and details on configuring environment variables for SLO-aware routing, refer to the [InferencePool Helm Chart README](../../config/charts/inferencepool/README.md#slo-aware-router-environment-variables).

## Monitoring

When SLO-aware routing is enabled, a number of Prometheus metrics are exposed to allow for monitoring and observability of the feature. These metrics provide insight into the performance of the latency predictor and the effectiveness of the SLO-based routing.

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
| `inference_objective_request_ttft_slo_threshold_seconds`   | Current TTFT SLO threshold in seconds for each model and target model.                                           |
| `inference_objective_request_tpot_slo_threshold_seconds`   | Current TPOT SLO threshold in seconds for each model and target model.                                           |
