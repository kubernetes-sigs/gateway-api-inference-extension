# Documentation

This documentation provides instructions for setting up grafana dashboards to see metrics emitted from the inference extension, body based router, and model servers.

## Requirements

Please follow [metrics](https://gateway-api-inference-extension.sigs.k8s.io/guides/metrics-and-observability/) page to configure the proxy to enable all metrics.

## Load Inference Extension dashboard into Grafana

Please follow [grafana instructions](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/import-dashboards/) to load the dashboard json.

## Configure Google Managed Prometheus as source for metrics

If you run the inference gateway with [Google Managed Prometheus](https://cloud.google.com/stackdriver/docs/managed-prometheus), please follow the [instructions](https://cloud.google.com/stackdriver/docs/managed-prometheus/query) to configure Google Managed Prometheus as data source for the grafana dashboard.

## Flow Control Dashboard Panels

The dashboard provides a dedicated row for the **Admission and Flow Control Layer** when enabled. It includes the following panels to help you monitor pool queuing and fairness:

![Flow Control Dashboard](../../site-src/images/flow_control_dashboard.png)

-   **Pool Saturation**: Current capacity vs protection (threshold at 1.0).
-   **Net Queue Flow**: Derivative of queue size (if it is growing or draining).
-   **Request Arrival Rate**: Per-tenant true user demand.
-   **Queue Depth & Memory Mass**: Counts and payload sizes held in EPP memory.
-   **Dispatch Outcomes**: Successful dispatches, rejections, and evictions.
-   **Wait Time (P99)**: Latency introduced by queuing.
-   **Internal Processing Latency**: Overhead of the flow control logic.
-   **Fairness Sharing Rate**: Visualization of equal resource convergence under contention.

## Troubleshooting

### No data in graph

Please configure the `scrape_interval` of your prometheus configuration to lower than `15s`, `rate` function returns empty string if data falls too apart. See https://www.robustperception.io/what-range-should-i-use-with-rate/ for more details.

Example:

```
    global:
      scrape_interval: 5s
```
