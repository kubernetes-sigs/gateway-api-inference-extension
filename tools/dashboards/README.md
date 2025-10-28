# Documentation

This documentation provides instructions for setting up grafana dashboards to see metrics emitted from the inference extension and model servers.

## Requirements

Please follow [metrics](https://gateway-api-inference-extension.sigs.k8s.io/guides/metrics-and-observability/) page to configure the proxy to enable all metrics.

## Load Inference Extension dashboard into Grafana

Please follow [grafana instructions](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/import-dashboards/) to load the dashboard json.

## Configure Google Managed Prometheus as source for metrics

If you run the inference gateway with [Google Managed Prometheus](https://cloud.google.com/stackdriver/docs/managed-prometheus), please follow the [instructions](https://cloud.google.com/stackdriver/docs/managed-prometheus/query) to configure Google Managed Prometheus as data source for the grafana dashboard.

## Troubleshooting

### No data in graph

Please configure the `scrape_interval` of your prometheus configuration to lower than `15s`, `rate` function returns empty string if data falls too apart. See https://www.robustperception.io/what-range-should-i-use-with-rate/ for more details.

Example:

```
    global:
      scrape_interval: 5s
```
