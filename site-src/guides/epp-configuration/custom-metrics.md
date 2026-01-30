# Custom Metric Scheduling

The Custom Metric Scheduling feature enables the Endpoint Picker (EPP) to use arbitrary Prometheus metrics as scoring signals. This allows for scheduling policies based on hardware telemetry, application-specific counters, or other environmental data not natively tracked by the default plugins purely through configuration, without modifying or recompiling EPP source code.

## Architecture

Custom Metric Scheduling is built on a "Source -> Extractor -> Scorer" pipeline:

1.  **Data Source** (`metrics-data-source`): Fetches raw Prometheus text from the model server's `/metrics` endpoint.
2.  **Extractor** (`prometheus-metric`): Parses the raw text to find a specific metric value, optionally filtering by labels (e.g., `gpu="0"`).
3.  **Scorer** (`metric-scorer`): Uses the extracted value to score the candidate pod.

## Configuration

To enable custom metric scheduling, you must configure all three components in your EPP configuration file.

### 1. Configure the Data Source

First, define a `metrics-data-source` plugin. This plugin manages the HTTP connection to your model server's metrics endpoint.

```yaml
plugins:
- name: model-server-metrics
  type: metrics-data-source
  parameters:
    path: /metrics
    scheme: http
```

### 2. Configure the Extractor

Next, define a `prometheus-metric` extractor. This plugin tells EPP *which* metric to look for.

*   `metricName`: The exact name of the Prometheus metric.
*   `labels`: (Optional) A map of labels to match. This enables **Series Selection**, allowing you to target specific metric series.

```yaml
plugins:
- name: running-requests-extractor
  type: prometheus-metric
  parameters:
    metricName: vllm:num_requests_running
    labels:
      model_name: "llama-3-8b"
```

### 3. Configure the Scorer

Finally, define the `metric-scorer`. This plugin translates the metric value into a scheduling score (0.0 to 1.0).

*   `metricName`: Must match the name used in the Extractor.
*   `optimizationMode`: `Minimize` (lower is better, e.g., latency, temperature) or `Maximize` (higher is better, e.g., throughput, available buffer).
*   `min` / `max`: Expected range for normalization. Values outside this range are clamped.
*   `normalizationAlgo`: `Linear` (default) or `Softmax` (distribution-aware).

```yaml
plugins:
- name: active-load-scorer
  type: metric-scorer
  parameters:
    metricName: vllm:num_requests_running
    optimizationMode: Minimize
    normalizationAlgo: Softmax
    min: 0
    max: 100
```

## Wiring It All Together

Once the plugins are defined, you must "wire" them together in the `data` and `schedulingProfiles` sections.

### Data Section (Wiring Source to Extractor)

You must tell EPP to extract the configured metric from the defined Data Source.

```yaml
data:
  sources:
  - pluginRef: model-server-metrics
    extractors:
    - pluginRef: running-requests-extractor
```

### Scheduling Profiles (Wiring Scorer)

You must add the Scorer to your scheduling profile, assigning it a weight relative to other scorers.

```yaml
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: active-load-scorer
    weight: 40  # Secondary: Avoid overloaded nodes.
  - pluginRef: prefix-cache-scorer
    weight: 60  # Primary: Prefer hot cache (affinity).
```

## Complete Example: Balancing Affinity with Active Load

This example demonstrates a sophisticated policy that balances **Prefix Affinity** (routing requests to where their KV-cache is hot) with **Active Load** (avoiding overloaded servers).


We use `vllm:num_requests_running` with a label filter (`model_name`) to ensure we are only counting load for the specific model we care about. We also use **Softmax** normalization to "softly" penalize load, allowing the Prefix Scorer to win in most cases unless a server is significantly more overloaded than others.

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
featureGates:
- dataLayer # Required for custom metrics.

plugins:
# 1. The Data Source
- name: prometheus-source
  type: metrics-data-source
  parameters:
    path: /metrics
    scheme: http

# 2. The Extractor
# Extracts the number of running requests SPECIFICALLY for Llama-3-8b
# This demonstrates Series Selection: only metrics matching this label will be
# extracted.
- name: running-requests-extractor
  type: prometheus-metric
  parameters:
    metricName: vllm:num_requests_running
    labels:
      model_name: "llama-3-8b"

# 3. The Scorer
# Penalize nodes with a high number of running requests using Softmax.
# Softmax (Minimize) acts as a "Softmin", aggressively penalizing outliers
# without unfairly punishing servers with average load.
- name: active-load-scorer
  type: metric-scorer
  parameters:
    metricName: vllm:num_requests_running
    optimizationMode: Minimize
    normalizationAlgo: Softmax # Use Softmax for distribution-aware scoring.
    min: 0
    max: 50

# 4. Standard Components (including Prefix Cache)
- type: max-score-picker
- type: single-profile-handler
- type: prefix-cache-scorer
  parameters:
    blockSizeTokens: 64

data:
  sources:
  - pluginRef: prometheus-source
    extractors:
    - pluginRef: running-requests-extractor

schedulingProfiles:
- name: default
  plugins:
  - pluginRef: active-load-scorer
    weight: 40  # Secondary: Avoid overloaded nodes.
  - pluginRef: prefix-cache-scorer
    weight: 60  # Primary: Prefer hot cache (affinity).
  - pluginRef: max-score-picker
```
