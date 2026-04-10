# Configuring via YAML

The Inference Gateway (IGW) can be configured via a YAML file.

At this time the YAML file based configuration allows for:

1. The set of the lifecycle hooks (plugins) that are used by the IGW.
2. The set of scheduling profiles that define how requests are scheduled to pods.
3. The configuration of the saturation detector.
4. The configuration of the Flow Control system.
5. The configuration of the data layer.
6. The configuration of the payload parser (used to understand requests and responses).
7. A set of feature gates that are used to enable experimental features.

The YAML file can either be specified as a path to a file or in-line as a parameter.

***NOTE***: While the configuration text looks like a Kubernetes CRD, it is
**NOT** a Kubernetes CRD. Specifically, the config is not reconciled upon, and is only read on startup.
This behavior is intentional, as augmenting the scheduling config without redeploying the EPP is not supported.

The configuration text has the following form:
```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- ....
- ....
schedulingProfiles:
- ....
- ....
saturationDetector:
  ...
data:
  ...
flowControl:
  ...
parser:
  ...
featureGates:
  ...
```

The first two lines of the configuration are constant and must appear as is.

The `featureGates` section allows the enablement of experimental features of the IGW. This section is described in more
detail in the section [Feature Gates](#feature-gates).

The `plugins` section defines the set of plugins that will be instantiated and their parameters. This section is described in more detail in the section [Plugin Configuration](#plugin-configuration).

The `schedulingProfiles` section defines the set of scheduling profiles that can be used in scheduling
requests to pods. This section is described in more detail in the section [Scheduling Profiles](#scheduling-profiles).

The `saturationDetector` section configures the saturation detector, which is used to determine if special action needs
to be taken due to the system being overloaded or saturated. This section is described in more detail in the section
[Saturation Detector configuration](#saturation-detector-configuration).

The `flowControl` section configures the Flow Control layer, which manages request concurrency and
fairness. This section is described in more detail in the section
[Flow Control configuration](#flow-control-configuration).

The `data` section configures the data layer, which is used to gather information (such as metrics) used in making scheduling
decisions. This section is described in more detail in the section [Data Layer configuration](#data-layer-configuration).

The `parser` section configures the parser, which is used to understand the payload of requests and responses for features like prefix-cache aware routing and usage tracking. This section is described in more detail in the section [Parser Configuration](#parser-configuration).

A complete configuration might look like this:
```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: prefix-cache-scorer
  parameters:
    blockSizeTokens: 5
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 31250
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefix-cache-scorer
    weight: 50
```

If the configuration is in a file, the EPP command line argument `--config-file`
should be used to specify the full path of the file in question. For example:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${EPP_NAME}
  ...
spec:
  ...
  template:
    ...
    spec:
      ...
      containers:
      - name: epp
        image: ghcr.io/llm-d/llm-d-inference-scheduler:latest
        imagePullPolicy: IfNotPresent
        args:
        - --pool-name
        - "${POOL_NAME}"
        ...
        - --config-file
        - "/etc/epp/epp-config.yaml"
```

If the configuration is passed as in-line text the EPP command line argument `--config-text`
should be used. For example:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${EPP_NAME}
  ...
spec:
  ...
  template:
    ...
    spec:
      ...
      containers:
      - name: epp
        image: ghcr.io/llm-d/llm-d-inference-scheduler:latest
        imagePullPolicy: IfNotPresent
        args:
        - --pool-name
        - "${POOL_NAME}"
        ...
        - --config-text
        - |
          apiVersion: inference.networking.x-k8s.io/v1alpha1
          kind: EndpointPickerConfig
          plugins:
          - type: prefix-cache-scorer
            parameters:
              blockSizeTokens: 5
              maxPrefixBlocksToMatch: 256
              lruCapacityPerServer: 31250
          schedulingProfiles:
          - name: default
            plugins:
            - pluginRef: prefix-cache-scorer
              weight: 50
```

The EPP configuration in the above two examples are equivalent to the following configuration after
defaults have been applied:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: prefix-cache-scorer
  parameters:
    blockSizeTokens: 5
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 31250
- type: single-profile-handler
- type: max-score-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefix-cache-scorer
    weight: 50
  - pluginRef: max-score-picker
```

## Plugin Configuration

The set of plugins that are used by the IGW is determined by how it is configured. The IGW is
primarily configured via a configuration file.

The configuration defines the set of plugins to be instantiated along with their parameters.
Each plugin can also be given a name, enabling the same plugin type to be instantiated multiple
times, if needed (such as when configuring multiple scheduling profiles).

The `plugins` section defines the set of plugins that will be instantiated and their parameters.
Each entry in this section has the following form:

```yaml
- name: aName
  type: a-type
  parameters:
    parm1: val1
    parm2: val2
```

The fields in a plugin entry are:

- *name* which is optional, provides a name by which the plugin instance can be referenced. If this
field is omitted, the plugin's type will be used as its name.
- *type* specifies the type of the plugin to be instantiated.
- *parameters* which is optional, defines the set of parameters used to configure the plugin in question.
The actual set of parameters varies from plugin to plugin.

The available plugins are categorized below by their function.

### Infrastructure Plugins

The set of plugins instantiated can include a Profile Handler, which determines which SchedulingProfiles
will be used for a particular request. A Profile Handler must be specified, unless the configuration only
contains one profile, in which case the `SingleProfileHandler` will be used.

#### SingleProfileHandler

Selects a single profile which is always the primary profile.

- *Type*: single-profile-handler
- *Parameters*: none

### Scheduling Plugins (Scorers & Pickers)

The set of instantiated plugins can also include a picker, which chooses the actual pod to which
the request is scheduled after filtering and scoring. If one is not referenced in a `SchedulingProfile`, an
instance of `MaxScorePicker` will be added to the SchedulingProfile in question. For a high-level overview of available pickers, see the [Scheduling Pickers](../../../pkg/epp/framework/plugins/scheduling/picker/README.md) guide.

These plugins are referenced within the `schedulingProfiles` section.

#### [PrefixCache Scorer](../../../pkg/epp/framework/plugins/scheduling/scorer/prefix/README.md)

Scores pods based on the amount of the prompt is believed to be in the pod's KvCache.

- *Type*: prefix-cache-scorer
- *Parameters*:
  - `blockSize` specified the size of the blocks to break up the input prompt when
    calculating the block hashes. If not specified defaults to `64`
  - `maxPrefixBlocksToMatch` specifies the maximum number of prefix blocks to match. If
   not specified defaults to `256`
  - `lruCapacityPerServer` specifies the capacity of the LRU indexer in number of entries
    per server (pod). If not specified defaults to `31250`

#### [LoRAAffinity Scorer](../../../pkg/epp/framework/plugins/scheduling/scorer/loraaffinity/README.md)


Scores pods based on whether the requested LoRA adapter is already loaded in the pod's HBM, or if
the pod is ready to load the LoRA on demand.

- *Type*: lora-affinity-scorer
- *Parameters*: none

#### [KvCacheUtilization Scorer](../../../pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization/README.md)

Scores the candidate pods based on their KV cache utilization.

- *Type*: kv-cache-utilization-scorer
- *Parameters*: none

#### [QueueDepth Scorer](../../../pkg/epp/framework/plugins/scheduling/scorer/queuedepth/README.md)

Scores list of candidate pods based on the pod's waiting queue size. The lower the
waiting queue size the pod has, the higher the score it will get (since it's more
available to serve new request).

- *Type*: queue-scorer
- *Parameters*: none

#### [RunningRequest Scorer](../../../pkg/epp/framework/plugins/scheduling/scorer/runningrequests/README.md)

Scores candidate pods based on the number of requests currently being processed (in-flight) on
each pod. Pods with fewer running requests receive a higher score. Scores are normalized across
the candidate set — the pod with the fewest running requests scores `1.0`, the pod with the most
scores `0.0`, and all others are linearly interpolated. When all candidates have the same count,
every pod receives a neutral score of `1.0`.

- *Type*: running-requests-size-scorer
- *Parameters*: none

#### [MaxScorePicker](../../../pkg/epp/framework/plugins/scheduling/picker/maxscore/README.md)

Picks the pod with the maximum score from the list of candidates. This is the default picker plugin
if not specified.

- *Type*: max-score-picker
- *Parameters*:
  - `maxNumOfEndpoints`: Maximum number of endpoints to pick from the list of candidates, based on
    the scores of those endpoints. If not specified defaults to `1`.

#### [RandomPicker](../../../pkg/epp/framework/plugins/scheduling/picker/random/README.md)

Picks a random pod from the list of candidates.

- *Type*: random-picker
- *Parameters*:
  - `maxNumOfEndpoints`: Maximum number of endpoints to pick from the list of candidates. If not
    specified defaults to `1`.

#### [WeightedRandomPicker](../../../pkg/epp/framework/plugins/scheduling/picker/weightedrandom/README.md)

Picks pod(s) from the list of candidates based on weighted random sampling using A-Res algorithm.

- *Type*: weighted-random-picker
- *Parameters*:
  - `maxNumOfEndpoints`: Maximum number of endpoints to pick from the list of candidates. If not
    specified defaults to `1`.

### Flow Control Plugins (Policies)

These plugins are referenced within the `flowControl` section (Priority Bands). This section includes policies for **[fairness](../../../pkg/epp/framework/plugins/flowcontrol/fairness/README.md)** and **[ordering](../../../pkg/epp/framework/plugins/flowcontrol/ordering/README.md)**.

#### [GlobalStrictFairnessPolicy](../../../pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict/README.md)

A Fairness Policy that ignores flow isolation and serves all requests in a single global order, delegating the actual ordering to the configured `OrderingPolicy`. It acts effectively as a single global queue per priority band.

- **Unit of Fairness**: None (Greedy)
- **Type**: `global-strict-fairness-policy`
- **Parameters**: none

> [!NOTE]
> This policy prioritizes absolute ordering over fairness. A single flow with a burst of high-priority traffic can starve others.

#### [RoundRobinFairnessPolicy](../../../pkg/epp/framework/plugins/flowcontrol/fairness/roundrobin/README.md)

A Fairness Policy that guarantees fair sharing of dispatch opportunities between different flows (e.g., different models or LoRA adapters) by cycling through active flows one by one.

- **Unit of Fairness**: Dispatch Attempts
- **Type**: `round-robin-fairness-policy`
- **Parameters**: none

> [!NOTE]
> While this policy prevents starvation, it may introduce global ordering violations, as a newer request in an under-served flow might be dispatched before an older request in a heavily loaded flow.

#### [FCFSOrderingPolicy](../../../pkg/epp/framework/plugins/flowcontrol/ordering/fcfs/README.md)

An Ordering Policy that implements First-Come, First-Served ordering based on logical arrival time. This is the default Ordering Policy.

- *Type*: fcfs-ordering-policy
- *Parameters*: none

#### [EDFOrderingPolicy](../../../pkg/epp/framework/plugins/flowcontrol/ordering/edf/README.md)

An Ordering Policy that implements Earliest Deadline First. It prioritizes requests with the closest expiration time (deadline).

- *Type*: edf-ordering-policy
- *Parameters*: none

#### [SLODeadlineOrderingPolicy](../../../pkg/epp/framework/plugins/flowcontrol/ordering/slodeadline/README.md)

An Ordering Policy that orders requests by an SLO-based deadline, computed from the time the request is received by the server. It prioritizes requests with the earliest such deadline.

- *Type*: slo-deadline-ordering-policy
- *Parameters*: none

### Saturation Detector Plugins

> **Note:** To see how to reference these plugins in your configuration, see [Saturation Detector Configuration](#saturation-detector-configuration).

These plugins are used to interpret system load and protect endpoints from overload. They are referenced in the `saturationDetector` section.

#### [Utilization Detector Plugin](../../../pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization/README.md)

This is the default saturation detector. It closed-loop reacts to telemetry emitted by individual model servers. It evaluates queue depth and KV cache utilization against user thresholds to score global saturation.

- **Type**: `utilization-detector`
- **Parameters**:
  - `queueDepthThreshold` (`int`): Target waiting queue depth limit. Serves as the "ideal" queue capacity for a single endpoint. Must be > 0. (Default: `5`)
  - `kvCacheUtilThreshold` (`float64`): Target KV cache memory utilization limit, expressed as a fraction. Must be in `(0.0, 1.0]`. (Default: `0.8`)
  - `metricsStalenessThreshold` (`string` duration): Maximum age of metrics before an endpoint is considered stale (e.g., `"150ms"`). Must be > 0. (Default: `"200ms"`)
  - `headroom` (`float64`): Allowed burst capacity above the ideal thresholds, expressed as a fraction (e.g., `0.2` for 20%). Must be >= 0.0. (Default: `0.0`)

#### [Concurrency Detector Plugin](../../../pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency/README.md)

Synchronous saturation detection mechanism based on active in-flight request accounting. Open-loop calculation of pool load with local endpoint limiting.

- **Type**: `concurrency-detector`
- **Parameters**:
  - `concurrencyMode` (`string`): Evaluation mode. Valid values are `"requests"` or `"tokens"`. (Default: `"requests"`)
  - `maxConcurrency` (`int64`): Maximum requests in flight. Serves as the "ideal" request capacity for a single endpoint. Must be > 0. (Default: `100`)
  - `maxTokenConcurrency` (`int64`): Maximum tokens in flight. The "tokens" mode equivalent of `maxConcurrency`. Must be > 0. (Default: `1000000`)
  - `headroom` (`float64`): Allowed burst capacity above the ideal threshold, expressed as a fraction (e.g., `0.2` for 20%). Must be >= 0.0. (Default: `0.0`)

## Scheduling Profiles


The `schedulingProfiles` section defines the set of scheduling profiles that can be used in scheduling
requests to pods. If one is not defined, a default one named `default` will be added and will reference all of
the instantiated plugins.

The number of scheduling profiles depends on the use case. For simple
serving of requests, one is enough. For disaggregated prefill, two profiles are required. Each entry
in this section has the following form:

```yaml
- name: aName
  plugins:
  - pluginRef: plugin1
  - pluginRef: plugin2
    weight: 50
```

The fields in a schedulingProfile entry are:

- *name* specifies the scheduling profile's name.
- *plugins* specifies the set of plugins to be used when this scheduling profile is chosen for a request.
Each entry in the schedulingProfile's plugins section has the following fields:
  - *pluginRef* is a reference to the name of the plugin instance to be used
  - *weight* is the weight to be used if the referenced plugin is a scorer. If omitted, a weight of one
    will be used.

## Saturation Detector Configuration

> **Note:** For a full list of available plugins and their parameters, see [Saturation Detector Plugins](#saturation-detector-plugins).

The Saturation Detector acts as a safety valve, continuously evaluating if the backend `InferencePool` is overloaded. It protects the pool from overload, ensuring that endpoints operate within their optimal capacity limits.

How the gateway reacts to a "saturated" signal depends strictly on the `flowControl` feature gate:

* **Flow Control Enabled**: The Saturation Detector acts as the gatekeeper for the gateway's centralized queues. When the pool is saturated, the gateway pauses dispatching and safely buffers incoming requests in memory (respecting priority and fairness policies) until capacity frees up on the backends.
* **Flow Control Disabled (Default)**: The gateway lacks centralized queuing. When the pool is saturated, the gateway immediately rejects (HTTP 503) incoming "sheddable" requests (those with a negative priority) to protect the backends. All other requests are passed directly to the model servers. For more details, see the [Priority and Capacity](../../concepts/priority-and-capacity.md) guide.

The Saturation Detector is configured via the `saturationDetector` section by referencing a plugin defined in the global `plugins` list.

It has the following form:

```yaml
plugins:
- type: utilization-detector
  parameters:
    queueDepthThreshold: 8
saturationDetector:
  pluginRef: utilization-detector
```

The fields in the `saturationDetector` section are:

- `pluginRef`: The name of the plugin instance to use for saturation detection. If omitted or empty, the system defaults to using the `utilization-detector`. *Note: If a `utilization-detector` is not explicitly defined in your `plugins` array, the gateway will automatically instantiate one under the hood using standard default parameters.*

## [Flow Control Configuration](../flow-control.md)

The Flow Control layer acts as a pool defense mechanism, shielding inference engines from overload to ensure stable,
predictable performance. By shifting Head-of-Line blocking left, it buffers requests before they reach the backends,
enabling the Scheduler to dispatch work only when capacity is available.

It manages traffic through a **3-Tier Dispatch Hierarchy**:

1.  **Priority (Band Selection)**: High-priority traffic is served first.
2.  **Fairness (Flow Selection)**: Requests are distributed fairly between tenants/models within a priority level
    (e.g., Round Robin).
3.  **Ordering (Request Selection)**: Requests within a flow are ordered (e.g., FCFS).

This ensures effective multi-tenancy, minimizes scheduling regret, and prevents resource exhaustion. This configuration
is only respected if the `flowControl` feature gate is enabled.

The Flow Control layer is configured via the `flowControl` section of the overall configuration. It has the following
form:

```yaml
flowControl:
  maxBytes: 10Gi # 10737418240 bytes
  defaultRequestTTL: 60s
  defaultPriorityBand:
    maxBytes: 10Gi
    maxRequests: 5000
  priorityBands:
  - priority: 100
    maxBytes: 5Gi
    maxRequests: 500
    orderingPolicyRef: fcfs-ordering-policy
    fairnessPolicyRef: global-strict-fairness-policy
```

The fields in the `flowControl` section are:

- `maxBytes`: Defines the global capacity limit for all active requests across all priority levels.
    - Supports Kubernetes quantity format (e.g., `10Gi`, `512Mi`, `1048576Ki`) as well as plain integers (in bytes).
    - If `0` or omitted, no global limit is enforced (unlimited), though individual priority band limits still apply.
- `maxRequests`: Defines an optional global maximum total request count limit aggregated across all priority bands.
    - Supports Kubernetes quantity format (e.g., `50Mi`, `5000Ki`) as plain integers.
    - If `0` or omitted, no global request count limit is enforced, though individual priority band `maxRequests` limits still apply.
- `defaultRequestTTL`: A fallback timeout for requests that do not specify their own deadline.
    - If `0` or omitted, it defaults to the client context deadline, meaning requests may wait indefinitely unless cancelled by the client.
- `defaultPriorityBand`: A template used to dynamically provision priority bands for requests arriving with priority
  levels not explicitly configured in `priorityBands`.
- `priorityBands`: A list of explicit configurations for specific priority levels.

### Priority Band Configuration

Both the `defaultPriorityBand` template and the entries in `priorityBands` use the following fields:

- `priority`: (Required for `priorityBands` entries) The integer priority level. Higher values indicate higher priority.
- `maxBytes`: The maximum aggregate byte size allowed for this specific priority band.
    - Supports Kubernetes quantity format (e.g., `5Gi`, `512Mi`) as well as plain integers (in bytes).
    - If `0` or omitted, the system default (1 GB) is used.
- `maxRequests`: The maximum number of concurrent requests allowed for this specific priority band.
    - Supports Kubernetes quantity format (e.g., `50Mi`, `5000Ki`) as plain integers.
    - If `0` or omitted, no per-band request count limit is enforced for this band.
- `orderingPolicyRef`: The name of the Ordering Policy plugin to use (e.g., `fcfs-ordering-policy`).
    - Defaults to `fcfs-ordering-policy` if omitted.
- `fairnessPolicyRef`: The name of the Fairness Policy plugin to use (e.g., `global-strict-fairness-policy`).
    - Defaults to `global-strict-fairness-policy` if omitted.

## Data Layer configuration

The Data Layer collects metrics and other data used in scheduling decisions made by the various configured
plugins. The exact data collected varies by the DataSource and Extractors configured. The baseline
provided in GAIE collects Prometheus metrics from the Model Servers in the InferencePool.

The Data Layer is configured via the `data` section of the overall configuration. It has the following form:

```yaml
data:
  sources:
  - pluginRef: source1
    extractors:
    - pluginRef: extractor1
    - pluginRef: extractor2
```

The data section has one field *sources* which configures the set of DataSources to be used to gather the metrics
and other data used for scheduling.

Each entry in the sources list has the following fields:

- *pluginRef* is a reference to the name of the plugin instance to be used.
- *extractors* specifies the list of the extractors to be used with this DataSource. Each entry in the extractors
list has the following field:
  - *pluginRef* is a reference to the name of the plugin instances to be used.

**Note**: The names of the plugin instances mentioned above, refer to plugin instances defined in the plugins section
of the configuration.

### Minimal configuration (core vLLM metrics)

The built-in `metrics-data-source` and `core-metrics-extractor` plugins collect the five vLLM Model
Server Protocol metrics out of the box - no `parameters` are required:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: metrics-data-source
  - type: core-metrics-extractor

data:
  sources:
    - pluginRef: metrics-data-source
      extractors:
        - pluginRef: core-metrics-extractor

schedulingProfiles:
  ...
  ...
```

Default metric specs collected for vLLM:

| Field | Default spec |
|---|---|
| Queued requests | `vllm:num_requests_waiting` |
| Running requests | `vllm:num_requests_running` |
| KV-cache utilization | `vllm:kv_cache_usage_perc` |
| LoRA adapter info | `vllm:lora_requests_info` |
| Cache block config | `vllm:cache_config_info` |

SGLang defaults are also built in and selected automatically when a Pod carries the
`inference.networking.k8s.io/engine-type: sglang` label. The label value or default engine
can be set by via the [extractor configuration](#core-metrics-extractor-parameters-reference).

### Disabling a specific metric

Set the corresponding spec field to an empty string inside an `engineConfigs` entry.

**Important: `engineConfigs` is full-replacement per engine name.**
Providing any entry with `name: "vllm"` replaces the _entire_ built-in vllm
configuration — the built-in defaults for that engine are not merged in.
You must restate every spec field you still want to collect.

#### Example: disable LoRA collection for vLLM

```yaml
plugins:
  - name: core-metrics-extractor
    type: core-metrics-extractor
    parameters:
      engineConfigs:
        - name: vllm
          queuedRequestsSpec:  "vllm:num_requests_waiting"
          runningRequestsSpec: "vllm:num_requests_running"
          kvUsageSpec:         "vllm:kv_cache_usage_perc"
          loraSpec:            ""    # empty string disables LoRA extraction
          cacheInfoSpec:       "vllm:cache_config_info"
```

When `loraSpec` is empty, no extraction is attempted. Empty strings produce nil metric
specifications, so you must restate every field you want to keep and not just those being
changed. At startup the factory logs a `"Not scraping metric"` line for each disabled
spec so operators can confirm which metrics are intentionally skipped.


### `data.sources: []` - empty source list

An explicit `sources: []` (non-nil but empty) is valid and silently disables all metric collection.
A warning is logged at startup.

This is distinct from `data: null` (or omitting the section entirely), which is an error when the
datalayer is enabled.

| `data` value | Behavior |
|---|---|
| omitted / `null`, datalayer **enabled** (`dataLayer` feature gate) | startup error: _"You must specify the Data section"_ |
| omitted / `null`, datalayer **disabled** (default) | no error, no collection |
| `sources: []` (empty list) | warning logged, no collection |
| `sources` references unknown plugin name | startup error |

### `core-metrics-extractor` parameters reference

All fields are optional and fall back to the built-in vLLM / SGLang defaults.

```yaml
parameters:
  # Pod label used to select which engineConfig to apply.
  # Default: "inference.networking.k8s.io/engine-type"
  engineLabelKey: "inference.networking.k8s.io/engine-type"

  # Engine to use for Pods without the engineLabelKey label.
  # Must match a name in engineConfigs. Default: "vllm"
  defaultEngine: "vllm"

  # Per-engine metric specifications.
  # Providing an entry for a name blocks the built-in default for that name.
  # Built-in names: "vllm", "sglang". All others are additive.
  engineConfigs:
    - name: vllm
      queuedRequestsSpec:  "vllm:num_requests_waiting"
      runningRequestsSpec: "vllm:num_requests_running"
      kvUsageSpec:         "vllm:kv_cache_usage_perc"
      loraSpec:            "vllm:lora_requests_info"   # "" to disable
      cacheInfoSpec:       "vllm:cache_config_info"    # "" to disable
    - name: sglang
      queuedRequestsSpec:  "sglang:num_queue_reqs"
      runningRequestsSpec: "sglang:num_running_reqs"
      kvUsageSpec:         "sglang:token_usage"
      loraSpec:            ""   # SGLang has no LoRA metric by default
      cacheInfoSpec:       ""
```

Spec strings use PromQL Instant Vector Selector syntax: `family_name{label=value}`.
Both quoted and unquoted label values are accepted.

### `metrics-data-source` parameters reference

```yaml
parameters:
  scheme: "http"    # or "https". Default: "http"
  path: "/metrics"  # Default: "/metrics"
  insecureSkipVerify: true  # Default: true
```

### Error handling

When a metric family is not found in the scraped data, the extractor appends a
"metric family not found" error to the poll result. The datalayer collector logs this error
**once** on first occurrence and once when it resolves — repeated identical errors are suppressed.
This prevents log flooding for persistent conditions such as a missing LoRA metric family on a
deployment that doesn't load LoRA adapters.

To eliminate the error entirely, disable the spec with an empty string as shown above.

## Parser Configuration

The `parser` section configures the parser to understand the request and response payloads. This is crucial for enabling advanced capabilities such as prefix-cache aware routing, request/response usage tracking, and other payload-specific processing. By default, if no parser is specified, the `openai-parser` is used, which supports the [OpenAI API](https://developers.openai.com/api/reference/overview).

Here is an example configuration that uses the `vllmgrpc-parser`:

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

## Feature Gates

The Feature Gates section allows for the enabling of experimental features of the IGW. These experimental
features are all disabled unless you explicitly enable them one by one.

The Feature Gates section has the following form:

```yaml
featureGates:
- dataLayer
- flowControl
```

The Feature Gates section is an array of flags, each of which enables one experimental feature.
The available values for these elements are:

- `dataLayer` which, if present, enables the experimental Datalayer APIs.
- `flowControl` which, if present, enables the [FlowControl](../flow-control.md) feature.

In all cases if the appropriate element isn't present, that experimental feature will be disabled.
