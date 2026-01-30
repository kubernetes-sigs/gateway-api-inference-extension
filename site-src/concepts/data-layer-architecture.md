# EPP Data Layer Architecture

The EPP Data Layer is a pluggable subsystem responsible for hydrating `Endpoint` objects with real-time signals (metrics, metadata) from external sources. It follows a highly decoupled architecture defined in `pkg/epp/framework/interface/datalayer`.

## Pattern: Driver-Based Extraction (DataSource Push)

Unlike a traditional "pull" model where consumers request data, the Data Layer uses a **Driver-Based Extraction** pattern:

1.  **The Driver (DataSource)**: An implementation of `fwkdl.DataSource` (e.g., `HTTPDataSource`) is the active component. Its `Collect` method is triggered by the framework on a schedule.
2.  **The Payload**: The `DataSource` fetches raw data (e.g., a Prometheus `/metrics` payload or a local status file).
3.  **The Push**: The `DataSource` then iterates through all registered `Extractor` plugins and "pushes" the raw data to them via their `Extract(ctx, data, ep)` method.
4.  **Wiring via Configuration**: Unlike scheduling plugins that are grouped in profiles, Extractors are explicitly associated with a DataSource in the `data` section of the configuration using `PluginRef`s. The configuration loader (`pkg/epp/config/loader/configloader.go`) resolves these references from the global `plugins` registry.
5.  **Type Validation**: During initialization, `AddExtractor` validates that the `DataSource` output type matches the `Extractor`'s `ExpectedInputType()`, ensuring runtime safety.

## Pattern: Driver-Based Push (Collector Loop)

The Data Layer follows a **Push Model** orchestrated by the `Collector` (`pkg/epp/datalayer/collector.go`):
1. The `Collector` runs a periodic loop for each endpoint.
2. On each tick, it calls `Collect()` on all registered `DataSource` instances.
3. The `DataSource` fetches data and "pushes" it to its registered `Extractors`.

This ensures that all extractors update their state (e.g., populating `Metrics.Custom`) in a synchronized fashion driven by the central collection cycle.

This architecture allows a single expensive network fetch (e.g., scraping vLLM metrics) to be shared by dozens of independent extraction plugins (Queue Depth, KV Cache, Custom Metrics) without redundant IO.

## The Metric Scoring Pipeline: Hose, Filter, Consumer

To configure scoring based on an arbitrary metric, it is helpful to visualize the three distinct roles in the pipeline:

1.  **The Hose (DataSource)**: Responsible for fetching the raw data blob (e.g., a Prometheus `/metrics` payload). In the configuration, this is a plugin like `prometheus-data-source` (or custom HTTP source). It doesn't know about specific metrics; it just knows how to connect to the model server and pull the full text.
2.  **The Filter (Extractor)**: A plugin like `prometheus-metric` that lives under the DataSource in the `data:` section. It parses the raw blob for **one specific metric** (e.g., `vllm:lora_requests_info`). It then populates the internal `Endpoint` object's `Metrics.Custom` map with that value.
3.  **The Consumer (Scorer)**: A scheduling plugin like `metric-scorer` (referenced in a `schedulingProfile`). It doesn't know about DataSources or Prometheus. It simply looks at the `Endpoint.Metrics.Custom` map for a key that matches its configured `metricName` and uses that value to produce a score (0.0 - 1.0).

**Key Wiring Rule**: For the pipeline to work, both the DataSource and the Extractor must be explicitly defined in the top-level `plugins:` section of the config, even though they are primarily used within the `data:` section.

## Pluggable Inventory

The Data Layer itself is fully pluggable. While the framework provides a robust `HTTPDataSource`, the architecture supports:
- **Custom DataSources**: Non-HTTP sources like local Unix sockets or shared memory.
- **Custom Extractors**: Plugins that parse arbitrary formats (JSON, Protobuf, custom text).
- **Metric Plugins**: Specialized Extractors that populate the `Custom` metric map. The **Prometheus Metric Plugin** is the primary implementation, allowing operators to extract any scalar gauge or counter from model servers without code changes.

## Validation and Consistency

To ensure reliable data flow, the Data Layer follows a strict initialization lifecycle:

1.  **Validation (Construction Time)**: The `ConfigLoader` (`pkg/epp/config/loader`) parses and validates the configuration structure.
    - **Plugin Existence**: Verifies that all referenced `pluginRef`s are defined in the global `plugins` section.
    - **Interface Compliance**: Verifies that Plugins implementing `fwkdl.DataSource` and `fwkdl.Extractor` are correctly typed.
    - **Runtime Safety**: During the subsequent initialization phase (in `Runner`), the framework enforces **Type Safety** by verifying that a generic `DataSource` producing type `T` is only connected to `Extractors` that accept type `T`.

2.  **Hydration (Scrape Cycle)**: Data is updated by the `Collector` on an independent, per-endpoint periodic loop.
    - **The Update**: When a scrape succeeds, the Extractor populates the `Metrics.Custom` map and sets a fresh `UpdateTime`.
    - **The Stale State**: If a scrape fails or is delayed beyond the `metricsStalenessThreshold`, the data remains in the map but becomes stale (indicated by an old `UpdateTime`).

3.  **Consumption (Scheduling Cycle)**: Data is "consumable" at any point, but is most critical during the **Scoring Phase** of a request.
    - **Freshness Check**: specialized plugins (like `UtilizationDetector`) can check the `UpdateTime` to decide if the metric is reliable.
    - **Fallback logic**: If a metric is missing (e.g., first-time initialization or persistent scrape failure), plugins like `MetricScorer` operate on a default/worst-case value to avoid scheduling blind.
