# Data Layer Architecture Proposal

Author(s): @elevran @nirrozenbaum

## Proposal Status

***Accepted***

## Summary

The EPP Architecture proposal identifies the need for an extensible
 [Data Layer](../0683-epp-architecture-proposal/README.md#data-layer).
 Recently, the scheduling subsystem underwent a major [architecture change](../0845-scheduler-architecture-proposal/README.md)
 to allow easier extension and pluggability. This proposal aims to apply
 similar extensibility to the Data Layer subsystem, allowing custom inference
 gateways to extend the Gateway API Inference extension (GIE) for their use
 cases without modifying the core GIE code base.

See [this document](https://docs.google.com/document/d/1eCCuyB_VW08ik_jqPC1__z6FzeWO_VOlPDUpN85g9Ww/edit?usp=sharing) for additional context amd reference.

## Goals

The Data Layer pluggability effort aims to address the following goals and
 requirements:

- Make endpoint attributes used by GIE components accessible via well defined
  Data Layer interfaces.
- Enable collection of additional (or different) subset of attributes from an
  existing data source (e.g., the `/metrics` endpoint scraper).
- Add a new data source that collects attributes not already collected.
- Follow best practices and experience from the Scheduling subsystem
  pluggability effort. For example, extending the system to support the above
  should be through implementing well defined Plugin interfaces and registering
  them in the GIE Data Layer subsystem; any configuration would be done in the
  same way (e.g., code and/or configuration file), etc.
- Be efficient (RAM, CPU, concurrency) in collecting and storing attributes.
- Limit change blast radius in GIE when making above changes. Core GIE code
  should not need to be modified in order to support collecting and storing new
  attributes. Affected code should be scoped only to modules that make use of
  the new attributes.
- The extensions should not increase coupling between GIE subsystems and
  Kubernetes (i.e., the environment specific code should be encapsulated and
  not “leaked” into the subsystem and its users).
- (Future) Allow non-uniform data collection (i.e., not all endpoints share the
  same data).

## Non-Goals

- Modify existing GIE abstractions, such as `InferencePool`, to conform to the
  Data Layer pluggability design. They are to remain first class concepts, as
  today.
- Enable reconciliation or modification of external state. The data sources are
  strictly read-only. For example, data source accessing Kubernetes state as part of
  the data collection would registered for `Watch()` notifications and shall not
  receive access to a k8s client.
- Inference scheduler Plugins, that rely on custom data collection, accept that
  the [Model Server Protocol](../003-model-server-protocol/README.md) no longer
  provides guarantees on portability of a model server out of the box.
- Intent is *not* to introduce a new scraping mechanism, and continue to support
  the current model of Go-routine per endpoint.

## Proposal

### Overview

There are two existing Data Sources in the Data Layer: a Pod reconciler that
 collects Pod IP address(es) and labels, copying them to endpoint attributes,
 and a metrics scraper that collects a defined set of metric values from the
 `/metrics` endpoint of each Pod. Note that the `InferencePool` reconciler is
 *not* considered part of the Data Layer.

### Components

The proposal is to make the Data Layer more extensible approaching by introducing
 these two interfaces:

- An **Attribute Collection** plugin interface responsible for extracting relevant
  attributes from a data source and storing them into the Data Layer for consumption
  by other components. The plugin can be registered with existing or new
  *Data Sources* (see below) and sources would call their registered plugins
  periodically or on change to process attributes.
- A **Data source** plugin interface that can be added to an inference gateway
  system, and on which *Attribute Collection* plugins can be registered to enrich
  the data model.

### Implementation Phases

In order to make iterative progress and validate the design alongside, we
 propose to implement and evolve the Data Layer extensibility over several
 phases:

1. Extend the backend, per endpoint, storage with a map from a name (i.e., the
  attribute collection interface) to the data it collected. Existing attributes,
  such as IP address or Pod labels, are not modified.
1. Introduce a Data Source registry where new data sources can be registered and
  bootstrap it by wrapping the existing `/metrics` with a Data Source API. At this
  point, the metrics scraping code includes only the `Data Source` interface and the
  `Data Collection` interface is not used/exposed.
1. Refactor the metrics scraping code into separate Data Source and Data Collection
  plugin interfaces.
1. Following that, and based on any lessons learnt, we’ll refactor the existing
  Kubernetes Pod reconciliation loop to the new plugin interfaces.

### Suggested Data Layer Plugin Interfaces

```go
// DataCollection interface consumes data updates from sources, stores
// it in the data layer for consumption.
// The plugin should not assume a deterministic invocation behavior beyond
// "the data layer believes the state should be updated"
type DataCollection interface {
    // Extract is called by data sources with (possibly) updated
    // data per endpoint. Extracted attributes are added to the 
    // Endpoint.
    Extract(ep Endpoint, data interface{}) error // or Collect?
}

// Endpoint interface allows setting and retrieving of attributes
// by a data collector.
// Note that actual endpoint structure would be something like (pseudocode)
// type EndpointState struct {
//   address
//   ...
//   data map[string]interface{}
// }
// The plugin interface would only mutate the `data` map
type Endpoint interface {
   // StoreAttributes sets the data for the Endpoint on behalf
   // of the named collection Plugin
   StoreAttributes(collector string, data interface{}) error
   
   // GetAttributes retrieves the attributes of the named collection
   // plugin for the Endpoint
   GetAttributes(collector string) (interface{}, error)
}

// DataLayerSourcesRegistry include the list of available 
// Data Sources (interface defined below) in the system.
// It is accompanied by functions (not shown) to register
// and retrieve sources
type DataLayerSourcesRegistry map[string]DataSource 

// DataSource interface represents a data source that tracks
// pods/resources and notifies data collection plugins to
// extract relevant attributes.
type DataSource interface {
    // Type of data available from this source
    Type() string

    // Start begins the data collection and notification loop
    Start(ctx context) error

    // Stop terminates data collection
    Stop() error

    // Subscribe a collector to receive updates for tracked endpoints
    Subscribe(collector DataCollection) error

    // UpdateEndpoints replaces the set of pods/resources tracked by
    // this source.
    // Alternative: add/remove individual endpoints?
    UpdateEndpoints(epIDs []string) error 
}
```

## Open Questions

1. Type safety in extensible data collection: `map[string]interface{}` seems
  like the simplest option to start, but may want to evolve to support
  type safety using generics or code generation.
1. Should we design a separate interface specifically for k8s object watching
  under GIE control or do we want these to be managed as yet another data source?
  This affects the design (e.g., who owns the k8s caches, clients, etc.).
  With a GIE controlled data source, collectors just register the types (and
  other constraints? Labels, namespaces, …) with GIE core, and all k8s
  functionality is under GIE control.
