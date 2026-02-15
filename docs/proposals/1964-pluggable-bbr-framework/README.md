# Pluggable Body-Based Routing (BBR) Framework

Author(s): @davidbreitgand @srampal

## Proposal Status

***Draft***

## Summary

The Gateway API Inference Extension (v1.2.1) includes an initial implementation of Body-Based Routing (BBR). Currently, BBR provides a single capability: it extracts the model name from the request body and adds it to the `X-Gateway-Model-Name` header. This header is then used to route the request to the appropriate InferencePool and its associated Endpoint Picker Extension (EPP) instances.

The current BBR implementation is limited and lacks extensibility. Similar to the [pluggability introduced in the scheduling subsystem](../0845-scheduler-architecture-proposal/README.md), BBR should support custom extensions without requiring modifications to the GIE code base.

This proposal introduces a plugin architecture for BBR that allows developers to implement custom logic. Plugins could be organized into a chain or DAG for ordered and concurrent execution.

See [this document](https://docs.google.com/document/d/1So9uRjZrLUHf7Rjv13xy_ip3_5HSI1cn1stS3EsXLWg/edit?tab=t.0#heading=h.55jwocr94axs) for additional context amd reference.

## Goals

The pluggable BBR Framework aims at addressing the following goals

### Immediate Goals

- Avoid monolithic architecture
- Mimic pluggability and configurability of the scheduling subsystem without coupling between the two
- Limit changes to the BBR feature to avoid any changes in the rest of the code base
- Follow best practices and experience from the Scheduling subsystem
  pluggability effort. For example, extending the system to support the above
  should be through implementing well defined `Plugin` interfaces and registering
  them in the BBR subsystem; any configuration would be done in the
  same way (e.g., code and/or configuration file)
- Reuse common code from EPP, such as `TypedName`, wherever make sense, but avoid reusing specialized code with non-BBR functionality to avoid abuse
- Provide reference plugin implementation(s).

### Extended Goals

- Enable organizing plugins into a topology for sequential and concurrent execution. Note that while BBR stands for Body-Based Routing and this proposal does not aim at general Payload Processing, routing decisions might require pre-processing/postprocessing operations
- Avoid redundant recurrent body parsing across plugins in a topology for the sake of performance
- Enable extensible collection and registration of metrics using lessons from the pluggable scheduling sub-system

## Non-Goals

- Modify existing GIE abstractions
- Fully align plugins, registries, and factories across BBR and EPP
- Dynamically reconfigure plugins and plugin topologies at runtime
- Enable extensibility of the BBRPlugin registration mechanisms in third party extensions

## Proposal

### Overview

There is an embedded `BBRPlugin` interface building on the `Plugin` interface adopted from EPP. This interface should be implemented by any BBR plugin. Each plugin is identified by its `TypedName` (adopted from EPP), where `TypedName().Type` gives the string representing the type of the plugin and `TypedName().Name()` returns the string representing the plugins implementation. BBR is refactored to implement the registered factory pattern.

In addition, as an extended functionality, a `PluginsChain` interface is defined to define an order of plugin executions. In the future, `PluginsChain` might be replaced by `PluginsDAG` to allow for more complex topological order and concurrency.

`PluginsChain` only contains ordered `BBRPlugin` types registered in the `PluginRegistry`. `RequestPluginsChain` and `ResponsePluginsChain` are optionally configured for handling requests and responses respectively. If no configuration is provided, default `PluginsChain` instances will be configured automatically.

Depending on a `BBRPlugin` functionality and implementation, the plugin might require full or selective body parsing. To save the parsing overhead, if there is at least one `BBRPlugin` in the `PluginsChain` that requires full body parsing, the parsing is performed only once into a shared official appropriate `openai-go` struct (either `openai.CompletionNewParams`  or `openai.ChatCompletionNewParams` depending on the request endpoint). This struct is shared for read-only to all plugins in the chain. Each `BBRplugin` receives the shared struct by value. If a plugin needs to mutate the body, in the initial implementation, it MUST work on its own copy, and the a mutated body is returned separately by each plugin.

Even simple BBR plugin implementations can considerably differ in their performance w.r.t. to latency and memory. This justifies different implementations of BBR Plugins in different contexts.

![Benchmarking different implementation of OpenAI message body parsing to extract `model` metadata](./images/benchmark-summary.png)

[The benchmark details and code can be found here](https://github.com/davidbreitgand/scripts/tree/main/benchmarks).

### Suggested Components

The sketch of the proposed framework is shown in the figure below.
![Pluggable BBR framework architecture showing components including BBRPlugin interface, PluginRegistry, PluginFactory, PluginsChain orchestrator, and data flow between request handler, shared parsed body struct, plugin execution chain, and response handler with headers and mutated body outputs](./images/pluggable-framework-architecture-sketch.png)

### Suggested BBR Pluggable Framework Interfaces

```go
// ------------------------------------ Defaults ------------------------------------------

const DefaultPluginType = "MetadataExtractor"
const DefaultPluginImplementation = "simple-model-selector"

// BBRPlugin defines the interface for plugins in the BBR framework 
type BBRPlugin interface {
    plugins.Plugin

    // Execute runs the plugin logic on the request body.
    // A plugin's implementation logic CAN mutate the body of the message.
    // A plugin's implementation MUST return a map of headers
    // If no headers are set by the implementation, the map must be empty
    // A value of a header in an extended implementation NEED NOT to be identical to the value of that same header as would be set
    // in a default implementation.
    // Example: in the body of a request model is set to "semantic-model-selector",
    // which, say, stands for "select a best model for this request at minimal cost"
    // A plugin implementation of "semantic-model-selector" sets X-Gateway-Model-Name to any valid
    // model name from the inventory of the backend models and also mutates the body accordingly

    Execute(requestBodyBytes []byte) (headers map[string]string, mutatedBodyBytes []byte, err error)
}


// NeedsFullParsing is an optional capability interface.
// Plugins that require full body parsing implement this marker method.
// The method has no return value; presence of the method is the signal.
type NeedsFullParsing interface {
    FullParsingNeeded(){}
}

// placeholder for BBRPlugin constructors
// Concrete constructors are assigned to this type

type PluginFactoryFunc func() (bbrplugins.BBRPlugin, error)

### Defaults

A default plugin instance that sets `X-Gateway-Model-Name` header will always be configured automatically if a specific plugin is not configured. The default plugin will only set the header without body mutation.

### Current BBR reimplementation as BBRPlugin

Will be done according to this proposal and phased approach detailed in the next section.

### Implementation Phases

The pluggable framework will be implemented iteratively over several phases and a series of small PRs.

1. Introduce `BBRPlugin` `MetadataExtractor`, interface, registry, default plugin implementation (`simple-model-selector`) and its factory. Plugin configuration will be implemented via environment variables set in helm chart
1. Introduce plugins topogy (initially a `PluginsChain`)
1. Introduce shared struct (shared among the plugins of a plugins chain) to 
1. Introduce an interface for guardrail plugin, introduce simple reference implementation, experiment with plugins chains on request and response messages
1. Refactor metrics as needed to work with the new pluggable framework
1. Implement configuration via manifests similar to those in EPP
1. Implement `PluginsDAG` to allow for more complex topological order and concurrency.
1. Continously learn lessons from this implementation and scheduling framework to improve the implementation
1. Aim at aligning and cross-polination with the [AI GW WG]("https://github.com/kubernetes-sigs/wg-ai-gateway").

## Open Questions

1. More elaborate topology definition and execution
1. More elaborate shared memory architecture for the best performance
1. Considerations for handling newer OpenAI API 
1. OpenAI API continues to evolve and most recently they added the "responses api" which has some stateful logic in addition to the ChatCompletions endpoint. The design will be extended also to cover the OpenAI Responses API. For example the `PluginsChain` might be extended to provide common utilities to either help with state caching or letting plugins handle that completely.
1. TBA

## Note 1

The proposed interfaces can slightly change from those implemented in the [initial PR 1981]("https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/1981").
The initial PR will be refactored into a series of small PRs which should be evaluated in reference to this proposal.
