# Request Handling Generalization Refactor

## Objective
The primary objective of this refactoring was to generalize the request handling mechanisms within the Endpoint Processing Plane (EPP). Previously, the EPP framework and its plugins were tightly coupled to an OpenAI-specific `LLMRequest` and `LLMRequestBody`. By abstracting these structures into a more generic `InferenceRequest` wrapper and a dedicated `requesthandling` package, the EPP is now better positioned to support arbitrary generative AI models and custom parsing logic.

## Design Changes & Implementation Details

### 1. Introduction of `InferenceRequest`
We introduced a new top-level request wrapper struct, `InferenceRequest`, within `pkg/epp/framework/interface/scheduling/types.go`:

```go
type InferenceRequest struct {
	LLM        *LLMRequest
	ParsedBody any `json:"-"`
}
```

**Why:**
`LLMRequest` previously served as the single source of truth for all request data, but its fields were intrinsically tied to OpenAI API semantics. `InferenceRequest` encapsulates the legacy `LLMRequest` while introducing an opaque `ParsedBody` field (`any`). This allows custom parsers (e.g., gRPC parsers, vLLM parsers, or other model-specific parsers) to attach generalized or internal domain representations of a request without breaking the strict `LLMRequest` contract. 

### 2. Creation of the `requesthandling` Package
We created a new package at `pkg/epp/framework/interface/requesthandling` to house the payload structures. 

*   Moved all request body types (`CompletionsRequest`, `ChatCompletionsRequest`, `ResponsesRequest`, `ConversationsRequest`, `Message`, `Content`, `ContentBlock`, etc.) from the `scheduling` package to the `requesthandling` package.
*   Renamed `LLMRequestBody` to a more generic `RequestBody`.

**Why:**
The `scheduling` package should focus strictly on concepts related to pod selection, profiling, filtering, and scoring. It was previously cluttered with API payload definitions (like `AudioBlock` or `ChatCompletionsRequest`). Moving these to a dedicated `requesthandling` package enforces a stronger separation of concerns.

### 3. Broadened Framework Interfaces
All framework interfaces—including the `Director`, Admission Controller, Scorer Plugins, Filter Plugins, and Request Control Plugins—were updated to accept `*scheduling.InferenceRequest` instead of `*scheduling.LLMRequest`.

For example, the Scorer Interface changed from:
```go
Score(context.Context, *CycleState, *LLMRequest, []Endpoint) map[Endpoint]float64
```
To:
```go
Score(context.Context, *CycleState, *InferenceRequest, []Endpoint) map[Endpoint]float64
```

**Why:**
Plugins require access to the entire request context. By passing the generic `InferenceRequest` down the pipeline, plugins can seamlessly transition to utilizing the model-agnostic `ParsedBody` when evaluating admission policies or scoring pods. This lays the groundwork for plugins that look beyond OpenAI semantics (e.g., scoring based on custom fields extracted by a proprietary gRPC parser).

### 4. Codebase Plumbing & Test Refactoring
*   Updated field references across the director and plugin implementations (e.g., changing `req.RequestId` and `req.TargetModel` to `req.LLM.RequestId` and `req.LLM.TargetModel`).
*   Reconciled test suites to initialize nested `LLM` struct literals inside `InferenceRequest`.

## Summary of Impact
This architectural change significantly reduces technical debt associated with API lock-in. The EPP pipeline is now decoupled from the OpenAI schema, empowering future development to focus on diverse AI endpoint compatibility (e.g., vLLM transcoders) without requiring sweeping interface changes across the codebase.
