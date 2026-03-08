# Upstreaming Prefill/Decode Disaggregation Support

Author(s): @yangligt2

## Proposal Status
 ***Proposed***

## Summary
This proposal outlines the strategy to upstream Prefill/Decode (P/D) disaggregated serving logic from the `llm-d-inference-scheduler` project into the core Gateway API Inference Extension (GAIE) framework. By providing standardized scheduling handlers, request injection plugins, and standard header conventions, GAIE will support interoperable P/D scaling across various model servers out-of-the-box.

## Motivation
Disaggregated serving is a critical optimization for LLM inference. By decoupling the compute-heavy prefill phase from the memory-heavy decode phase, we can scale resources independently, drastically improving hardware utilization and reducing time-to-first-token.

Currently, this orchestration logic is embedded in the `llm-d` repository framework. Upstreaming the core protocol to GAIE allows any model server (e.g., vLLM, sglang, TGI) deployed behind the Gateway to leverage disaggregated serving seamlessly without reinventing orchestration logic.

## Goals
- Define a standard protocol for the EPP and Gateway to communicate P/D targeting to backend components.
- Establish standard HTTP headers for conveying prefill server metadata.
- Introduce a specialized ProfileHandler for P/D scheduling logic in the GAIE framework.
- Introduce a specialized PreRequest plugin to inject selected prefill endpoints into request headers.

## Non-Goals
- Modifying the underlying data plane routing implementation (Gateway vs. simple ExtProc vs. Istio directly) for handling P/D.
- Standardizing the KV cache transfer protocol *between* model servers (this is left to the specific inference engines; GAIE merely provides the discovery/routing metadata).
- Altering the core general-purpose SchedulingResult struct to include P/D-specific strongly-typed fields.

## Proposal

The proposed approach standardizes how the EPP determines the disjoint servers required for a P/D session and how the Gateway passes that information to the backend.

### P/D Request Flow
1. **Endpoint Selection**: The EPP executes two distinct scheduling profiles: a "decode" profile (the primary target) and a "prefill" profile (the orchestration helper).
2. **Metadata Injection**: The EPP injects the address of the selected prefill server(s) into the request via a standard header (`x-gateway-prefill-endpoints`).
3. **Primary Routing**: The Gateway routes the request to the *Decode* server (the primary destination).
4. **Backend Coordination**: The Decode server reads the injected header and initiates a direct KV cache pull from the specified Prefill server(s).

### Header Standardization
To remain consistent with GAIE's existing header schema, we propose a new standard header:
*   **Name**: `x-gateway-prefill-endpoints`
*   **Value**: A comma-separated list of authority tuples (`<ip>:<port>`). Example: `10.1.2.3:8000,10.1.2.4:8000`.

*Note: While many implementations will only select a single prefill server per request, a comma-separated list provides future-proofing for multi-prefill or tensor-parallel discovery.*

### Plugin Architecture
To align with the generalization principles of GAIE, this logic is implemented entirely via the standard Plugin mechanism without altering the core framework structs.

#### 1. Scheduling: `PrefillDecodeProfileHandler`
A new ProfileHandler implementation (`pkg/plugins/profile/prefill_decode.go`) that acts as the orchestration point for P/D logic.
- Executes the configured `decodeProfile` first.
- Calculates dynamic hit rates (e.g., utilizing prefix-cache state). If the request is short enough, it may optimize by skipping the prefill profile entirely (`DecisionTypeDecodeOnly`).
- If prefill is required, executes the `prefillProfile`.
- Returns both results in the `SchedulingResult.ProfileResults` map, explicitly setting the `PrimaryProfileName` to the Decode profile.

#### 2. Ergonomics: `ProfileResults` Helper Methods
To address concerns about magic strings while avoiding overly specific fields in the core result structs, we will provide typed helper methods on the existing results map:
```go
type ProfileResults map[string]*ProfileRunResult

func (p ProfileResults) DecodeResult(profileName string) *ProfileRunResult {
    return p[profileName]
}

func (p ProfileResults) PrefillResult(profileName string) *ProfileRunResult {
    return p[profileName]
}
```

#### 3. Request Control: `PrefillInjectionPlugin`
A new PreRequest plugin (`pkg/plugins/requestcontrol/prefill_injection.go`) responsible for finalizing the contract before the Gateway forwards the request.
- Looks up the configured prefill profile's result using the helper methods.
- If a prefill endpoint is selected, extracts the IP and Port.
- Sets the `x-gateway-prefill-endpoints` header in the LLMRequest.

## Configuration API
Consumers will configure the P/D plugins cleanly within their `plugins.json` or equivalent configuration payload:

```json
{
  "profileHandler": {
    "name": "pd-profile-handler",
    "parameters": {
      "decodeProfile": "standard-decode",
      "prefillProfile": "standard-prefill",
      "threshold": 100
    }
  },
  "preRequest": [
    {
      "name": "prefill-header-handler",
      "parameters": {
        "targetProfile": "standard-prefill"
      }
    }
  ]
}
```

## Migration & Backward Compatibility
Current `llm-d-inference-scheduler` users rely on an older header (`x-prefiller-host-port`).
To ensure a smooth transition:
1.  The newly upstreamed `PrefillInjectionPlugin` should support an optional configuration parameter `headerNameOverride` to allow users to temporarily retain older headers.
2.  Once validated, the native model servers within the `llm-d` stack will be updated to consume the standard `x-gateway-prefill-endpoints` header, removing custom logic completely.