# Inference Scheduling Objective

Author(s): @ahg-g, @kfswain

## Proposal Status
 ***Draft***

[model]:(https://platform.openai.com/docs/api-reference/chat/create#chat-create-model)

## Summary

The [InferenceModel](../002-api-proposal/README.md#inferencemodel) has been found to have a few key issues:

- The naming is misleading & not indicative of its purpose
- The [model] param is the only matching rule
- The InferenceModel is a mandatory inclusion to serve requests to the pool

Due to this, we propose a restructuring of the `InferenceModel` to the `InferenceServingObjective`(ISO), to help solve some of the current pain points.

Original discussion doc: https://docs.google.com/document/d/1x6aI9pbTF5oOsaEQYc9n4pBBY3_AuEY2X51VKxmBSnU/edit?tab=t.0#heading=h.geklwdrtzbph


## Goals

- Update name to something more descriptive & agreed-upon
- Broaden the match from just [model] to a more ubiquitous system

## Non-Goals

- Create a generic, multi-applicable 'policy' field
  - Future iterations may transform this field to be more policy-like, but currently ISO is still the identifier for fairness calculation, and so acts as the primary key for fairness budget allocation

## Proposal

`InferenceSchedulingObjective` is focused on defining scheduling objectives for a matching request flow, concretely these changes will: 

- Drop the `InferenceModel` API, but no changes to the InferencePool API
- Replace the API with `InferenceSchedulingObjective`. The API will define `scheduling objectives` for matching requests. The inference-scheduler (run by the EPP) will be the primary actuator of this API.
- We will keep the Criticality field, with the intent to add more serving objectives (e.g., latency SLOs)
- Request matching beyond model name to include headers. This allows defining different scheduling policies for different request flows (apps or users) while targeting the same model. The semantics should be adopted from HTTPRouteMatch as defined in HTTPRouteRules.
- A default InferenceSchedulingObjective per InferencePool will be included as a fallback policy when no one matches the request. These defaults can be adjusted.
- Traffic splitting is not part of the InferenceSchedulingObjective API. Traffic splitting is not an endpoint scheduling objective, it is a request routing objective. As we describe below, with some creativity, we can offload traffic splitting to  HTTPRoute.
  - An intended side effect of this is that users will more easily be able to define different scheduling policies for the same target models, something that required some shenanigans with the current API (2 inferenceModels with distinct `modelNames` both pointing to the same target model).

```golang
type InferenceSchedulingObjectivesSpec struct {
  // Match defines what requests this objective applies to.
  Match

  // Criticality defines how important it is to serve the requests that match this objective
  // compared to requests that match other objectives.
  // Criticality impacts how traffic is handled in resource constrained situations. It handles this 
  // by queuing or rejecting requests of lower criticality. Objectives of an equivalent Criticality 
  // will fairly share resources over throughput of tokens. In the future, the metric used to 
  // calculate fairness, and the proportionality of fairness will be configurable.
  //
  //
  // Default values for this field will not be set, to allow for future additions of a new field that 
  // may 'one of' with this field.
  // Any implementations that may consume this field may treat an unset value as the 
  // 'Standard' range.
  Criticality *Criticality 

  // Future scheduling objectives, like SLOs.

  // PoolRef is a reference to the inference pool, the pool must exist in the same namespace.
  PoolRef PoolObjectReference
}
type Match struct {
  // Only one of the following can be set.

  // HTTPMatches is a list of http requests matchers, the list of matchers are ORed.
  HTTPMatches []HTTPMatch
  // GRPCMatches is a list of gRPC requests matchers, the list of matchers are ORed.
  GRPCMatches []GRPCMatch
}

// HTTPMatch is an http matching rule. The rules are ANDed.
type HTTPMatch struct {
  // ModelName matches against the model name in the body as per OpenAI protocol
  ModelName *string
  // Headers specifies HTTP request header matchers.
  Headers []HTTPHeaderMatch  // mostly as defined in the gateway api
  // version of it that only supports exact header matching.
}

// GRPCMatch is a gRPC matching rule. The rules are ANDed.
type GRPCMatch struct {
  // ModelName matches against the model name in the body as per OpenAI protocol.
  ModelName *string
  // Headers specifies gRPC request header matchers.
  Headers []GRPCHeaderMatch  // mostly as defined in the gateway api, likely a more limited 
  // version of it that only supports exact header matching.
}
```

## API: Before and After

### Default Policy

#### Before

Not possible today, but could be done if we define a catch all modelName expression:

```yaml
kind: InferenceModel
metadata:
  name: default
spec:
  modelName: *
  criticality: Standard
  poolRef:
    name: gemma-pool
```

#### After

```yaml
kind: InferenceSchedulingObjective
metadata:
  name: default
spec:
  criticality: Standard
  poolRef:
    name: gemma-pool
```

### Separate scheduling objectives for the same target model

#### Before
Possible, requires multiple entries

```yaml
kind: InferenceModel
metadata:
  name: llama4
spec:
  modelName: llama4-prod
  targetModels:
    - name: llama4
  criticality: Critical
  poolRef:
    name: gemma-pool

kind: InferenceModel
metadata:
  name: llama4
spec:
  modelName: llama4-dev
  targetModels:
    - name: llama4
  criticality: Sheddable
  poolRef:
    name: gemma-pool
```

#### After
Possible, requires multiple entries

```yaml
kind: InferenceSchedulingObjective
metadata:
  name: critical-llama4
spec:
  httpMatches:
  - modelName: llama4
    headers:
    - name: “app”
      value: “prod”
  criticality: Critical
  poolRef:
    name: llama4-pool

---

kind: InferenceSchedulingObjective
metadata:
  name: sheddable-llama4
spec:
  httpMatches: 
  - modelName: llama4
    headers:
    - name: “app”
      value: “dev”
  criticality: Sheddable 
  poolRef:
    name: llama4-pool
```

### Traffic Splitting

#### Before

EPP handling model rewrite & splitting/weighting

```yaml
kind: InferenceModel
metadata:
  name: llama4
spec:
  modelName: llama4-prod
  targetModels:
    - name: llama4
      weight: 10
    - name: llama42
      weight: 50
  criticality: Critical
  poolRef:
    name: gemma-pool
```

#### After

Offload to httpRoute, EPP is now extended to override model name on: `X-Gateway-Model-Name`, added benefit of splitting on pools at the same place.

```yaml
kind: HTTPRoute
apiVersion: gateway.networking.k8s.io/v1
metadata:
  name: my-route
spec:
  parentRefs:
    - name: my-inference-gateway
  rules:
  - matches:
    - headers:
      - type: Exact
        name: X-Gateway-Model-Name
        value: food-review
    backendRefs:
    - name: vllm-llama3-8b-instruct
      kind: InferencePool
      group: inference.networking.x-k8s.io
      weight: 90
      - filters:
        - type: RequestHeaderModifier
          requestHeaderModifier:
            set:
            - name: X-Gateway-Model-Name
              value: food-review-v1
    - name: vllm-llama3-8b-instruct
      kind: InferencePool
      group: inference.networking.x-k8s.io
      weight: 10
      - filters:
        - type: RequestHeaderModifier
          requestHeaderModifier:
            set:
            - name: X-Gateway-Model-Name
              value: food-review-v2
```

### Open Questions

- How might `Match` conflict/converge with HTTPRoute?
- Is it easier to make changes piecewise? (We are currently: renaming, adjusting how matching works, & offloading traffic splitting to HTTPRoute)
- Should we split the `Match` into its own CRD (named something like `InferenceWorkload`) that can be used for fairness budget tracking/workload affiliation, and then translate `ISO` to a more objective policy-like object that the `Match` CRD subscribes to, reducing duplicate config