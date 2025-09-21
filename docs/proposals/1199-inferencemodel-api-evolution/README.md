# InferenceModel API Evolution

Author(s): @kfswain, @ahg-g, @lukeavandrie
## Proposal Status
 ***Implemented***
 
 Note
 - Phase 1 is complete
 - Phase 2 is still WIP

## Summary
Multiple docs have discussed the restructuring of the InferenceModel API. This [doc](https://docs.google.com/document/d/1x6aI9pbTF5oOsaEQYc9n4pBBY3_AuEY2X51VKxmBSnU/edit?tab=t.0#heading=h.towq7jyczzgo) proposes an InferenceSchedulingObjective CRD, and this [doc](https://docs.google.com/document/d/1G-CQ17CM4j1vNE3T6u9uP2q-m6jK14ANPCwTfJ2qLS4/edit?tab=t.0) builds upon the previous document to solidify the requirement for the new iteration of the InferenceModel API to continue to solve the identity problem. Both these documents were useful in continuing to gather feedback & iterate on a proper solution. 

This proposal is intended to act as the plan of record for solution that will be implemented.

## Implementation Phases

### Phase 1 - Rename, Split, & Modify InferenceModel 
A few points were used in composing justification & structure of this change:
 - the Criticality field of InferenceModel is in use, & provides functionality 
 - InferenceModel is an Alpha API
 - InferenceModel is not depended upon by upstream or downstream components

 Phase 1 will retain the Criticality functionality, but will rename the InferenceModel API and slim down the spec. Additionally, this slimmed down spec will be able to be applied at a _per request_ level. Justification in [Phase 1](#phase-1).

### Phase 2 - Introduce new Usage Tracking, Fairness, & SLO CRDs
Phase 2 will happen over a longer period of time & slowly introduce new CRDs to Inference Gateway, much of what is discussed in this proposal is keeping Phase 2 in mind, but phase 2 can be considered experimental & subject to change.

Primarily phase 2 will introduce and develop these these CRDs:
- Usage tracking (used in fairness)
- Fairness configuration
- SLO configuration

## Design Principles

### Goals
- Reliable and predictable fairness allocation
- Disconnect identity from policy-like objects where possible
- Anonymous identity/defaults are graceful (fault-tolerant) & unsurprising
- Scalable, simple, and reusable config
- Retain the functionality of InferenceModel
  - Traffic splitting models & modelName rewrite
  - Criticality

### Non-Goals
- Addressing security concerns with the API, this is currently expected to either be:
  - Entirely contained within a trusted system
  - Or auth handled upstream
- IGW implementing a custom auth mechanism


## Definitions

- **Tenant** Kuberenetes chooses the term ***tenant*** as described [here](https://kubernetes.io/docs/concepts/security/multi-tenancy/#tenants). Fairness APIs _may_ be used in multitenant scenarios, so as an example, multi-tenancy may be used.

# Proposal

Discussion of the problem(s) can be seen in the linked documents. Here we will describe the new API surface.

## Phase 1

### Structure change
This API solves 3 general pillars of problem, that can also be categorized into 2 areas:

Higher-order Request Gropuing (Usage tracking):
 - This API describes Resource Sharing (Criticality/Fairness)
 - This API describes Identification (used in Fairness)

Request specific objectives:
 - This API describes Specific Request Policy (SLO/Criticality)


As such, the InferenceModel API will be split into separate CRDs to reflect the difference in these scopes. Phase 1 will focus on the **Request specific objectives**. Specifically it will maintain the inclusion of criticality. Other phase 1 changes:

- The EPP will expose a flag to define the header key (default: `x-gateway-inference-objectives`) that will be used to assign InferenceObjectives to requests
- The EPP will expose a flag to define the header key (default: `x-gateway-inference-fairness-id`) that will be used in tracking Request Usage (which will act as the identifier for simple fairness implementation)
- The EPP will expose a flag to define the header key (default: `x-gateway-model-name-rewrite`) that will be used when the provided model name is desired to be overwritten
- The modelName rewrite functionality will be included into EPP as a core feature (also handled by header) **NOTE**: _Relying on this feature for writing a proper model name disables the ability to use the fail-open feature_
- Continue to support traffic splitting across models, although not necessarily via GIE CRDs directly (e.g., delegated to GW API/HTTPRoute) - example [here](https://docs.google.com/document/d/1s4U4T_cjQkk4UeIDyAJl2Ox6FZoBigXBXn9Ai0qV7As/edit?tab=t.0#heading=h.bkttj79mzxlz)

### Naming
The current name for the CRD that will house **Request specific objectives** is planned to be `InferenceObjectives`


### CRD spec

This CRD definition is a slimmed version of InferenceModel with a name change. Example here:

```golang
type InferenceObjectives struct {
  metav1.TypeMeta
  metav1.ObjectMeta

  Spec InferenceObjectivesSpec
}

type InferenceObjectivesSpec struct {
  PoolRef InferenceObjectReference

  // this is a departure from InferenceModel that used string for criticality.
  // We got quite a bit of feedback around allowing for custom criticality bands, so an int/enum is more flexible & carries inherent stack rank value.
  Criticality *int
}

```

## Phase 2 - SUBJECT TO CHANGE
 
***NOTE: `InferenceUsageMeter` Name is a placeholder***

### CRD spec
```golang

type InferenceUsageMeter struct {
  metav1.TypeMeta
  metav1.ObjectMeta

  Spec InferenceUsageMeterSpec
}

type InferenceUsageMeterSpec struct {
  // optional field that defaults to kube object name if not included
  ID *string
  PoolRef InferenceObjectReference

  // one of; This allows for embedded configuration or reference to a commonly used config.
  UsageLimits *NotYetDefinedFairnessCRD
  UsageLimitsRef *InferenceObjectReference
}

type InferenceObjectives struct {
  metav1.TypeMeta
  metav1.ObjectMeta

  Spec InferenceObjectivesSpec
}

type InferenceObjectivesSpec struct {
  PoolRef InferenceObjectReference

  // this is a departure from InferenceModel that used string for criticality.
  // We got quite a bit of feedback around allowing for custom criticality bands, so an int/enum is more flexible & carries inherent stack rank value.
  Criticality *int
  PerformanceObjectives NotYetDefinedSLOCRD
  PerformanceObjectivesRef *InferenceObjectReference
  // Doc on SLO CRD here: https://docs.google.com/document/d/1j2KRAT68_FYxq1iVzG0xVL-DHQhGVUZBqiM22Hd_0hc/edit?resourcekey=0-5cSovS8QcRQNYXj0_kRMiw&tab=t.0#heading=h.emkaixupvf39
}
```

### Intent

The purpose(s) of the `InferenceUsageMeter` is:
- Create a strong concept of usage tracking within the inference pool; used to associate groups of requests together for the purpose of Flow Clontrol (Fairness) - which can enforce:
  - Fair resource sharing
  - Inter-tenant prioritization
  - SLO attainment
- Detach identification from the modelName field

## Design points
Included is some discussion around specific choices made in the API design

### Identification
**Note**: The ID field would default to the kube name.

The only field associated with identification is the `ID` field. An optional ID field was chosen (rather than strictly using the metadata name), because:
- A user may not want to put the same restrictions on the id that is enfored on a kube resource name
- The ID name may be duplicated across different pools
  - This could also be solved by allowing the UsageMeter & Objectives to reference multiple pools
- Use of a kube-generated name would force an upstream Auth mechanism to be aware of the `InferenceObjectives` API

***Discussion point***: In order to support a high volume of tenants, we could allow IGW to accept unique IDs that do not have an explicit InferenceUsageMeter object defined. Instead using a default fairness configuration. **Feedback here requested.**

#### Alternative consideration(s)
- Expanding the PoolRef field to be plural was considered, however that was not selected to maintain simplicity. It is a decision that can be revisited in the future, however.

