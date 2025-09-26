# Inference Pool Import

??? example "Alpha since v1.1.0"

    The `InferencePoolImport` resource is alpha and may have breaking changes in
    future releases of the API.

## Background

The **InferencePoolImport** API is a cluster-local, controller-managed resource that represents an imported InferencePool.
It primarily communicates a relationship between an exported InferencePool and the exporting cluster name. It is not
user-authored; status carries the effective import. Inference Platform Owners can reference the InferencePoolImport,
even if the local cluster does not have an InferencePool. In the context of Gateway API, it means that an HTTPRoute can
be configured to reference an InferencePoolImport to route matching requests to endpoints of backing InferencePools.

Key ideas:

- Map an exported InferencePool to exporting controller and cluster.
- Name/namespace sameness with the exported InferencePool (avoids extra indirection).
- Conditions: Surface a controller-level status condition to indicate whether the InferencePoolImport is ready for use.
- Conditions: Surface parent-level status conditions to indicate whether the InferencePoolImport is referenced by a parent,
  e.g. Gateway.

## Spec

The full spec of the InferencePoolImport is defined [here](/reference/x-v1a1-spec/#inferencepoolimport).
