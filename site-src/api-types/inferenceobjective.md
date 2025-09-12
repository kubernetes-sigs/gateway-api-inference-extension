# Inference Objective

??? example "Alpha since v1.0.0"

    The `InferenceObjective` resource is alpha and may have breaking changes in
    future releases of the API.

## Background

The **InferenceObjective** API defines a set of serving objectives of the specific request it is associated with. This CRD currently houses only `Priority` but will be expanded to include fields such as SLO attainment.

## Spec

The full spec of the InferenceModel is defined [here](/reference/x-spec/#inferenceobjective).