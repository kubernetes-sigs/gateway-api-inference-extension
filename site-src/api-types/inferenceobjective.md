# Inference Objective

??? example "Alpha since v1.0.0"

    The `InferenceObjective` resource is alpha and may have breaking changes in
    future releases of the API.

## Background

The **InferenceObjective** API defines a set of serving objectives of the specific request it is associated with. This CRD currently houses only `Priority` but will be expanded to include fields such as SLO attainment.

## Usage

To associate a request to the InferencePool with a specific InferenceObjective, the system uses a specific header: `x-gateway-inference-objective` with the value of the header set to the InferenceObjective metadata name. So the calling client must set the header key/value on the request to associate the selected InferenceObjective. If no InferenceObjective is selected, default values are used.  

## Spec

The full spec of the InferenceObjective is defined [here](/reference/x-v1a2-spec/#inferenceobjective).
