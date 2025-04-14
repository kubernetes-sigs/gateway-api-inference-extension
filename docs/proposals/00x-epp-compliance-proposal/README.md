# Gateway API Inference Extension

## Proposal Status
 ***Draft***

## Table of Contents

<!-- toc -->

-   [Summary](#summary)
-   [Goals](#goals)
-   [Non-Goals](#non-goals)
-   [Proposal](#proposal)
    -   [Personas](#personas)
        -   [Inference Platform Admin](#inference-platform-admin)
        -   [Inference Workload Owner](#workload-owner)
    -   [Axioms](#axioms)
    -   [InferencePool](#inferencepool)
    -   [InferenceModel](#inferencemodel)
    -   [Spec](#spec)
    -   [Diagrams](#diagrams)
    -   [Alternatives](#alternatives) 
- [Open Questions](#open-questions)
    
<!-- /toc -->

## Summary

This proposal seeks to standardize the implementation of an EPP (End-point Picker) for the Inference Gateway extension (also known as Gateway API Inference Extension). Additionally, this proposes to restructure the current implementation of the EPP to be more modularize, and approachable.

## Goals

- Set a standard on how the EPP & APIs interact
- Settle on common nomenclature for clearer communication
- Allow for modularization of the EPP, to be extended to a users specific needs

## Non-Goals

- Reshaping the current API
- A change in scope of the current project

## Proposal

This 