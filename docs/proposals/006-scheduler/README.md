# Gateway API Scheduler

Authors: @kfswain, @smarterclayton

## Proposal Status
 ***Draft***

## Table of Contents

<!-- toc -->

-   [Summary](#summary)
-   [Goals](#goals)
-   [Non-Goals](#non-goals)
-   [Proposal](#proposal)
    -   [Personas](#personas)
    -   [Requirements](#requirements)
    -   [Design](#design)
    -   [Alternatives](#alternatives)
- [FAQ](#faq)
- [Open Questions](#open-questions)
    
<!-- /toc -->

## Summary

This proposal defines the inference gateway scheduler subsystem and constrains its scope. The scheduler is responsible for determining which endpoints the load balancer should route requests to. Future proposals may extend its scope.

## Goals

- The scheduler should be reasonably fast - decide request mapping to endpoints within O(10ms) on average
- The scheduler should be effective - requiring little configuration out of the box to get great performance
- The scheduler should be maintainable - new in-tree features should compose cleanly
- The scheduler should be forkable - downstream consumers should expect some stability of interface
- The scheduler should be educatable - extending the [model server protocol](../003-model-server-protocol/) with new metrics or adding a new source of data should be minimally invasive
- The scheduler should be replaceable - the reference endpoint picker implementation should support delegating scheduling decisions per pool to an alternative **replacement scheduler**

## Non-Goals

- Dynamic reconfiguration of the reference scheduler algorithms at runtime
- Being a general scheduler framework for load balancing
- Determining the characteristics of the underlying model servers and hardware

## Proposal

### Personas

These are the personas we target with the scheduler subsystem:

#### OSS Algorithm Researcher

The OSS Researcher forks and extends the reference scheduler to add new algorithmic improvements and shows how it impacts workloads. They:

- Provide a replacement scheduler OR extend the reference scheduler
- Test their changes repeatedly against a set of scenarios
- Validate that their changes do not regress other scenarios
- Propose changes to the reference scheduler or the replacement scheduler protocol

#### Production Algorithm Contributor

The production algorithm contributor is an ML engineer or platform owner who observes that a specific scheduling outcome is non-optimal for their workloads and must rapidly fix the issue and get it live. They:

- Fix a scheduler bug OR Extend the reference scheduler with changes specific to their environment
- Quickly deploy a custom EPP with their changes to their environment, and sustain that fix until upstream merges
- Add new test cases to validate their issue is resolved and does not regress
- If necessary, open a feature request and proposal to cover the novel requirement

#### Inference Platform Admin

The Inference Platform Admin creates and manages the infrastructure necessary to run LLM workloads. They:

- Configure the model server under an InferencePool to accomplish the objectives of the workloads
- Configure the scheduler associated with an InferencePool to be more efficient or more predictable
- Observe rollouts for degradation of existing workload performance and stop rollout

#### Inference Workload Owner

An Inference Workload Owner persona owns and manages 1 or many Generative AI Workloads.  They:

- Configure API objects to leverage new algorithm features in test and production environments
- Reproducibly measure production traffic against new algorithms
- Identify regressions in performance when new algorithm changes are rolled out via alerting

### Requirements

We desire the following outcomes from the reference scheduler:

1. Keep model servers optimally utilized without saturating
2. Make user-visible request latency more predictable
3. Provide isolation between multiple workloads on the same model servers before saturation
4. Prioritize and fairly share resources between multiple workloads on the same model servers after saturation

We desire the following outcomes from the act of using a replacement scheduler:

1. Fast iteration with the ML ecosystem, namely other languages
2. Benefit from existing informers without having multiple implementations
3. Acceptable speed of scheduling for 10-1000 QPS systems

### Design

We expect the following challenges to be addressed by the reference scheduler design:

1. Understand the cost of an incoming request and its impact on the target model server before placing it
2. Track the cost of previously issued requests to avoid overloading servers
3. Integrate future cost features such as prefix cache routing into a holistic cost model
4. Support heterogenous model server capabilities in terms of capacity, latency, memory, and features

#### Reference Scheduler

The reference scheduler will be a monolithic Golang scheduler that is expected to run cooperatively with other instances of the scheduler with the same configuration or with appropriate 1 version/config skew.

The reference scheduler receives a list of **candidate endpoints** from the EPP and is responsible for selecting a match.

The reference scheduler is **informed** about the current state of model servers via **informers**, of which the current informer is a fast-polling loop retrieving model server metrics via the [model server protocol](../003-model-server-protocol/).

The reference scheduler is configured with a series of **predicates** that **filter** candidate endpoints, removing impossible matches.  If no matches or only one match is feasible, that endpoint is selected. If multiple matches are made, the scheduler will consult a list of configured **scorers** to **score** the matches into a **prioritized** list of endpoints, and then **sample** from that list.

Once an endpoint is selected, the endpoint is **assumed** to be running that request until the EPP observes the termination of that request (most common) OR an informer invalidates the execution of those requests.  The scheduler must integrate the impact of assumed load to with informer state, especially when traffic spikes.

Given that we anticipate a significant amount of future work to integrate heterogenous hardware (different generations / topologies) and heterogeous server roles (prefill-heavy, prefill/decode split, latency objectives), we expect that there will be an **assignment** informer that partitions the candidate endpoints over multiple dimensions for the scheduler.  This will decouple the scheduling algorithm from the process of determining the capacity and suitability of different model servers to different dimensions of request cost.

#### Replacement Scheduler

The replacement scheduler will be a low-latency mechanism for out-of-process execution of the core endpoint selection option.  The replacement scheduler will accept one or more requests to schedule, a list of endpoints, and optionally the associated informer state for those endpoints. The replacement scheduler will return one or zero endpoints per request.

#### Scheduler Validation

The proper functioning of the scheduler to prevent regression of performance is critical.  A multi-level test strategy will be required:

- Unit tests that verify scheduling decisions are accurate for all predictates and scorers
- Integration tests that verify concurrent execution as well as cooperative scheduling
- End to end tests that verify production traces against default scheduling achieve specific behavior

 A benchmarking harness will be provided to capture and reproduce a production trace, primarily to aid algorithmic contributors. A small but diverse set of production traces will be used initially to anchor expectations, and scaling both the number of supported traces and efficient regression testing at scale will be critical.

 We anticipate that accelerator availability will limit the scale of e2e testing and contribution. We will develop a **model server stub** that can emulate the behavior of the core expected algorithm for model servers and does not require accelerators. We will support both time-accurate and configurable ratio emulation to allow fast execution.

### Alternatives

#### Replaceable but not extensible scheduler

A non-extensible scheduler would be a black-box that could be replaced, and would be ideal if we do not intend the reference implementation to be featureful or if there is no wide set of scheduler features valuable to many users.

Given that we desire to have a strong out of the box reference implementation that improves performance for many users with no configuration, we do not select this alternative.

#### Highly-parameterizable scheduler

A parameterizable scheduler would have a rich configuration syntax exposed to InferencePool admins (and potentially InferenceModel users).  It would be ideal if most inference workloads had no similarities and every workload needed to be configured at the pool level or higher.

Given that we desire to have a strong reference implementation that improves performance for many users with no out of the box configuration, and that we desire to have many implementations able to directly consume the InferenceModel and InferencePool APIs, we at this time recommend not exposing full configurability of the extension via the Inference* APIs (collectively referred to as Model Routing APIs).  Instead, we recommend that algorithms be configurable either by parameterization to the EPP until we have clear design evidence for a need to add new CRDs.  At that time, in keeping with the project principles around API extension, we will reassess.