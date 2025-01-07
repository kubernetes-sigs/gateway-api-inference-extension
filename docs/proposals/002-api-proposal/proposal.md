
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
- [FAQ](#faq)
- [Open Questions](#open-questions)
    
<!-- /toc -->

## Summary

This proposal presents 2 new CRD objects to express the needs of the Gateway API Inference Extension. **InferencePool** and **InferenceModel**. The InferencePool is the logical grouping of compute, owned by the Inference Platform Admin persona. While the InferenceModel defines the serving objectives of a specific model or LoRA adapter, and is owned by the Inference Workload Owner.


## Goals

- Drive concensus on direction of Gateway API Inference Extension Solution
- Documentation of API decisions for posterity

## Non-Goals

- Hash out every implementation detail
- Be a formal KEP

## Proposal

### Personas

Before diving into the details of the API, decriptions of the personas will help shape the thought process of the API design.

#### Inference Platform Admin

The Inference Platform Admin creates and manages the infrastructure necessary to run LLM workloads. Including handling Ops for: 
  - Hardware
  - Model Server
  - Base Model
  - Resource Allocation for Workloads
  - Gateway configuration
  - etc

#### Inference Workload Owner

An Inference Workload Owner persona owns and manages 1 or many Generative AI Workloads (LLM focused *currently*). This includes:
- Defining criticality
- Managing fine-tunes
  - LoRA Adapters
  - System Prompts
  - Prompt Cache
  - etc.
- Managing rollout of adapters

### Axioms 

The API design is based on these axioms:

- Pools of shared compute should be *discrete* for scheduling to properly work
- Pod-level scheduling should not be handled by a high-level gateway 
- Simple services should be simple to define (or are implicitly defined via reasonable defaults)
- This solution should be composable with other Gateway solutions and flexible to fit customer needs
- The MVP will heavily assume requests are done using the OpenAI spec, but open to extension in the future
- The Gateway should route in a way that does not generate a queue of requests at the model server level
/* REVIEW: Note that models are served in a way that leads to many models running at the same time, vs many existing kube networking assumptions that assume there is a user-facing service that the implementation changes in user-invisible ways over time */

The [PoC](https://youtu.be/NUBZg_uqqXk?si=v681EeYdGUGEVqQQ&t=1458) was focused on lower-level scheduling. And the API follows that similar logic, which lead to the proposal of the **InferencePool**.

### InferencePool

The InferencePool at its core is a logical grouping of compute, expressed in the form of Pods (typically model servers), akin to a K8s Service. The InferencePool would deploy its own routing, and offer administrative configuration to the Platform Admin. 

 It is expected for the InferencePool to:
 - Enforce fair consumption of resources across competing workloads
 - Efficiently route requests across shared compute (as displayed by the PoC)
 
It is _not_ expected for the InferencePool to:
 - Enforce any common set of adapters or base models are available on the Pods
 - Manage Deployments of Pods within the Pool
 - Manage Pod lifecycle of pods within the pool 

Additionally, any Pod that seeks to join an InferencePool would need to support a protocol, defined by this project, to ensure the Pool has adequate information to intelligently route requests.

### InferenceModel

An InferenceModel allows the Inference Workload Owner to define:
- Which Model/LoRA adapter(s) to consume .
  - Mapping from a client facing model name to the target model name in the InferencePool.
  - InferenceModel allows for traffic splitting between adapters _in the same InferencePool_ to allow for new LoRA adapter versions to be easily rolled out.
- Criticality of the requests to the InferenceModel.
- The InferencePools this InferenceModel is relevant to.

### Spec

**InferencePool**
```golang
// The InferencePool is a construct for pooling compute (often model servers) to
// serve large models, that have the ability to share capacity across multiple
// services (such as through prompt engineering, LoRA adapters, etc).
// InferencePools have a dependency on a Gateway that is compatible with ext-proc
// (External Processing). When a new InferencePool object is created, a new ext proc
// deployment is created. InferencePools require at minimum a single InferenceModel to
// be subscribed to them to accept traffic, any traffic with a model not
// defined within an InferenceModel will be rejected.
type InferencePool struct {
        metav1.ObjectMeta
        metav1.TypeMeta

        Spec InferencePoolSpec
}

type InferencePoolSpec struct {
        // ModelServerSelector uses label selection to watch model server pods
        // that should be included in the InferencePool. ModelServers should not
        // be with any other Service or InferencePool, that behavior is not supported
        // and will result in sub-optimal utilization.
/* REVIEW: This is generally true for all services, but since the ig needs to deal with monitoring, health checking, and changes to membership in the pool (i.e. a member removed from the selector or gracefully shutting down), I think it's more of a distraction than necessary. */
        ModelServerSelector map[string]string `json:"modelServerSelector,omitempty"`
}
```

**InferenceModel**
```golang
// InferenceModel represents a set of Models/Adapters that are multiplexed onto one 
// or more Inferencepools. This resource is managed by the "Inference Workload Owner"
// persona. The Inference Workload Owner persona is: a team that trains, verifies, and
// leverages a large language model from a model frontend, drives the lifecycle
// and rollout of new versions of those models, and defines the specific
// performance and latency goals for the model. These workloads are
// expected to operate within an InferencePool sharing compute capacity with other
/* REVIEW: "operate" sounds wierd */
// InferenceModels, defined by the Inference Platform Admin. We allow a user who
// has multiple InferenceModels across multiple pools (with the same config) to
// specify the configuration exactly once, and deploy to many pools
/* REVIEW: be more consistent about "to" or "on" or "in" */
// simultaneously. Enabling a simpler config and single source of truth
/* REVIEW: broken clause, this should be part of the previous sentence or more naturally integrated */
// for a given user. InferenceModel ModelNames are unique for a given InferencePool,
/* Review: missing clause?  How is uniqueness enforced? */
type InferenceModel struct {
        metav1.ObjectMeta
        metav1.TypeMeta

        Spec InferenceModelSpec
}

type InferenceModelSpec struct {
        // The name of the model as the users set in the "model" parameter in the requests.
        // The name should be unique among the workloads that reference the same backend pool.
/* REVIEW: Describe what happens if the name isn't unique, and specify the behavior here */
        // This is the parameter that will be used to match the request with. In the future, we may
        // allow to match on other request parameters. The other approach to support matching on 
        // on other request parameters is to use a different ModelName per HTTPFilter.
        // Names can be reserved without implementing an actual model in the pool.
/* REVIEW: "without a model of that name already being configured in the pool"?  "Implementing a model" doesn't sound right */
        // This can be done by specifying a target model and setting the weight to zero,
        // an error will be returned specifying that no valid target model is found.
        ModelName string
        // Optional
        // Defines how important it is to serve the model compared to other models referencing the same pool.
/* REVIEW: define the behavior when multiple models have the same criticality in a sentence */
        Criticality *Criticality
        // Optional.
	    // Allow multiple versions of a model for traffic splitting. 
	    // If not specified, the target model name is defaulted to the ModelName parameter.
        // ModelName is often in reference to a LoRA adapter.
        TargetModels []TargetModel
        // Reference to the InferencePool that the model registers to. It must exist in the same namespace.
        PoolReference *LocalObjectReference
}

// Defines how important it is to serve the model compared to other models.
/* REVIEW: Describe why there are only three levels and suggest that we don't expect to extend this enumeration, probably within the field name above.  Open enumerations can be problematic, but a novice user reading this could reasonably ask "why not just have a priority number?" and so we should answer that. */
type Criticality string
const (
    // Most important. Requests to this band will be shed last.
    Critical  Criticality = "Critical"
    // More important than Sheddable, less important than Critical.
    // Requests in this band will be shed before critical traffic.
    Default  Criticality = "Default"
    // Least important. Requests to this band will be shed before all other bands.
    Sheddable  Criticality = "Sheddable"
 )

// TargetModel represents a deployed model or a LoRA adapter. The
// Name field is expected to match the name of the LoRA adapter
// (or base model) as it is registered within the model server. This
// assumes that the model exists on the model server and it is the
// responsibility of the user to validate a correct match. Should a model fail
// to exist at request time, the error is processed by the extension,
// and then emitted on the appropriate InferenceModel object status.
type TargetModel struct {
        // The name of the adapter as expected by the ModelServer.
        Name string
        // Weight is used to determine the percentage of traffic that should be 
        // sent to this target model when multiple versions of the model are specified.
/* REVIEW: What's the closest analog to weight in the existing API? */
        Weight int
}

// LocalObjectReference identifies an API object within the namespace of the
// referrer.
type LocalObjectReference struct {
	// Group is the group of the referent. 
	Group Group

	// Kind is kind of the referent. For example "InferencePool".
	Kind Kind

	// Name is the name of the referent.
	Name ObjectName
}

```

### Yaml Examples

#### InferencePool(s)
Here we create a pool that selects the appropriate pods
```yaml
apiVersion: inference.x-k8s.io/v1alpha1
kind: InferencePool
metadata:
  name: base-model-pool
  modelServerSelector:
  - app: llm-server
```

#### InferenceModel

Here we consume the pool with two InferenceModels. Where `sql-code-assist` is both the name of the model and the name of the LoRA adapter on the model server. And `npc-bot` has a layer of indirection for those names, as well as a specified criticality. Both `sql-code-assist` and `npc-bot` have available LoRA adapters on the InferencePool and routing to each InferencePool happens earlier (at the K8s Gateway).
```yaml
apiVersion: inference.x-k8s.io/v1alpha1
kind: InferenceModel
metadata:
  name: sql-code-assist
spec:
  modelName: sql-code-assist
  poolRef: base-model-pool
---
apiVersion: inference.x-k8s.io/v1alpha1
kind: InferenceModel
metadata:
  name: npc-bot
spec:
  modelName: npc-bot
  criticality: Critical
  targetModels:
    targetModelName: npc-bot-v1
      weight: 50
    targetModelName: npc-bot-v2
      weight: 50 	
  poolRef: base-model-pool
```


### Alternatives

#### Key Decisions

Our alternatives hinge on some key decisions:
- Allowing HTTPRoute to treat the InferencePool as the backendRef
  - Whereas the alternatives might have the InferenceModel as the backend ref
- Creating a separate layer of abstraction, instead of extending HTTPRoute
  - Explained in more detail in the LLMRoute section

#### InferenceModel as a backend ref

We toyed with the idea of allowing an InferenceModel be the target of an HTTPRouteRules backend ref. However, doing so would require the Kubernetes Gateway to be able to interpret body level parameters (assuming OpenAI protocol continues to require the model param in the body), and require that the HTTPRoute also specify the backend the InferenceModel is intended to run on. Since our primary proposal already specifies the backend, packing this functionality would require substantial work on the Kubernetes Gateway, while not providing much flexibility.

#### LLMRoute

Our original idea was to define all InferenceModel config at the Kubernetes Gateway layer, and have no InferencePool. This is inherently challenging, as LLMRoute would become a superset of HTTPRoute, or the Gateway would become bespoke, and work only for the LLMRoute use case.

## FAQ
- **Why 2 layers of weighting?** (HttpRoute & InferenceModel)
  - Feasibly done - No extension of HttpRoute. Just works, as InferencePool operates like a service.
  - Complexity is only expressed during transition states (model version upgrade)
  - Keeps Pools self contained - multiple K8s gateways can direct traffic to the same pool without needing to re-express Pool-level behavior
- **What is an InferencePool attempting to define?**
  - InferencePool groups resources that should be shared over the InferenceModels that are affiliated with the pool
  - Best practice would also suggest keeping the same base model for all ModelServers in the pool, but that is not enforced
- **How is this deployed?**
  - We will follow [common patterns](https://gateway.envoyproxy.io/docs/tasks/quickstart/#installation) to install the CRDs & Controllers
- **Are all controllers necessary for this solution going to be provided by this project?**
  - Yes


