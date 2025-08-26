
# Gateway API Inference Extension

## Proposal Status
 ***Implemented/Obsolete*** 
 - Refer to [the InferencePool v1 API review](https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/1173) for the InferencePool modifications
 - Refer to [the InferenceModel evolution proposal](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/1199-inferencemodel-api-evolution) for the InferenceModel modifications
 - Refer to the `/api/` & `/apix/` directories for the current status


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
- Model serving differs from web-serving in critical ways. One of these is the existence of multiple models for the same service, which can materially impact behavior, depending on the model served. As opposed to a web-service that has mechanisms to render implementation changes invisible to an end user 

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

        Spec   InferencePoolSpec
        Status InferencePoolStatus
}

type InferencePoolSpec struct {
        // Selector defines a map of labels to watch model server pods
        // that should be included in the InferencePool.
        // In some cases, implementations may translate this field to a Service selector, so this matches the simple
        // map used for Service selectors instead of the full Kubernetes LabelSelector type.
        // If sepecified, it will be applied to match the model server pods in the same namespace as the InferencePool.
        // Cross namesoace selector is not supported.
        Selector map[LabelKey]LabelValue `json:"selector"`

        // TargetPortNumber defines the port number to access the selected model servers.
        // The number must be in the range 1 to 65535.
        TargetPortNumber int32 `json:"targetPortNumber"`

        // EndpointPickerConfig specifies the configuration needed by the proxy to discover and connect to the endpoint
        // picker service that picks endpoints for the requests routed to this pool.
        EndpointPickerConfig `json:",inline"`
}

// EndpointPickerConfig specifies the configuration needed by the proxy to discover and connect to the endpoint picker extension.
// This type is intended to be a union of mutually exclusive configuration options that we may add in the future.
type EndpointPickerConfig struct {
        // Extension configures an endpoint picker as an extension service.
        ExtensionRef *Extension `json:"extensionRef,omitempty"`
}

// Extension specifies how to configure an extension that runs the endpoint picker.
type Extension struct {
      // Reference is a reference to a service extension.
      ExtensionReference `json:",inline"`

      // ExtensionConnection configures the connection between the gateway and the extension.
      ExtensionConnection `json:",inline"`
}

// ExtensionReference is a reference to the extension.
//
// If a reference is invalid, the implementation MUST update the `ResolvedRefs`
// Condition on the InferencePool's status to `status: False`. A 5XX status code MUST be returned
// for the request that would have otherwise been routed to the invalid backend.
type ExtensionReference struct {
      // Group is the group of the referent.
      // The default value is "", representing the Core API group.
      Group *Group `json:"group,omitempty"`

      // Kind is the Kubernetes resource kind of the referent. For example
      // "Service".
      //
      // Defaults to "Service" when not specified.
      //
      // ExternalName services can refer to CNAME DNS records that may live
      // outside of the cluster and as such are difficult to reason about in
      // terms of conformance. They also may not be safe to forward to (see
      // CVE-2021-25740 for more information). Implementations MUST NOT
      // support ExternalName Services.
      Kind *Kind `json:"kind,omitempty"`

      // Name is the name of the referent.
      Name ObjectName `json:"name"`

      // The port number on the service running the extension. When unspecified,
      // implementations SHOULD infer a default value of 9002 when the Kind is
      // Service.
      PortNumber *PortNumber `json:"portNumber,omitempty"`
}

// ExtensionConnection encapsulates options that configures the connection to the extension.
type ExtensionConnection struct {
      // Configures how the gateway handles the case when the extension is not responsive.
      // Defaults to failClose.
      FailureMode *ExtensionFailureMode `json:"failureMode"`
}

// ExtensionFailureMode defines the options for how the gateway handles the case when the extension is not
type ExtensionFailureMode string


// PoolStatus defines the observed state of InferencePool from a Gateway.
type PoolStatus struct {
      // GatewayRef indicates the gateway that observed state of InferencePool.
      GatewayRef corev1.ObjectReference `json:"parentRef"`

      // Conditions track the state of the InferencePool.
      //
      // Known condition types are:
      //
      // * "Accepted"
      // * "ResolvedRefs"
      Conditions []metav1.Condition `json:"conditions,omitempty"`
}
```

**InferenceModel**
```golang
// InferenceModel represents a set of Models/Adapters that are multiplexed onto one 
// or more InferencePools. This resource is managed by the "Inference Workload Owner"
// persona. The Inference Workload Owner persona is: a team that trains, verifies, and
// leverages a large language model from a model frontend, drives the lifecycle
// and rollout of new versions of those models, and defines the specific
// performance and latency goals for the model. These workloads are
// expected to coexist within an InferencePool: sharing compute capacity with other
// InferenceModels, with sharing limitations defined by the Inference Platform Admin.
type InferenceModel struct {
        metav1.ObjectMeta
        metav1.TypeMeta

        Spec InferenceModelSpec
        Status InferenceModelStatus
}

type InferenceModelSpec struct {
        // The name of the model as it will be set in the "model" parameter for an incoming request.
        // ModelNames are expected to be unique for a specific InferencePool 
        // (names can be reused for a different pool in the same cluster). 
        // The modelName with the oldest creation timestamp is retained, and the incoming
        // InferenceModel's Ready status is set to false with a corresponding reason.
        // In the rare case of a race condition, one Model will be selected randomly to be considered valid, and the other rejected.
        // Names can be reserved without an underlying model configured in the pool.
        // This can be done by specifying a target model and setting the weight to zero,
        // an error will be returned specifying that no valid target model is found.
        ModelName string
        // Optional
        // Defines how important it is to serve the model compared to other models referencing the same pool.
        // Criticality impacts how traffic is handled in resource constrained situations. It handles this by 
        // queuing or rejecting requests of lower criticality. InferenceModels of an equivalent Criticality will
        // fairly share resources over throughput of tokens. In the future, the metric used to calculate fairness,
        // and the proportionality of fairness will be configurable.
        Criticality *Criticality
        // Optional.
        // Allow multiple versions of a model for traffic splitting.
        // If not specified, the target model name is defaulted to the ModelName parameter.
        // ModelName is often in reference to a LoRA adapter.
        TargetModels []TargetModel
        // PoolRef is a reference to the inference pool, the pool must exist in the same namespace.
        PoolRef PoolObjectReference
}

// PoolObjectReference identifies an API object within the namespace of the
// referrer.
type PoolObjectReference struct {
        // Group is the group of the referent.
        Group Group

        // Kind is kind of the referent. For example "InferencePool".
        Kind Kind

        // Name is the name of the referent.
        Name ObjectName
}

// Defines how important it is to serve the model compared to other models.
// Criticality is intentionally a bounded enum to contain the possibilities that need to be supported by the load balancing algorithm. Any reference to the Criticality field should ALWAYS be optional(use a pointer), and set no default.
// This allows us to union this with a oneOf field in the future should we wish to adjust/extend this behavior.
type Criticality string
const (
        // Critical defines the highest level of criticality. Requests to this band will be shed last.
        Critical Criticality = "Critical"

        // Standard defines the base criticality level and is more important than Sheddable but less
        // important than Critical. Requests in this band will be shed before critical traffic.
        // Most models are expected to fall within this band.
        Standard Criticality = "Standard"

        // Sheddable defines the lowest level of criticality. Requests to this band will be shed before
        // all other bands.
        Sheddable Criticality = "Sheddable"
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
        Weight *int32
}

// InferenceModelStatus defines the observed state of InferenceModel
type InferenceModelStatus struct {
	// Conditions track the state of the InferenceModel.
	Conditions []metav1.Condition
}
```

### Yaml Examples

#### InferencePool(s)
Here we create a pool that selects the appropriate pods
```yaml
apiVersion: inference.x-k8s.io/v1alpha2
kind: InferencePool
metadata:
  name: base-model-pool
spec:
  selector:
    app: llm-server
  targetNumber: 8080
  extensionRef:
    name: infra-backend-v1-app
```

#### InferenceModel

Here we consume the pool with two InferenceModels. Where `sql-code-assist` is both the name of the model and the name of the LoRA adapter on the model server. And `npc-bot` has a layer of indirection for those names, as well as a specified criticality. Both `sql-code-assist` and `npc-bot` have available LoRA adapters on the InferencePool and routing to each InferencePool happens earlier (at the K8s Gateway).
```yaml
apiVersion: inference.x-k8s.io/v1alpha2
kind: InferenceModel
metadata:
  name: sql-code-assist
spec:
  poolRef:
    name: base-model-pool
---
apiVersion: inference.x-k8s.io/v1alpha2
kind: InferenceModel
metadata:
  name: npc-bot
spec:
  criticality: Critical
  targetModels:
    - name: npc-bot-v1
      weight: 50
    - name: npc-bot-v2
      weight: 50
  poolRef:
    name: base-model-pool
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


