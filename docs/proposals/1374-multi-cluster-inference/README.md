# Multi-Cluster Inference Pooling

Author(s): @danehans, @bexxmodd, @robscott

## Proposal Status

 ***Draft***

## Summary

An Inference Gateway (IG) provides efficient routing to LLM workloads in Kubernetes by sending requests to an Endpoint Picker (EPP) associated with
an [InferencePool](https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/) and routing the request to a backend model server
based on the EPP-provided endpoint. Although other multi-cluster inference approaches may exist, this proposal extends the current model to support
multi-cluster routing so capacity in one cluster can serve traffic originating in another cluster or outside the cluster.

### Why Multi-Cluster?

GPU capacity is scarce and fragmented. Many users operate multiple clusters across regions and providers. A single cluster rarely satisfies peak or
sustained demand, so a prescribed approach is required to share GPU capacity across clusters by:

- Exporting an InferencePool from a source (“exporting”) cluster.
- Importing the exported InferencePool into one or more destination (“importing”) clusters with enough detail for IGs to route requests to the associated
  remote model server Pods.

### Goals

- Enable IGs to route to a group of common model server Pods, e.g. InferencePools, that exist in different clusters.
- Align the UX with familiar [Multi-Cluster Services (MCS)](https://multicluster.sigs.k8s.io/concepts/multicluster-services-api/) concepts (export/import).
- Keep the API simple and implementation-agnostic.

### Non-Goals

- Managing DNS or automatic naming.
- Over-specifying implementation details to satisfy a single approach to multi-cluster inference.

## Design Proposal

The Multi-Cluster Inference Pooling (MCIP) model will largely follow the Multi-Cluster Services (MCS) model, with a few key differences:

- DNS and ClusterIP resolution will be omitted, e.g. ClusterSetIP.
- A separate export resource will be avoided, e.g. ServiceExport, by inlining the concept within InferencePool.

An InferencePoolImport resource is introduced that is meant to be fully managed by a controller. This resource provides the information
required for IGs to route LLM requests to model server endpoints of an InferencePool in remote clusters. How the IG routes the request to the remote
cluster is implementation-specific.

### Routing Modes

The proposal supports the following routing modes:

- Endpoint Mode: An IG of an importing cluster routes to endpoints selected by the EPP of the exported InferencePool. Pod and Service network connectivity
  MUST exist between cluster members.
- Parent Mode: An IG of an importing cluster routes to parents, e.g. Gateways, of the exported InferencePool. Parent connectivity MUST exist between cluster
  members.

### Sync Topology (Implementation-Specific)

MCIP supports two distribution topologies. The API does not change between them (same export annotation + InferencePoolImport). Implementations pick one—or support both.

1. **Hub/Spoke**
   - A hub controller has visibility into member clusters.
   - It watches exported InferencePools and creates/updates the corresponding InferencePoolImport (same namespace/name) in each member cluster.
   - Typical when a central control plane has K8s API server access for each member cluster.
   - Consider [KEP-5339-style](https://github.com/kubernetes/enhancements/tree/master/keps/sig-multicluster/5339-clusterprofile-plugin-credentials) pluggable credential issuance to avoid hub-stored long-lived secrets.

2. **Push/Pull**
   - A cluster-local controller watches exported InferencePools and publishes export state to a central hub.
   - A cluster-local controller watches the central hub and CRUDs the local InferencePoolImport.
   - Typical when you want no hub-stored member credentials, looser coupling, and fleet-scale fan-out.

### Workflow

1. **Export an InferencePool:** An [Inference Platform Owner](https://gateway-api-inference-extension.sigs.k8s.io/concepts/roles-and-personas/)
   exports an InferencePool by annotating it.
2. **Distribution (topology-dependent, API-agnostic):**
   - **Hub/Spoke:** A central hub controller watches exported InferencePools and mirrors a same-name/namespace InferencePoolImport into each member cluster, updating `status.clusters[]` to reflect exporting clusters.
   - **Push/Pull:** A cluster-local controller watches exported InferencePools and publishes export records to a central hub. In each member cluster, a controller watches the hub and CRUDs the local InferencePoolImport (same name/namespace), maintaining `status.clusters[]`.
3. **Importing Controller (common):**
   - Watches local InferencePoolImport and programs the IG dataplane for Endpoint Mode or Parent Mode.
4. **Data Path:**
   The data path is dependant on the export mode selected by the implementation.
   - Endpoint Mode: Client → local IG → (make scheduling decision) → local/remote EPP → selected model server endpoint → response.
   - Parent Mode: Client → local IG → (make scheduling decision) → local EPP/remote parent → remote EPP → selected model server endpoint → response.

### InferencePoolImport Naming

The exporting controller will create an InferencePoolImport resource using the exported InferencePool namespace and name. A cluster name entry in
`status.clusters[]` is added for each cluster that exports an InferencePool with the same ns/name.

**Note:** EPP ns/name sameness is not required.

### InferencePool Selection

InferencePool selection is implementation-specific. The following are examples of how an IG may select one exported InferencePool over another:

- **Metrics-based:** Scrape EPP-exposed metrics (e.g., ready pods) to bias InferencePool choice.
- **Active-Passive:** Basic EPP readiness checks (gRPC health).

**Note:** When an exported InferencePool is selected by an IG, standard EPP semantics are used to select endpoints of that pool.

### API Changes

#### Export Annotation

The following annotation is being proposed to indicate the desire to export the InferencePool to member clusters of a ClusterSet.

The `inference.networking.x-k8s.io/export` annotation key indicates a desire to export the InferencePool:

```yaml
inference.networking.x-k8s.io/export: "<value>"
```

Supported Values:

- `ClusterSet` – export to all members of the current [ClusterSet](https://multicluster.sigs.k8s.io/api-types/cluster-set/).

**Note:** Additional annotations, e.g. region/domain scoping, filter clusters in the ClusterSet, routing mode configuration, etc. and
potentially adding an InferencePoolExport resource may be considered in the future.

#### InferencePool Status

A TBD InferencePool parent condition type will be introduced to surface status of the exported InferencePool. An implementation MUST set
this status condition to `True` when the annotated InferencePool has been exported to all member clusters of the ClusterSet and `False`
for all other reasons. When the export annotation is removed from the InferencePool, an implementation MUST remove this condition type.

#### InferencePoolImport

A cluster-local, controller-managed resource that represents an imported InferencePool. It primarily communicates a relationship between an exported
InferencePool and the exporting cluster name. It is not user-authored; status carries the effective import. Inference Platform Owners can reference the InferencePoolImport, even if the local cluster does not have an InferencePool. In the context of Gateway API, it means that an HTTPRoute can be configured to reference an InferencePoolImport to route matching requests to remote InferencePool endpoints. This API will be used almost exclusively for tracking endpoints,
but unlike MCS, we actually have two distinct sets of endpoints to track:

1. Endpoint Picker Extensions (EPPs)
2. InferencePool parents, e.g. Gateways

Key ideas:

- Map exported InferencePool to exporting cluster name.
- Name/namespace sameness with the exported InferencePool (avoids extra indirection).
- Conditions: Surface at least one condition, e.g. Accepted, to indicate to a user that the InferencePoolImport is ready to be used.

See the full Go type below for additional details.

## Controller Responsibilities

**Export Controller:**

- Discover exported InferencePools.
- For each ClusterSet member cluster, CRUD InferencePoolImport (mirrored namespace/name).
- Populate `inferencepoolimport.status.clusters[]` entries with the cluster name associated with the exported InferencePool.

**Import Controller:**

- Watch InferencePoolImports.
- Program the IG data plane to either:
  - Connect to to remote EPPs and exported InferencePool endpoints (Endpoint Mode).
  - Connect to parent(s) of the exported InferencePool (Parent Mode).
  - Load-balance matching requests.

## Examples

### Exporting Cluster (Cluster A) Manifests

In this example, Cluster A exports the InferencePool to all clusters in the Cluster set using Endpoint Mode. This will
cause the exporting controller to create an InferencePoolImport resource in all clusters.

```yaml
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: llm-pool
  namespace: example
  annotations:
    inference.networking.x-k8s.io/export: "ClusterSet" # Export the pool to all clusters in the ClusterSet
spec:
  endpointPickerRef:
    name: epp
    portNumber: 9002
  selector:
    matchLabels:
      app: my-model
  targetPorts:
  - number: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: epp
  namespace: example
spec:
  selector:
    app: epp
  ports:
  - name: ext-proc
    port: 9002
    targetPort: 9002
    appProtocol: http2
  type: LoadBalancer # EPP exposed via LoadBalancer
```

### Importing Cluster (Cluster B) Manifests

In this example, the InferencePlatform Owner has configured an HTTPRoute to route to endpoints of the Cluster A InferencePool
by referencing the InferencePoolImport as a `backendRef`. The parent IG(s) of the HTTPRoute are responsible for routing to the
endpoints selected by the EPP referenced by the exported InferencePool.

The InferencePoolImport is controller-managed; shown here only to illustrate the expected status shape.

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: InferencePoolImport
metadata:
  name: llm-pool      # mirrors exporting InferencePool name
  namespace: example  # mirrors exporting InferencePool namespace
status:
  clusters:
  - name: cluster-a
  conditions:
  - type: Accepted
    status: "True"
  observedGeneration: 1
---
# Route in the importing cluster that targets the imported pool
apiVersion: gateway.networking.k8s.io/v1beta1
kind: HTTPRoute
metadata:
  name: llm-route
  namespace: example
spec:
  parentRefs:
  - name: inf-gw
  hostnames:
  - my.model.com
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /completions
    backendRefs:
    - group: inference.networking.x-k8s.io
      kind: InferencePoolImport
      name: llm-pool
```

An implementation MUST conform to Gateway API specifications, including when the HTTPRoute contains InferencePool and InferencePoolImport `backendRefs`,
e.g. `weight`-based load balancing. In the following example, traffic MUST be split equally between Cluster A and B InferencePool endpoints when
using the following `backendRefs`:

```yaml
    backendRefs:
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: llm-pool
      weight: 50
    - group: inference.networking.x-k8s.io
      kind: InferencePoolImport
      name: llm-pool
      weight: 50
```

**Note:** The above example does not export the local "llm-pool" InferencePool. If this InferencePool was exported, it would be included in
the example InferencePoolImport and the implementation would be responsible for balancing the traffic between the two pools.

### Go Types

```go
package v1alpha1

import (
    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +kubebuilder:object:root=true
// +kubebuilder:resource:scope=Namespaced,shortName=ipimp
// +kubebuilder:subresource:status
//
// InferencePoolImport represents an imported InferencePool from another cluster.
// This resource is controller-managed; users typically do not author it directly.
type InferencePoolImport struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`

    // Spec is intentionally empty since this resource is fully managed by controllers.
    // Future versions may surface user-tunable knobs here (e.g., local policy hints).
    Spec InferencePoolImportSpec `json:"spec,omitempty"`

    // Status communicates the imported targets (EPPs and/or parents) and readiness.
    Status InferencePoolImportStatus `json:"status,omitempty"`
}

type InferencePoolImportSpec struct{}

type InferencePoolImportStatus struct {
    // Clusters is the set of exporting clusters that currently back this import.
    //
    // +kubebuilder:validation:Required
    Clusters []ImportedCluster `json:"clusters"`

    // Conditions include:
    // - Accepted: controller synthesized a valid import.
    //
    // +kubebuilder:validation:Optional
    Conditions []metav1.Condition `json:"conditions,omitempty"`
}

type ImportedCluster struct {
    // Name of the exporting cluster (must be unique within the list). 
    //
    // +kubebuilder:validation:Required
    Name string `json:"name"`
}

// +kubebuilder:object:root=true
type InferencePoolImportList struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ListMeta `json:"metadata,omitempty"`
    Items           []InferencePoolImport `json:"items"`
}
```

### Failure Mode

EPP failure modes continue to work as-is.

#### EPP Selection

Since an IG decides which EPP to use for endpoint selection when multiple InferencePool/InferencePoolImport `backendRefs` exist,
an implementation MAY use EPP metrics and/or health data to make a load-balancing decision.

## Alternatives

### Option 1: Reuse MCS API for EPP

Reuse MCS to export EPP Services. This approach provides simple infra, but may be confusing to users (you “export EPPs” not pools) and
requires a separate MCS parent export for parent-based inter-cluster routing.

**Pros**:

- Reuses existing MCS infrastructure.
- Relatively simple to implement.

**Cons**:

- Referencing InferencePools in other clusters requires you to create an InferencePool locally.
- In this model, you don’t actually choose to export an InferencePool, you export the EPP or InferencePool parent(s) service, that could lead to confusion.
- InferencePool is meant to be a replacement for a Service so it may seem counterintuitive for a user to create a Service to achieve multi-cluster inference.

## Option 2: New MCS API

One of the key pain points we’re seeing here is that the current iteration of the MCS API requires a tight coupling between name/namespace and kind, with Service being the only kind of backend supported right now. This goes against the broader SIG-Network direction of introducing more focused kinds of backends (like InferencePool). To address this, we could create a resource that has an `exportRef` that allows for exporting different types of resources.

While we were at it, we could combine the separate `export` and `import` resources that exist today, with `export` acting as the (optional) spec of this new resource, and `import` acting as `status` of the resource. Instead of `import` resources being automatically created, users would create them wherever they wanted to reference or export something to a MultiClusterService.

Here’s a very rough example:

```yaml
apiVersion: networking.k8s.io/v1
kind: MultiClusterService
metadata:
  name: epp
  namespace: example
spec:
  exportRef:
    group: v1
    kind: Service
    name: epp
    scope: ClusterSet
status:
  conditions:
  - type: Accepted
    status: "True"
    message: "MultiClusterService has been accepted"
    lastTransitionTime: "2025-03-30T01:33:51Z"
  targetCount: 1
  ports:
  - protocol: TCP
    appProtocol: HTTP
    port: 8080
```

### Open Questions

#### InferencePool Status

- Should EPP Deployment/Pod discovery be standardized (labels/port names) for health/metrics auto-discovery?

#### Security

- Provide a standard way to bootstrap mTLS between importing IG and exported EPP/parents, e.g. use BackendTLSPolicy?
- Should the export controller mirror secrets into the importing cluster, e.g. secure metric scraping (Endpoint Mode)?

#### Scheduling and Policy

- Should we define a standard cluster preference knob (e.g., PreferLocal, Any, region-affinity, weights) on InferencePoolImport status or IG-local policy CRD?

#### EPP Scale

- If the EPP has multiple replicas, should the export controller publish per-replica addresses, e.g. service subsetting, for health/metrics scraping?

#### Ownership and Lifecycle

- What happens if the importing namespace doesn’t exist? Should the export controller surface a status condition in the exported InferencePool?
- Garbage collection when export is withdrawn (delete import?) and how to drain traffic safely.

### Prior Art

- [GEP-1748: Gateway API Interaction with Multi-Cluster Services](https://gateway-api.sigs.k8s.io/geps/gep-1748/)
- [Envoy Gateway with Multi-Cluster Services](https://gateway.envoyproxy.io/latest/tasks/traffic/multicluster-service/)
- [Multi-Cluster Service API](https://multicluster.sigs.k8s.io/concepts/multicluster-services-api/)

### References

- [Initial Multi-Cluster Inference Design Doc](https://docs.google.com/document/d/1QGvG9ToaJ72vlCBdJe--hmrmLtgOV_ptJi9D58QMD2w/edit?tab=t.0#heading=h.q6xiq2fzcaia)

### Notes for reviewers

- The InferencePoolImport CRD is intentionally status-only to keep the UX simple and controller-driven.
- The InferencePool namespace sameness simplifies identity and lets HTTPRoute authors reference imports without new indirection.
