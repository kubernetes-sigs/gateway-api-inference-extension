# Multi-Cluster InferencePools

Author(s): @danehans, @bexxmodd, @robscott

## Proposal Status

 ***Draft***

## Summary

An Inference Gateway (IG) provides efficient routing to LLM workloads in Kubernetes by sending requests to an Endpoint Picker (EPP) associated with
an [InferencePool](https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/) and routing the request to a backend model server
based on the EPP-provided endpoint. Although other multi-cluster inference approaches may exist, this proposal extends the current model to support
multi-cluster routing so capacity in one cluster can serve traffic originating in another cluster or outside the clusters.

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
- Over-specifying implementation details to satisfy a single approach to Multi-Cluster InferencePools.

## Design Proposal

The Multi-Cluster InferencePools (MCIP) model will largely follow the Multi-Cluster Services (MCS) model, with a few key differences:

- DNS and ClusterIP resolution will be omitted, e.g. ClusterSetIP.
- A separate export resource will be avoided, e.g. ServiceExport, by inlining the concept within InferencePool.

An InferencePoolImport resource is introduced that is meant to be fully managed by a controller. This resource provides the information
required for IGs to route LLM requests to model server endpoints of an InferencePool in remote clusters. How the IG routes the request to the remote
cluster is implementation-specific.

### Routing Modes

An implementation must support at least one of the following routing modes:

- Endpoint Mode: An IG of an importing cluster routes to endpoints selected by the EPP of the exported InferencePool. Pod and Service network connectivity
  MUST exist between cluster members.
- Parent Mode: An IG of an importing cluster routes to parents, e.g. Gateways, of the exported InferencePool. Parent connectivity MUST exist between cluster
  members.

### Sync Topology (Implementation-Specific)

An implementation must support at least one of the following distribution topologies. The API does not change between them (same export annotation and InferencePoolImport).

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
   - **Hub/Spoke:** A central hub controller watches exported InferencePools and mirrors a same-name/namespace InferencePoolImport into each member cluster, updating `status.controllers[]` to reflect the managing controller, exporting clusters, etc..
   - **Push/Pull:** A cluster-local controller watches exported InferencePools and publishes export records to a central hub. In each member cluster, a controller watches the hub and CRUDs the local InferencePoolImport (same name/namespace) and maintains `status.controllers[]`.
3. **Importing Controller (common):**
   - Watches local InferencePoolImport and:
     - Programs the IG dataplane based on the supported routing mode.
     - If this controller differs from the Exporting Controller, populate `status.controllers[]`.
     - Manages `status.controllers[].parentRefs` with the Group, Kind, and Name of the local parent resource, e.g. Gateway, of the InferencePoolImport.
4. **Data Path:**
   The data path is dependent on the export mode selected by the implementation.
   - Endpoint Mode: Client → local IG → (make scheduling decision) → local/remote EPP → selected model server endpoint → response.
   - Parent Mode: Client → local IG → (make scheduling decision) → local EPP/remote parent → remote EPP → selected model server endpoint → response.

### InferencePoolImport Naming

The exporting controller will create an InferencePoolImport resource using the exported InferencePool namespace and name. A cluster name entry in
`status.controllers[]` is added for each cluster that exports an InferencePool with the same ns/name.

**Note:** EPP ns/name sameness is not required.

### InferencePool Selection

InferencePool selection is implementation-specific. The following are examples of how an IG may select one exported InferencePool over another:

- **Metrics-based:** Scrape EPP-exposed metrics (e.g., ready pods) to bias InferencePool choice.
- **Active-Passive:** Basic EPP readiness checks (gRPC health).

**Note:** When an exported InferencePool is selected by an IG, standard EPP semantics are used to select endpoints of that pool.

### API Changes

#### InferencePool Annotation

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

An implementation MUST set a parent status entry with a parentRef of kind `InferencePoolImport` and the ns/name of the exported InferencePool.
This informs the user that the request to export the InferencePool has been recognized by the implementation, along with the implementation's
unique `ControllerName`.  `ControllerName` is a domain/path string that indicates the name of the controller that wrote the status entry.

An `Exported` parent condition type is being added to surface status of the exported InferencePool. An implementation MUST set
this status condition to `True` when the annotated InferencePool has been exported to all member clusters of the ClusterSet and `False`
for all other reasons. When the export annotation is removed from the InferencePool, an implementation MUST remove this condition type.

#### InferencePoolImport

A cluster-local, controller-managed resource that represents an imported InferencePool. It primarily communicates a relationship between an exported
InferencePool and the exporting cluster name. It is not user-authored; status carries the effective import. Inference Platform Owners can reference
the InferencePoolImport, even if the local cluster does not have an InferencePool. In the context of Gateway API, it means that an HTTPRoute can be
configured to reference an InferencePoolImport to route matching requests to remote InferencePool endpoints. This API will be used almost exclusively
for tracking endpoints, but unlike MCS, we actually have two distinct sets of endpoints to track:

1. Endpoint Picker Extensions (EPPs)
2. InferencePool parents, e.g. Gateways

Key ideas:

- Map exported InferencePool to exporting controller and cluster.
- Name/namespace sameness with the exported InferencePool (avoids extra indirection).
- Conditions: Surface a controller-level status condition to indicate that the InferencePoolImport is ready to be used.
- Conditions: Surface parent-level status conditions to indicate that the InferencePoolImport is referenced by a parent, e.g. Gateway.

See the full Go type below for additional details.

## Controller Responsibilities

**Export Controller:**

- Discover exported InferencePools.
- For each ClusterSet member cluster, CRUD InferencePoolImport (mirrored namespace/name).
- Populate the exported InferencePool with status to indicate a unique name of the managing controller and an `Exported`  condition that
  indicates the status of the exported pool.
- Populate an InferencePoolImport `status.controllers[]` entry with the managing controller name, cluster, and status conditions associated
  with the exported InferencePool.

**Import Controller:**

- Watch InferencePoolImports.
- Program the IG data plane to route matching requests based on the supported routing mode.
- Manage InferencePoolImport `status.controllers[].parentRefs` with a cluster-local parentRef, e.g. Gateway, when the InferencePoolImport
  is referenced by a managed HTTProute.

## Examples

### Exporting Cluster (Cluster A) Manifests

In this example, Cluster A exports the InferencePool to all clusters in the ClusterSet. This will
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

In this example, the Inference Platform Owner has configured an HTTPRoute to route to endpoints of the Cluster A InferencePool
by referencing the InferencePoolImport as a `backendRef`. The parent IG(s) of the HTTPRoute are responsible for routing to the
endpoints selected by the exported InferencePool's EPP.

The InferencePoolImport is controller-managed; shown here only to illustrate the expected status shape.

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: InferencePoolImport
metadata:
  name: llm-pool      # mirrors exporting InferencePool name
  namespace: example  # mirrors exporting InferencePool namespace
status:
  controllers:
  - controllerName: example.com/mcip-controller
    type: MultiCluster
    exportingClusters:
    - name: cluster-a
  - controllerName: example.com/ig-controller
    type: GatewayClass
    parents:
    - parentRef:
        group: gateway.networking.k8s.io
        kind: Gateway
        name: inf-gw # Cluster-local parent, e.g. gateway
      conditions:
      - type: Accepted
        status: "True"
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

The following Go types define the InferencePoolImport API being introduced by this proposal. Note that a separate Pull Request will be used
to finalize and merge all Go types into the repository.

```go
package v1alpha1

import (
    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

    v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
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

    // Spec defines the desired state of the InferencePoolImport.
    Spec InferencePoolImportSpec `json:"spec,omitempty"`

    // Status defines the current state of the InferencePoolImport.
    Status InferencePoolImportStatus `json:"status,omitempty"`
}

// Unused but defined for potential future use.
type InferencePoolImportSpec struct{}

type InferencePoolImportStatus struct {
    // Controllers is a list of controllers that are responsible for managing this InferencePoolImport.
    //
    // +kubebuilder:validation:Required
    Controllers []ImportController `json:"controllers"`
}

// ImportController defines a controller that is responsible for managing this InferencePoolImport.
type ImportController struct {
    // Name is a domain/path string that indicates the name of the controller that manages this
    // InferencePoolImport. This corresponds to the GatewayClass controllerName field when the
    // controller will manage parents of type "Gateway". Otherwise, the name is implementation-specific.
    //
    // Example: "example.net/import-controller".
    //
    // The format of this field is DOMAIN "/" PATH, where DOMAIN and PATH are valid Kubernetes
    // names (https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#names).
    //
    // A controller MUST populate this field when writing status and ensure that entries to status
    // populated with their controller name are removed when they are no longer necessary.
    //
    // +required
    Name ControllerName `json:"name"`

    // ExportingClusters is a list of clusters that exported the InferencePool associated with this
    // InferencePoolImport. Required when the controller is responsible for CRUD'ing the InferencePoolImport
    // from the exported InferencePool(s).
    //
    // +optional
    ExportingClusters []ExportingCluster `json:"exportingClusters"`

    // Parents is a list of parent resources, typically Gateways, that are associated with the
    // InferencePoolImport, and the status of the InferencePoolImport with respect to each parent.
    //
    // Required when the controller manages the InferencePoolImport as an HTTPRoute backendRef. The controller
    // must add an entry for each parent it manages and remove the parent entry when the controller no longer
    // considers the InferencePoolImport to be associated with that parent.
    //
    // +optional
    // +listType=atomic
    Parents []v1.ParentStatus `json:"parents,omitempty"`

    // Conditions track the state of the InferencePoolImport.
    //
    // Known condition types are:
    //
    //  * "Accepted"
    //
    // +optional
    // +listType=map
    // +listMapKey=type
    // +kubebuilder:validation:MaxItems=8
    Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// ControllerName is the name of a controller that manages a resource. It must be a domain prefixed path.
//
// Valid values include:
//
//  * "example.com/bar"
//
// Invalid values include:
//
//  * "example.com" - must include path
//  * "foo.example.com" - must include path
//
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:MaxLength=253
// +kubebuilder:validation:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*\/[A-Za-z0-9\/\-._~%!$&'()*+,;=:]+$`
type ControllerName string

// ExportingCluster defines a cluster that exported the InferencePool associated to this InferencePoolImport.
type ExportingCluster struct {
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

EPP failure modes continue to work as-is and are independent of MCIP.

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

#### EPP Discovery

- Should EPP Deployment/Pod discovery be standardized (labels/port names) for health/metrics auto-discovery?

#### Security

- Provide a standard way to bootstrap mTLS between importing IG and exported EPP/parents, e.g. use BackendTLSPolicy?

#### Ownership and Lifecycle

- Garbage collection when export is withdrawn (delete import?) and how to drain traffic safely. See [this comment](https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/1374#discussion_r2357523781) for additional context.

### Prior Art

- [GEP-1748: Gateway API Interaction with Multi-Cluster Services](https://gateway-api.sigs.k8s.io/geps/gep-1748/)
- [Envoy Gateway with Multi-Cluster Services](https://gateway.envoyproxy.io/latest/tasks/traffic/multicluster-service/)
- [Multi-Cluster Service API](https://multicluster.sigs.k8s.io/concepts/multicluster-services-api/)

### References

- [Initial Multi-Cluster Inference Design Doc](https://docs.google.com/document/d/1QGvG9ToaJ72vlCBdJe--hmrmLtgOV_ptJi9D58QMD2w/edit?tab=t.0#heading=h.q6xiq2fzcaia)

### Notes for reviewers

- The InferencePoolImport CRD is intentionally status-only to keep the UX simple and controller-driven.
- The InferencePool namespace sameness simplifies identity and lets HTTPRoute authors reference imports without new indirection.
