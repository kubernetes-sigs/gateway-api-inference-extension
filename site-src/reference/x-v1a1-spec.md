# API Reference

## Packages
- [inference.networking.x-k8s.io/v1alpha1](#inferencenetworkingx-k8siov1alpha1)


## inference.networking.x-k8s.io/v1alpha1

Package v1alpha1 contains API Schema definitions for the
inference.networking.x-k8s.io API group.


### Resource Types
- [InferencePoolImport](#inferencepoolimport)



#### ClusterName

_Underlying type:_ _string_

ClusterName is the name of a cluster that exported the InferencePool.

_Validation:_
- MaxLength: 253
- MinLength: 1

_Appears in:_
- [ExportingCluster](#exportingcluster)



#### ControllerName

_Underlying type:_ _string_

ControllerName is the name of a controller that manages a resource. It must be a domain prefixed path.

Valid values include:

  - "example.com/bar"

Invalid values include:

  - "example.com" - must include path
  - "foo.example.com" - must include path

_Validation:_
- MaxLength: 253
- MinLength: 1
- Pattern: `^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*\/[A-Za-z0-9\/\-._~%!$&'()*+,;=:]+$`

_Appears in:_
- [ImportController](#importcontroller)



#### ExportingCluster



ExportingCluster defines a cluster that exported the InferencePool that backs this InferencePoolImport.



_Appears in:_
- [ImportController](#importcontroller)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `name` _[ClusterName](#clustername)_ | Name of the exporting cluster (must be unique within the list). |  | MaxLength: 253 <br />MinLength: 1 <br />Required: \{\} <br /> |


#### ImportController



ImportController defines a controller that is responsible for managing the InferencePoolImport.



_Appears in:_
- [InferencePoolImportStatus](#inferencepoolimportstatus)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `name` _[ControllerName](#controllername)_ | Name is a domain/path string that indicates the name of the controller that manages the<br />InferencePoolImport. Name corresponds to the GatewayClass controllerName field when the<br />controller will manage parents of type "Gateway". Otherwise, the name is implementation-specific.<br />Example: "example.net/import-controller".<br />The format of this field is DOMAIN "/" PATH, where DOMAIN and PATH are valid Kubernetes<br />names (https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#names).<br />A controller MUST populate this field when writing status and ensure that entries to status<br />populated with their controller name are removed when they are no longer necessary. |  | MaxLength: 253 <br />MinLength: 1 <br />Pattern: `^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*\/[A-Za-z0-9\/\-._~%!$&'()*+,;=:]+$` <br /> |
| `exportingClusters` _[ExportingCluster](#exportingcluster) array_ | ExportingClusters is a list of clusters that exported the InferencePool(s) that back the<br />InferencePoolImport. Required when the controller is responsible for CRUD'ing the InferencePoolImport<br />from the exported InferencePool(s). |  |  |
| `parents` _ParentStatus array_ | Parents is a list of parent resources, typically Gateways, that are associated with the<br />InferencePoolImport, and the status of the InferencePoolImport with respect to each parent.<br />Ancestor would be a more accurate name, but Parent is consistent with InferencePool terminology.<br />Required when the controller manages the InferencePoolImport as an HTTPRoute backendRef. The controller<br />must add an entry for each parent it manages and remove the parent entry when the controller no longer<br />considers the InferencePoolImport to be associated with that parent. |  |  |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#condition-v1-meta) array_ | Conditions track the state of the InferencePoolImport.<br />Known condition types are:<br /> * "Accepted" |  | MaxItems: 8 <br /> |


#### InferencePoolImport



InferencePoolImport is the Schema for the InferencePoolImports API.





| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `inference.networking.x-k8s.io/v1alpha1` | | |
| `kind` _string_ | `InferencePoolImport` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `status` _[InferencePoolImportStatus](#inferencepoolimportstatus)_ | Status defines the observed state of the InferencePoolImport. |  |  |


#### InferencePoolImportStatus



InferencePoolImportStatus defines the observed state of the InferencePoolImport.



_Appears in:_
- [InferencePoolImport](#inferencepoolimport)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `controllers` _[ImportController](#importcontroller) array_ | Controllers is a list of controllers that are responsible for managing the InferencePoolImport. |  | MaxItems: 8 <br />Required: \{\} <br /> |


