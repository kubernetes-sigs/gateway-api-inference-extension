# API Reference

## Packages
- [inference.networking.x-k8s.io/v1alpha1](#inferencenetworkingx-k8siov1alpha1)


## inference.networking.x-k8s.io/v1alpha1

Package v1alpha1 contains API Schema definitions for the gateway v1alpha1 API group

### Resource Types
- [InferenceModel](#inferencemodel)
- [InferencePool](#inferencepool)



#### Criticality

_Underlying type:_ _string_

Defines how important it is to serve the model compared to other models.

_Validation:_
- Enum: [Critical Default Sheddable]

_Appears in:_
- [InferenceModelSpec](#inferencemodelspec)

| Field | Description |
| --- | --- |
| `Critical` | Most important. Requests to this band will be shed last.<br /> |
| `Default` | More important than Sheddable, less important than Critical.<br />Requests in this band will be shed before critical traffic.<br />+kubebuilder:default=Default<br /> |
| `Sheddable` | Least important. Requests to this band will be shed before all other bands.<br /> |


#### InferenceModel



InferenceModel is the Schema for the InferenceModels API





| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `inference.networking.x-k8s.io/v1alpha1` | | |
| `kind` _string_ | `InferenceModel` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `spec` _[InferenceModelSpec](#inferencemodelspec)_ |  |  |  |
| `status` _[InferenceModelStatus](#inferencemodelstatus)_ |  |  |  |


#### InferenceModelSpec



InferenceModelSpec represents a specific model use case. This resource is
managed by the "Inference Workload Owner" persona.


The Inference Workload Owner persona is: a team that trains, verifies, and
leverages a large language model from a model frontend, drives the lifecycle
and rollout of new versions of those models, and defines the specific
performance and latency goals for the model. These workloads are
expected to operate within an InferencePool sharing compute capacity with other
InferenceModels, defined by the Inference Platform Admin.


InferenceModel's modelName (not the ObjectMeta name) is unique for a given InferencePool,
if the name is reused, an error will be shown on the status of a
InferenceModel that attempted to reuse. The oldest InferenceModel, based on
creation timestamp, will be selected to remain valid. In the event of a race
condition, one will be selected at random.



_Appears in:_
- [InferenceModel](#inferencemodel)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `modelName` _string_ | The name of the model as the users set in the "model" parameter in the requests.<br />The name should be unique among the workloads that reference the same backend pool.<br />This is the parameter that will be used to match the request with. In the future, we may<br />allow to match on other request parameters. The other approach to support matching on<br />on other request parameters is to use a different ModelName per HTTPFilter.<br />Names can be reserved without implementing an actual model in the pool.<br />This can be done by specifying a target model and setting the weight to zero,<br />an error will be returned specifying that no valid target model is found. |  | MaxLength: 253 <br /> |
| `criticality` _[Criticality](#criticality)_ | Defines how important it is to serve the model compared to other models referencing the same pool. | Default | Enum: [Critical Default Sheddable] <br /> |
| `targetModels` _[TargetModel](#targetmodel) array_ | Allow multiple versions of a model for traffic splitting.<br />If not specified, the target model name is defaulted to the modelName parameter.<br />modelName is often in reference to a LoRA adapter. |  | MaxItems: 10 <br /> |
| `poolRef` _[PoolObjectReference](#poolobjectreference)_ | Reference to the inference pool, the pool must exist in the same namespace. |  | Required: \{\} <br /> |


#### InferenceModelStatus



InferenceModelStatus defines the observed state of InferenceModel



_Appears in:_
- [InferenceModel](#inferencemodel)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#condition-v1-meta) array_ | Conditions track the state of the InferencePool. |  |  |


#### InferencePool



InferencePool is the Schema for the Inferencepools API





| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `inference.networking.x-k8s.io/v1alpha1` | | |
| `kind` _string_ | `InferencePool` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `spec` _[InferencePoolSpec](#inferencepoolspec)_ |  |  |  |
| `status` _[InferencePoolStatus](#inferencepoolstatus)_ |  |  |  |


#### InferencePoolSpec



InferencePoolSpec defines the desired state of InferencePool



_Appears in:_
- [InferencePool](#inferencepool)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `selector` _object (keys:[LabelKey](#labelkey), values:[LabelValue](#labelvalue))_ | Selector uses a map of label to watch model server pods<br />that should be included in the InferencePool. ModelServers should not<br />be with any other Service or InferencePool, that behavior is not supported<br />and will result in sub-optimal utilization.<br />In some cases, implementations may translate this to a Service selector, so this matches the simple<br />map used for Service selectors instead of the full Kubernetes LabelSelector type. |  | Required: \{\} <br /> |
| `targetPortNumber` _integer_ | TargetPortNumber is the port number that the model servers within the pool expect<br />to receive traffic from.<br />This maps to the TargetPort in: https://pkg.go.dev/k8s.io/api/core/v1#ServicePort |  | Maximum: 65535 <br />Minimum: 0 <br />Required: \{\} <br /> |


#### InferencePoolStatus



InferencePoolStatus defines the observed state of InferencePool



_Appears in:_
- [InferencePool](#inferencepool)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#condition-v1-meta) array_ | Conditions track the state of the InferencePool. |  |  |


#### LabelKey

_Underlying type:_ _string_

Originally copied from: https://github.com/kubernetes-sigs/gateway-api/blob/99a3934c6bc1ce0874f3a4c5f20cafd8977ffcb4/apis/v1/shared_types.go#L694-L731
Duplicated as to not take an unexpected dependency on gw's API.


LabelKey is the key of a label. This is used for validation
of maps. This matches the Kubernetes "qualified name" validation that is used for labels.


Valid values include:


* example
* example.com
* example.com/path
* example.com/path.html


Invalid values include:


* example~ - "~" is an invalid character
* example.com. - can not start or end with "."

_Validation:_
- MaxLength: 253
- MinLength: 1
- Pattern: `^([a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*/)?([A-Za-z0-9][-A-Za-z0-9_.]{0,61})?[A-Za-z0-9]$`

_Appears in:_
- [InferencePoolSpec](#inferencepoolspec)



#### LabelValue

_Underlying type:_ _string_

LabelValue is the value of a label. This is used for validation
of maps. This matches the Kubernetes label validation rules:
* must be 63 characters or less (can be empty),
* unless empty, must begin and end with an alphanumeric character ([a-z0-9A-Z]),
* could contain dashes (-), underscores (_), dots (.), and alphanumerics between.


Valid values include:


* MyValue
* my.name
* 123-my-value

_Validation:_
- MaxLength: 63
- MinLength: 0
- Pattern: `^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$`

_Appears in:_
- [InferencePoolSpec](#inferencepoolspec)



#### PoolObjectReference



PoolObjectReference identifies an API object within the namespace of the
referrer.



_Appears in:_
- [InferenceModelSpec](#inferencemodelspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `group` _string_ | Group is the group of the referent. | inference.networking.x-k8s.io | MaxLength: 253 <br />Pattern: `^$\|^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$` <br /> |
| `kind` _string_ | Kind is kind of the referent. For example "InferencePool". | InferencePool | MaxLength: 63 <br />MinLength: 1 <br />Pattern: `^[a-zA-Z]([-a-zA-Z0-9]*[a-zA-Z0-9])?$` <br /> |
| `name` _string_ | Name is the name of the referent. |  | MaxLength: 253 <br />MinLength: 1 <br />Required: \{\} <br /> |


#### TargetModel



TargetModel represents a deployed model or a LoRA adapter. The
Name field is expected to match the name of the LoRA adapter
(or base model) as it is registered within the model server. Inference
Gateway assumes that the model exists on the model server and is the
responsibility of the user to validate a correct match. Should a model fail
to exist at request time, the error is processed by the Instance Gateway,
and then emitted on the appropriate InferenceModel object.



_Appears in:_
- [InferenceModelSpec](#inferencemodelspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `name` _string_ | The name of the adapter as expected by the ModelServer. |  | MaxLength: 253 <br /> |
| `weight` _integer_ | Weight is used to determine the proportion of traffic that should be<br />sent to this target model when multiple versions of the model are specified. | 1 | Maximum: 1e+06 <br />Minimum: 0 <br /> |


