# API Reference

## Packages
- [inference.networking.x-k8s.io/v1alpha2](#inferencenetworkingx-k8siov1alpha2)


## inference.networking.x-k8s.io/v1alpha2

Package v1alpha2 contains API Schema definitions for the
inference.networking.x-k8s.io API group.


### Resource Types
- [InferenceModel](#inferencemodel)
- [InferencePool](#inferencepool)



#### Criticality

_Underlying type:_ _string_

Criticality defines how important it is to serve the model compared to other models.
Criticality is intentionally a bounded enum to contain the possibilities that need to be supported by the load balancing algorithm. Any reference to the Criticality field must be optional(use a pointer), and set no default.
This allows us to union this with a oneOf field in the future should we wish to adjust/extend this behavior.

_Validation:_
- Enum: [Critical Standard Sheddable]

_Appears in:_
- [InferenceModelSpec](#inferencemodelspec)

| Field | Description |
| --- | --- |
| `Critical` | Critical defines the highest level of criticality. Requests to this band will be shed last.<br /> |
| `Standard` | Standard defines the base criticality level and is more important than Sheddable but less<br />important than Critical. Requests in this band will be shed before critical traffic.<br />Most models are expected to fall within this band.<br /> |
| `Sheddable` | Sheddable defines the lowest level of criticality. Requests to this band will be shed before<br />all other bands.<br /> |


#### EndpointPickerConfig



EndpointPickerConfig specifies the configuration needed by the proxy to discover and connect to the endpoint picker extension.
This type is intended to be a union of mutually exclusive configuration options that we may add in the future.



_Appears in:_
- [InferencePoolSpec](#inferencepoolspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `extensionRef` _[Extension](#extension)_ | Extension configures an endpoint picker as an extension service. |  | Required: \{\} <br /> |


#### Extension



Extension specifies how to configure an extension that runs the endpoint picker.



_Appears in:_
- [EndpointPickerConfig](#endpointpickerconfig)
- [InferencePoolSpec](#inferencepoolspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `group` _[Group](#group)_ | Group is the group of the referent.<br />The default value is "", representing the Core API group. |  | MaxLength: 253 <br />Pattern: `^$\|^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$` <br /> |
| `kind` _[Kind](#kind)_ | Kind is the Kubernetes resource kind of the referent. For example<br />"Service".<br />Defaults to "Service" when not specified.<br />ExternalName services can refer to CNAME DNS records that may live<br />outside of the cluster and as such are difficult to reason about in<br />terms of conformance. They also may not be safe to forward to (see<br />CVE-2021-25740 for more information). Implementations MUST NOT<br />support ExternalName Services. | Service | MaxLength: 63 <br />MinLength: 1 <br />Pattern: `^[a-zA-Z]([-a-zA-Z0-9]*[a-zA-Z0-9])?$` <br /> |
| `name` _[ObjectName](#objectname)_ | Name is the name of the referent. |  | MaxLength: 253 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `portNumber` _[PortNumber](#portnumber)_ | The port number on the service running the extension. When unspecified,<br />implementations SHOULD infer a default value of 9002 when the Kind is<br />Service. |  | Maximum: 65535 <br />Minimum: 1 <br /> |
| `failureMode` _[ExtensionFailureMode](#extensionfailuremode)_ | Configures how the gateway handles the case when the extension is not responsive.<br />Defaults to failClose. | FailClose | Enum: [FailOpen FailClose] <br /> |


#### ExtensionConnection



ExtensionConnection encapsulates options that configures the connection to the extension.



_Appears in:_
- [Extension](#extension)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `failureMode` _[ExtensionFailureMode](#extensionfailuremode)_ | Configures how the gateway handles the case when the extension is not responsive.<br />Defaults to failClose. | FailClose | Enum: [FailOpen FailClose] <br /> |


#### ExtensionFailureMode

_Underlying type:_ _string_

ExtensionFailureMode defines the options for how the gateway handles the case when the extension is not
responsive.

_Validation:_
- Enum: [FailOpen FailClose]

_Appears in:_
- [Extension](#extension)
- [ExtensionConnection](#extensionconnection)

| Field | Description |
| --- | --- |
| `FailOpen` | FailOpen specifies that the proxy should not drop the request and forward the request to and endpoint of its picking.<br /> |
| `FailClose` | FailClose specifies that the proxy should drop the request.<br /> |


#### ExtensionReference



ExtensionReference is a reference to the extension deployment.



_Appears in:_
- [Extension](#extension)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `group` _[Group](#group)_ | Group is the group of the referent.<br />The default value is "", representing the Core API group. |  | MaxLength: 253 <br />Pattern: `^$\|^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$` <br /> |
| `kind` _[Kind](#kind)_ | Kind is the Kubernetes resource kind of the referent. For example<br />"Service".<br />Defaults to "Service" when not specified.<br />ExternalName services can refer to CNAME DNS records that may live<br />outside of the cluster and as such are difficult to reason about in<br />terms of conformance. They also may not be safe to forward to (see<br />CVE-2021-25740 for more information). Implementations MUST NOT<br />support ExternalName Services. | Service | MaxLength: 63 <br />MinLength: 1 <br />Pattern: `^[a-zA-Z]([-a-zA-Z0-9]*[a-zA-Z0-9])?$` <br /> |
| `name` _[ObjectName](#objectname)_ | Name is the name of the referent. |  | MaxLength: 253 <br />MinLength: 1 <br />Required: \{\} <br /> |
| `portNumber` _[PortNumber](#portnumber)_ | The port number on the service running the extension. When unspecified,<br />implementations SHOULD infer a default value of 9002 when the Kind is<br />Service. |  | Maximum: 65535 <br />Minimum: 1 <br /> |


#### Group

_Underlying type:_ _string_

Group refers to a Kubernetes Group. It must either be an empty string or a
RFC 1123 subdomain.

This validation is based off of the corresponding Kubernetes validation:
https://github.com/kubernetes/apimachinery/blob/02cfb53916346d085a6c6c7c66f882e3c6b0eca6/pkg/util/validation/validation.go#L208

Valid values include:

* "" - empty string implies core Kubernetes API group
* "gateway.networking.k8s.io"
* "foo.example.com"

Invalid values include:

* "example.com/bar" - "/" is an invalid character

_Validation:_
- MaxLength: 253
- Pattern: `^$|^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$`

_Appears in:_
- [Extension](#extension)
- [ExtensionReference](#extensionreference)
- [PoolObjectReference](#poolobjectreference)



#### InferenceModel



InferenceModel is the Schema for the InferenceModels API.





| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `inference.networking.x-k8s.io/v1alpha2` | | |
| `kind` _string_ | `InferenceModel` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `spec` _[InferenceModelSpec](#inferencemodelspec)_ |  |  |  |
| `status` _[InferenceModelStatus](#inferencemodelstatus)_ |  |  |  |






#### InferenceModelSpec



InferenceModelSpec represents the desired state of a specific model use case. This resource is
managed by the "Inference Workload Owner" persona.

The Inference Workload Owner persona is someone that trains, verifies, and
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
| `modelName` _string_ | ModelName is the name of the model as it will be set in the "model" parameter for an incoming request.<br />ModelNames must be unique for a referencing InferencePool<br />(names can be reused for a different pool in the same cluster).<br />The modelName with the oldest creation timestamp is retained, and the incoming<br />InferenceModel is sets the Ready status to false with a corresponding reason.<br />In the rare case of a race condition, one Model will be selected randomly to be considered valid, and the other rejected.<br />Names can be reserved without an underlying model configured in the pool.<br />This can be done by specifying a target model and setting the weight to zero,<br />an error will be returned specifying that no valid target model is found. |  | MaxLength: 256 <br />Required: \{\} <br /> |
| `criticality` _[Criticality](#criticality)_ | Criticality defines how important it is to serve the model compared to other models referencing the same pool.<br />Criticality impacts how traffic is handled in resource constrained situations. It handles this by<br />queuing or rejecting requests of lower criticality. InferenceModels of an equivalent Criticality will<br />fairly share resources over throughput of tokens. In the future, the metric used to calculate fairness,<br />and the proportionality of fairness will be configurable.<br />Default values for this field will not be set, to allow for future additions of new field that may 'one of' with this field.<br />Any implementations that may consume this field may treat an unset value as the 'Standard' range. |  | Enum: [Critical Standard Sheddable] <br /> |
| `targetModels` _[TargetModel](#targetmodel) array_ | TargetModels allow multiple versions of a model for traffic splitting.<br />If not specified, the target model name is defaulted to the modelName parameter.<br />modelName is often in reference to a LoRA adapter. |  | MaxItems: 10 <br /> |
| `poolRef` _[PoolObjectReference](#poolobjectreference)_ | PoolRef is a reference to the inference pool, the pool must exist in the same namespace. |  | Required: \{\} <br /> |


#### InferenceModelStatus



InferenceModelStatus defines the observed state of InferenceModel



_Appears in:_
- [InferenceModel](#inferencemodel)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#condition-v1-meta) array_ | Conditions track the state of the InferenceModel.<br />Known condition types are:<br />* "Accepted" | [map[lastTransitionTime:1970-01-01T00:00:00Z message:Waiting for controller reason:Pending status:Unknown type:Ready]] | MaxItems: 8 <br /> |


#### InferencePool



InferencePool is the Schema for the InferencePools API.





| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `inference.networking.x-k8s.io/v1alpha2` | | |
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
| `selector` _object (keys:[LabelKey](#labelkey), values:[LabelValue](#labelvalue))_ | Selector defines a map of labels to watch model server pods<br />that should be included in the InferencePool.<br />In some cases, implementations may translate this field to a Service selector, so this matches the simple<br />map used for Service selectors instead of the full Kubernetes LabelSelector type.<br />If sepecified, it will be applied to match the model server pods in the same namespace as the InferencePool.<br />Cross namesoace selector is not supported. |  | Required: \{\} <br /> |
| `targetPortNumber` _integer_ | TargetPortNumber defines the port number to access the selected model servers.<br />The number must be in the range 1 to 65535. |  | Maximum: 65535 <br />Minimum: 1 <br />Required: \{\} <br /> |
| `extensionRef` _[Extension](#extension)_ | Extension configures an endpoint picker as an extension service. |  | Required: \{\} <br /> |


#### InferencePoolStatus



InferencePoolStatus defines the observed state of InferencePool



_Appears in:_
- [InferencePool](#inferencepool)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `parent` _[PoolStatus](#poolstatus) array_ | Parents is a list of parent resources (usually Gateways) that are<br />associated with the route, and the status of the InferencePool with respect to<br />each parent.<br />A maximum of 32 Gateways will be represented in this list. An empty list<br />means the route has not been attached to any Gateway. |  | MaxItems: 32 <br /> |


#### Kind

_Underlying type:_ _string_

Kind refers to a Kubernetes Kind.

Valid values include:

* "Service"
* "HTTPRoute"

Invalid values include:

* "invalid/kind" - "/" is an invalid character

_Validation:_
- MaxLength: 63
- MinLength: 1
- Pattern: `^[a-zA-Z]([-a-zA-Z0-9]*[a-zA-Z0-9])?$`

_Appears in:_
- [Extension](#extension)
- [ExtensionReference](#extensionreference)
- [PoolObjectReference](#poolobjectreference)



#### LabelKey

_Underlying type:_ _string_

LabelKey was originally copied from: https://github.com/kubernetes-sigs/gateway-api/blob/99a3934c6bc1ce0874f3a4c5f20cafd8977ffcb4/apis/v1/shared_types.go#L694-L731
Duplicated as to not take an unexpected dependency on gw's API.

LabelKey is the key of a label. This is used for validation
of maps. This matches the Kubernetes "qualified name" validation that is used for labels.
Labels are case sensitive, so: my-label and My-Label are considered distinct.

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



#### ObjectName

_Underlying type:_ _string_

ObjectName refers to the name of a Kubernetes object.
Object names can have a variety of forms, including RFC 1123 subdomains,
RFC 1123 labels, or RFC 1035 labels.

_Validation:_
- MaxLength: 253
- MinLength: 1

_Appears in:_
- [Extension](#extension)
- [ExtensionReference](#extensionreference)
- [PoolObjectReference](#poolobjectreference)



#### PoolObjectReference



PoolObjectReference identifies an API object within the namespace of the
referrer.



_Appears in:_
- [InferenceModelSpec](#inferencemodelspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `group` _[Group](#group)_ | Group is the group of the referent. | inference.networking.x-k8s.io | MaxLength: 253 <br />Pattern: `^$\|^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$` <br /> |
| `kind` _[Kind](#kind)_ | Kind is kind of the referent. For example "InferencePool". | InferencePool | MaxLength: 63 <br />MinLength: 1 <br />Pattern: `^[a-zA-Z]([-a-zA-Z0-9]*[a-zA-Z0-9])?$` <br /> |
| `name` _[ObjectName](#objectname)_ | Name is the name of the referent. |  | MaxLength: 253 <br />MinLength: 1 <br />Required: \{\} <br /> |


#### PoolStatus



PoolStatus defines the observed state of InferencePool from a Gateway.



_Appears in:_
- [InferencePoolStatus](#inferencepoolstatus)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `parentRef` _[ObjectReference](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#objectreference-v1-core)_ | GatewayRef indicates the gateway that observed state of InferencePool. |  |  |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#condition-v1-meta) array_ | Conditions track the state of the InferencePool.<br />Known condition types are:<br />* "Accepted"<br />* "ResolvedRefs" | [map[lastTransitionTime:1970-01-01T00:00:00Z message:Waiting for controller reason:Pending status:Unknown type:Accepted]] | MaxItems: 8 <br /> |


#### PortNumber

_Underlying type:_ _integer_

PortNumber defines a network port.

_Validation:_
- Maximum: 65535
- Minimum: 1

_Appears in:_
- [Extension](#extension)
- [ExtensionReference](#extensionreference)



#### TargetModel



TargetModel represents a deployed model or a LoRA adapter. The
Name field is expected to match the name of the LoRA adapter
(or base model) as it is registered within the model server. Inference
Gateway assumes that the model exists on the model server and it's the
responsibility of the user to validate a correct match. Should a model fail
to exist at request time, the error is processed by the Inference Gateway
and emitted on the appropriate InferenceModel object.



_Appears in:_
- [InferenceModelSpec](#inferencemodelspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `name` _string_ | Name is the name of the adapter or base model, as expected by the ModelServer. |  | MaxLength: 253 <br />Required: \{\} <br /> |
| `weight` _integer_ | Weight is used to determine the proportion of traffic that should be<br />sent to this model when multiple target models are specified.<br />Weight defines the proportion of requests forwarded to the specified<br />model. This is computed as weight/(sum of all weights in this<br />TargetModels list). For non-zero values, there may be some epsilon from<br />the exact proportion defined here depending on the precision an<br />implementation supports. Weight is not a percentage and the sum of<br />weights does not need to equal 100.<br />If a weight is set for any targetModel, it must be set for all targetModels.<br />Conversely weights are optional, so long as ALL targetModels do not specify a weight. |  | Maximum: 1e+06 <br />Minimum: 1 <br /> |


