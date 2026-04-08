# API Reference

## Packages
- [inference.networking.x-k8s.io/v1alpha2](#inferencenetworkingx-k8siov1alpha2)


## inference.networking.x-k8s.io/v1alpha2

Package v1alpha2 contains API Schema definitions for the
inference.networking.x-k8s.io API group.


### Resource Types
- [InferenceModelRewrite](#inferencemodelrewrite)
- [InferenceObjective](#inferenceobjective)



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
- [PoolObjectReference](#poolobjectreference)



#### InferenceModelRewrite



InferenceModelRewrite is the Schema for the InferenceModelRewrite API.





| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `inference.networking.x-k8s.io/v1alpha2` | | |
| `kind` _string_ | `InferenceModelRewrite` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `spec` _[InferenceModelRewriteSpec](#inferencemodelrewritespec)_ |  |  |  |
| `status` _[InferenceModelRewriteStatus](#inferencemodelrewritestatus)_ |  |  |  |






#### InferenceModelRewriteRule



InferenceModelRewriteRule defines the match criteria and corresponding action.
For details on how precedence is determined across multiple rules and
InferenceModelRewrite resources, see the "Precedence and Conflict Resolution"
section in InferenceModelRewriteSpec.



_Appears in:_
- [InferenceModelRewriteSpec](#inferencemodelrewritespec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `matches` _[Match](#match) array_ |  |  |  |
| `targets` _[TargetModel](#targetmodel) array_ |  |  | MinItems: 1 <br /> |


#### InferenceModelRewriteSpec



InferenceModelRewriteSpec defines the desired state of InferenceModelRewrite.



_Appears in:_
- [InferenceModelRewrite](#inferencemodelrewrite)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `poolRef` _[PoolObjectReference](#poolobjectreference)_ | PoolRef is a reference to the inference pool. |  | Required: \{\} <br /> |
| `rules` _[InferenceModelRewriteRule](#inferencemodelrewriterule) array_ |  |  |  |


#### InferenceModelRewriteStatus



InferenceModelRewriteStatus defines the observed state of InferenceModelRewrite.



_Appears in:_
- [InferenceModelRewrite](#inferencemodelrewrite)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#condition-v1-meta) array_ | Conditions track the state of the InferenceModelRewrite.<br />Known condition types are:<br />* "Accepted" | [map[lastTransitionTime:1970-01-01T00:00:00Z message:Waiting for controller reason:Pending status:Unknown type:Accepted]] | MaxItems: 8 <br /> |


#### InferenceObjective



InferenceObjective is the Schema for the InferenceObjectives API.





| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `inference.networking.x-k8s.io/v1alpha2` | | |
| `kind` _string_ | `InferenceObjective` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `spec` _[InferenceObjectiveSpec](#inferenceobjectivespec)_ |  |  |  |
| `status` _[InferenceObjectiveStatus](#inferenceobjectivestatus)_ |  |  |  |






#### InferenceObjectiveSpec



InferenceObjectiveSpec represents the desired state of a specific model use case. This resource is
managed by the "Inference Workload Owner" persona.

The Inference Workload Owner persona is someone that trains, verifies, and
leverages a large language model from a model frontend, drives the lifecycle
and rollout of new versions of those models, and defines the specific
performance and latency goals for the model. These workloads are
expected to operate within an InferencePool sharing compute capacity with other
InferenceObjectives, defined by the Inference Platform Admin.



_Appears in:_
- [InferenceObjective](#inferenceobjective)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `priority` _integer_ | Priority defines how important it is to serve the request compared to other requests in the same pool.<br />Priority is an integer value that defines the priority of the request.<br />The higher the value, the more critical the request is; negative values _are_ allowed.<br />No default value is set for this field, allowing for future additions of new fields that may 'one of' with this field.<br />However, implementations that consume this field (such as the Endpoint Picker) will treat an unset value as '0'.<br />Priority is used in flow control, primarily in the event of resource scarcity(requests need to be queued).<br />All requests will be queued, and flow control will _always_ allow requests of higher priority to be served first.<br />Fairness is only enforced and tracked between requests of the same priority.<br />Example: requests with Priority 10 will always be served before<br />requests with Priority of 0 (the value used if Priority is unset or no InferenceObjective is specified).<br />Similarly requests with a Priority of -10 will always be served after requests with Priority of 0. |  |  |
| `poolRef` _[PoolObjectReference](#poolobjectreference)_ | PoolRef is a reference to the inference pool, the pool must exist in the same namespace. |  | Required: \{\} <br /> |


#### InferenceObjectiveStatus



InferenceObjectiveStatus defines the observed state of InferenceObjective



_Appears in:_
- [InferenceObjective](#inferenceobjective)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#condition-v1-meta) array_ | Conditions track the state of the InferenceObjective.<br />Known condition types are:<br />* "Accepted" | [map[lastTransitionTime:1970-01-01T00:00:00Z message:Waiting for controller reason:Pending status:Unknown type:Ready]] | MaxItems: 8 <br /> |


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
- [PoolObjectReference](#poolobjectreference)







#### Match



Match defines the criteria for matching the LLM requests.



_Appears in:_
- [InferenceModelRewriteRule](#inferencemodelrewriterule)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `model` _[ModelMatch](#modelmatch)_ | Model specifies the criteria for matching the 'model' field<br />within the JSON request body. |  |  |


#### MatchValidationType

_Underlying type:_ _string_

MatchValidationType specifies the type of string matching to use.

_Validation:_
- Enum: [Exact]

_Appears in:_
- [ModelMatch](#modelmatch)

| Field | Description |
| --- | --- |
| `Exact` | MatchExact indicates that the model name must match exactly.<br /> |


#### ModelMatch



ModelMatch defines how to match against the model name in the request body.



_Appears in:_
- [Match](#match)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `type` _[MatchValidationType](#matchvalidationtype)_ | Type specifies the kind of string matching to use.<br />Supported value is "Exact". Defaults to "Exact". | Exact | Enum: [Exact] <br /> |
| `value` _string_ | Value is the model name string to match against. |  | MinLength: 1 <br /> |




#### ObjectName

_Underlying type:_ _string_

ObjectName refers to the name of a Kubernetes object.
Object names can have a variety of forms, including RFC 1123 subdomains,
RFC 1123 labels, or RFC 1035 labels.

_Validation:_
- MaxLength: 253
- MinLength: 1

_Appears in:_
- [PoolObjectReference](#poolobjectreference)



#### PoolObjectReference



PoolObjectReference identifies an API object within the namespace of the
referrer.



_Appears in:_
- [InferenceModelRewriteSpec](#inferencemodelrewritespec)
- [InferenceObjectiveSpec](#inferenceobjectivespec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `group` _[Group](#group)_ | Group is the group of the referent. | inference.networking.k8s.io | MaxLength: 253 <br />Pattern: `^$\|^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$` <br /> |
| `kind` _[Kind](#kind)_ | Kind is kind of the referent. For example "InferencePool". | InferencePool | MaxLength: 63 <br />MinLength: 1 <br />Pattern: `^[a-zA-Z]([-a-zA-Z0-9]*[a-zA-Z0-9])?$` <br /> |
| `name` _[ObjectName](#objectname)_ | Name is the name of the referent. |  | MaxLength: 253 <br />MinLength: 1 <br />Required: \{\} <br /> |




#### TargetModel



TargetModel defines a weighted model destination for traffic distribution.



_Appears in:_
- [InferenceModelRewriteRule](#inferencemodelrewriterule)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `weight` _integer_ | (The following comment is copied from the original targetModel)<br />Weight is used to determine the proportion of traffic that should be<br />sent to this model when multiple target models are specified.<br />Weight defines the proportion of requests forwarded to the specified<br />model. This is computed as weight/(sum of all weights in this<br />TargetModels list). For non-zero values, there may be some epsilon from<br />the exact proportion defined here depending on the precision an<br />implementation supports. Weight is not a percentage and the sum of<br />weights does not need to equal 100.<br />If a weight is set for any targetModel, it must be set for all targetModels.<br />Conversely weights are optional, so long as ALL targetModels do not specify a weight. |  | Maximum: 1e+06 <br />Minimum: 1 <br /> |
| `modelRewrite` _string_ |  |  |  |


