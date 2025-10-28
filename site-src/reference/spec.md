# API Reference

## Packages
- [inference.networking.k8s.io/v1](#inferencenetworkingk8siov1)


## inference.networking.k8s.io/v1

Package v1 contains API Schema definitions for the
inference.networking.k8s.io API group.


### Resource Types
- [InferencePool](#inferencepool)



#### ControllerName

_Underlying type:_ _string_

ControllerName is the name of a controller that manages ParentStatus. It must be a domain prefixed
path.

Valid values include:

* "example.com/bar"

Invalid values include:

* "example.com" - must include path
* "foo.example.com" - must include path

_Validation:_
- MaxLength: 253
- MinLength: 1
- Pattern: `^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*\/[A-Za-z0-9\/\-._~%!$&'()*+,;=:]+$`

_Appears in:_
- [ParentStatus](#parentstatus)



#### EndpointPickerFailureMode

_Underlying type:_ _string_

EndpointPickerFailureMode defines the options for how the parent handles the case when the
Endpoint Picker extension is non-responsive.

_Validation:_
- Enum: [FailOpen FailClose]

_Appears in:_
- [EndpointPickerRef](#endpointpickerref)

| Field | Description |
| --- | --- |
| `FailOpen` | EndpointPickerFailOpen specifies that the parent should forward the request to an endpoint<br />of its picking when the Endpoint Picker extension fails.<br /> |
| `FailClose` | EndpointPickerFailClose specifies that the parent should drop the request when the Endpoint<br />Picker extension fails.<br /> |


#### EndpointPickerRef



EndpointPickerRef specifies a reference to an Endpoint Picker extension and its
associated configuration.



_Appears in:_
- [InferencePoolSpec](#inferencepoolspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `group` _[Group](#group)_ | Group is the group of the referent API object. When unspecified, the default value<br />is "", representing the Core API group. |  | MaxLength: 253 <br />MinLength: 0 <br />Pattern: `^$\|^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$` <br /> |
| `kind` _[Kind](#kind)_ | Kind is the Kubernetes resource kind of the referent.<br />Required if the referent is ambiguous, e.g. service with multiple ports.<br />Defaults to "Service" when not specified.<br />ExternalName services can refer to CNAME DNS records that may live<br />outside of the cluster and as such are difficult to reason about in<br />terms of conformance. They also may not be safe to forward to (see<br />CVE-2021-25740 for more information). Implementations MUST NOT<br />support ExternalName Services. | Service | MaxLength: 63 <br />MinLength: 1 <br />Pattern: `^[a-zA-Z]([-a-zA-Z0-9]*[a-zA-Z0-9])?$` <br /> |
| `name` _[ObjectName](#objectname)_ | Name is the name of the referent API object. |  | MaxLength: 253 <br />MinLength: 1 <br /> |
| `port` _[Port](#port)_ | Port is the port of the Endpoint Picker extension service.<br />Port is required when the referent is a Kubernetes Service. In this<br />case, the port number is the service port number, not the target port.<br />For other resources, destination port might be derived from the referent<br />resource or this field. |  |  |
| `failureMode` _[EndpointPickerFailureMode](#endpointpickerfailuremode)_ | FailureMode configures how the parent handles the case when the Endpoint Picker extension<br />is non-responsive. When unspecified, defaults to "FailClose". | FailClose | Enum: [FailOpen FailClose] <br /> |


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
- MinLength: 0
- Pattern: `^$|^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$`

_Appears in:_
- [EndpointPickerRef](#endpointpickerref)
- [ParentReference](#parentreference)



#### InferencePool



InferencePool is the Schema for the InferencePools API.






| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `inference.networking.k8s.io/v1` | | |
| `kind` _string_ | `InferencePool` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `spec` _[InferencePoolSpec](#inferencepoolspec)_ | Spec defines the desired state of the InferencePool. |  |  |
| `status` _[InferencePoolStatus](#inferencepoolstatus)_ | Status defines the observed state of the InferencePool. |  |  |






#### InferencePoolSpec



InferencePoolSpec defines the desired state of the InferencePool.



_Appears in:_
- [InferencePool](#inferencepool)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `selector` _[LabelSelector](#labelselector)_ | Selector determines which Pods are members of this inference pool.<br />It matches Pods by their labels only within the same namespace; cross-namespace<br />selection is not supported.<br />The structure of this LabelSelector is intentionally simple to be compatible<br />with Kubernetes Service selectors, as some implementations may translate<br />this configuration into a Service resource. |  |  |
| `targetPorts` _[Port](#port) array_ | TargetPorts defines a list of ports that are exposed by this InferencePool.<br />Every port will be treated as a distinctive endpoint by EPP,<br />addressable as a 'podIP:portNumber' combination. |  | MaxItems: 8 <br />MinItems: 1 <br /> |
| `endpointPickerRef` _[EndpointPickerRef](#endpointpickerref)_ | EndpointPickerRef is a reference to the Endpoint Picker extension and its<br />associated configuration. |  |  |


#### InferencePoolStatus



InferencePoolStatus defines the observed state of the InferencePool.



_Appears in:_
- [InferencePool](#inferencepool)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `parents` _[ParentStatus](#parentstatus) array_ | Parents is a list of parent resources, typically Gateways, that are associated with<br />the InferencePool, and the status of the InferencePool with respect to each parent.<br />A controller that manages the InferencePool, must add an entry for each parent it manages<br />and remove the parent entry when the controller no longer considers the InferencePool to<br />be associated with that parent.<br />A maximum of 32 parents will be represented in this list. When the list is empty,<br />it indicates that the InferencePool is not associated with any parents. |  | MaxItems: 32 <br /> |


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
- [EndpointPickerRef](#endpointpickerref)
- [ParentReference](#parentreference)



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
- [LabelSelector](#labelselector)



#### LabelSelector



LabelSelector defines a query for resources based on their labels.
This simplified version uses only the matchLabels field.



_Appears in:_
- [InferencePoolSpec](#inferencepoolspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `matchLabels` _object (keys:[LabelKey](#labelkey), values:[LabelValue](#labelvalue))_ | MatchLabels contains a set of required \{key,value\} pairs.<br />An object must match every label in this map to be selected.<br />The matching logic is an AND operation on all entries. |  | MaxItems: 64 <br />MinItems: 1 <br /> |


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
- [LabelSelector](#labelselector)



#### Namespace

_Underlying type:_ _string_

Namespace refers to a Kubernetes namespace. It must be a RFC 1123 label.

This validation is based off of the corresponding Kubernetes validation:
https://github.com/kubernetes/apimachinery/blob/02cfb53916346d085a6c6c7c66f882e3c6b0eca6/pkg/util/validation/validation.go#L187

This is used for Namespace name validation here:
https://github.com/kubernetes/apimachinery/blob/02cfb53916346d085a6c6c7c66f882e3c6b0eca6/pkg/api/validation/generic.go#L63

Valid values include:

* "example"

Invalid values include:

* "example.com" - "." is an invalid character

_Validation:_
- MaxLength: 63
- MinLength: 1
- Pattern: `^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`

_Appears in:_
- [ParentReference](#parentreference)



#### ObjectName

_Underlying type:_ _string_

ObjectName refers to the name of a Kubernetes object.
Object names can have a variety of forms, including RFC 1123 subdomains,
RFC 1123 labels, or RFC 1035 labels.

_Validation:_
- MaxLength: 253
- MinLength: 1

_Appears in:_
- [EndpointPickerRef](#endpointpickerref)
- [ParentReference](#parentreference)



#### ParentReference



ParentReference identifies an API object. It is used to associate the InferencePool with a
parent resource, such as a Gateway.



_Appears in:_
- [ParentStatus](#parentstatus)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `group` _[Group](#group)_ | Group is the group of the referent API object. When unspecified, the referent is assumed<br />to be in the "gateway.networking.k8s.io" API group. | gateway.networking.k8s.io | MaxLength: 253 <br />MinLength: 0 <br />Pattern: `^$\|^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$` <br /> |
| `kind` _[Kind](#kind)_ | Kind is the kind of the referent API object. When unspecified, the referent is assumed<br />to be a "Gateway" kind. | Gateway | MaxLength: 63 <br />MinLength: 1 <br />Pattern: `^[a-zA-Z]([-a-zA-Z0-9]*[a-zA-Z0-9])?$` <br /> |
| `name` _[ObjectName](#objectname)_ | Name is the name of the referent API object. |  | MaxLength: 253 <br />MinLength: 1 <br /> |
| `namespace` _[Namespace](#namespace)_ | Namespace is the namespace of the referenced object. When unspecified, the local<br />namespace is inferred.<br />Note that when a namespace different than the local namespace is specified,<br />a ReferenceGrant object is required in the referent namespace to allow that<br />namespace's owner to accept the reference. See the ReferenceGrant<br />documentation for details: https://gateway-api.sigs.k8s.io/api-types/referencegrant/ |  | MaxLength: 63 <br />MinLength: 1 <br />Pattern: `^[a-z0-9]([-a-z0-9]*[a-z0-9])?$` <br /> |


#### ParentStatus



ParentStatus defines the observed state of InferencePool from a Parent, i.e. Gateway.



_Appears in:_
- [InferencePoolStatus](#inferencepoolstatus)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.31/#condition-v1-meta) array_ | Conditions is a list of status conditions that provide information about the observed<br />state of the InferencePool. This field is required to be set by the controller that<br />manages the InferencePool.<br />Supported condition types are:<br />* "Accepted"<br />* "ResolvedRefs" |  | MaxItems: 8 <br /> |
| `parentRef` _[ParentReference](#parentreference)_ | ParentRef is used to identify the parent resource that this status<br />is associated with. It is used to match the InferencePool with the parent<br />resource, such as a Gateway. |  |  |
| `controllerName` _[ControllerName](#controllername)_ | ControllerName is a domain/path string that indicates the name of the controller that<br />wrote this status. This corresponds with the GatewayClass controllerName field when the<br />parentRef references a Gateway kind.<br />Example: "example.net/gateway-controller".<br />The format of this field is DOMAIN "/" PATH, where DOMAIN and PATH are valid Kubernetes names:<br /> https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#names<br />Controllers MAY populate this field when writing status. When populating this field, controllers<br />should ensure that entries to status populated with their ControllerName are cleaned up when they<br />are no longer necessary. |  | MaxLength: 253 <br />MinLength: 1 <br />Pattern: `^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*\/[A-Za-z0-9\/\-._~%!$&'()*+,;=:]+$` <br /> |


#### Port



Port defines the network port that will be exposed by this InferencePool.



_Appears in:_
- [EndpointPickerRef](#endpointpickerref)
- [InferencePoolSpec](#inferencepoolspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `number` _[PortNumber](#portnumber)_ | Number defines the port number to access the selected model server Pods.<br />The number must be in the range 1 to 65535. |  | Maximum: 65535 <br />Minimum: 1 <br /> |


#### PortNumber

_Underlying type:_ _integer_

PortNumber defines a network port.

_Validation:_
- Maximum: 65535
- Minimum: 1

_Appears in:_
- [Port](#port)



