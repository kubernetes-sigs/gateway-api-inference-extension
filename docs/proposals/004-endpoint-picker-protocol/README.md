# Endpoint Picker Protocol

## Proposal Status
***Implemented***

# Proposal

The Endpoint Picker, or EPP, is a core component of the inference extension. Ultimately it's
responsible for picking an endpoint from the `InferencePool`. A reference implementation can be
found [here](../../../pkg/epp/).

This doc defines the protocol between the EPP and the proxy (e.g, Envoy).

The EPP MUST implement the Envoy
[external processing service](https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/ext_proc/v3/ext_proc.proto) protocol.

## Version History

| Version  | Date       | Changes                                          |
|----------|------------|--------------------------------------------------|
| v1.0.0   | 2025-07-29 | Added status metadata field for picked endpoints |
| v0.4.0   | 2025-06-03 | Added support for multiple fallback endpoints    |
| v0.3.0   | 2025-03-14 | Added subsetting and fallback support            |
| v0.2.0   | 2025-02-22 | Initial protocol definition                      |

## Endpoint Subset

[REQUEST: Data Plane -> EPP]

For each HTTP request, the data plane CAN communicate the subset of endpoints the EPP MUST pick from by setting an unstructured entry in the [filter metadata](https://github.com/envoyproxy/go-control-plane/blob/63a55395d7a39a8d43dcc7acc3d05e4cae7eb7a2/envoy/config/core/v3/base.pb.go#L819) field of the ext-proc request. The metadata entry for the subset list MUST be wrapped with an outer key (which represents the metadata namespace) with a default of `envoy.lb.subset_hint`.

```go
filterMetadata: {
  "envoy.lb.subset_hint" {
     "x-gateway-destination-endpoint-subset": [<ip:port>, <ip:port>, ...]
  }
}
```

If the key `x-gateway-destination-endpoint-subset` is set, the EPP MUST only select endpoints from the specified list. If none of the endpoints in the list is eligible or the list is empty, then the EPP MUST return a [ImmediateResponse](https://github.com/envoyproxy/envoy/blob/f2023ef77bdb4abaf9feef963c9a0c291f55568f/api/envoy/service/ext_proc/v3/external_processor.proto#L195) with 503 (Service Unavailable) HTTP status code. If the EPP does not select from the list, then this leads to unpredictable behavior.

If the key `x-gateway-destination-endpoint-subset` is not set, then the EPP MUST select from the set defined by the `InferencePool` selector.

## Destination Endpoint

[REQUEST: EPP -> Data Plane]

For each HTTP request, the EPP MUST communicate to the data plane one or more selected model server endpoints via:

1. Setting the `x-gateway-destination-endpoint` HTTP header to one or more selected endpoints.

2. Set an unstructured entry in the [dynamic_metadata](https://github.com/envoyproxy/go-control-plane/blob/c19bf63a811c90bf9e02f8e0dc1dcef94931ebb4/envoy/service/ext_proc/v3/external_processor.pb.go#L320) field of the ext-proc response. The metadata entry for the picked endpoints MUST be wrapped with an outer key (which represents the metadata namespace) with a default of `envoy.lb`.

The primary endpoint MUST be set using the key `x-gateway-destination-endpoint` as follows:
```go
dynamicMetadata: {
  "envoy.lb": {
    "x-gateway-destination-endpoint": <ip:port>
  }
}
```
or:
```go
dynamicMetadata: {
  "envoy.lb": {
    "x-gateway-destination-endpoint": <ip:port>,<ip:port>,...
  }
}
```

The value of the header or metadata entry MUST contain at least one endpoint in `<ip:port>` format or multiple endpoints in `<ip:port>,<ip:port>,...` format. Multiple endpoints are separated by commas. The first valid endpoint in the list will be used. If retry is configured, the proxy will go sequentially down the list until one valid endpoint is found.

Constraints:
- If the EPP did not communicate the server endpoint via these two methods, it MUST return an error as follows:
  -  [ImmediateResponse](https://github.com/envoyproxy/envoy/blob/f2023ef77bdb4abaf9feef963c9a0c291f55568f/api/envoy/service/ext_proc/v3/external_processor.proto#L195) with 503 (Serivce Unavailable) HTTP status code if there are no ready endpoints.
  -  [ImmediateResponse](https://github.com/envoyproxy/envoy/blob/f2023ef77bdb4abaf9feef963c9a0c291f55568f/api/envoy/service/ext_proc/v3/external_processor.proto#L195) with 429 (Too Many Requests) HTTP status code if the request should be dropped (e.g., a Sheddable request, and the servers under heavy load).
- The EPP MUST not set two different values in the header and the inner response metadata value.
- Setting different value leads to unpredictable behavior because proxies aren't guaranteed to support both paths, and so this protocol does not define what takes precedence.

## Destination Endpoint Served

[RESPONSE: Data Plane -> EPP]

For each HTTP response, the data plane MUST communicate to the EPP the endpoint that served the request as follows:

In the ext_proc [ProcessingRequest.metadata_context](https://github.com/envoyproxy/envoy/blob/v1.35.0/api/envoy/service/ext_proc/v3/external_processor.proto#L130) field, a new metadata field "x-gateway-destination-endpoint-served" is added as an unstructured entry with an outer key (which represents the metadata namespace) with a default value of `envoy.lb`:

```go
filterMetadata: {
  "envoy.lb": {
    "x-gateway-destination-endpoint-served": ip:port
}
```

This metadata is required because the EPP provides a list of endpoints to the data plane (see [Destination Endpoint](#destination-endpoint)), and the data plane, according to retry configuration, will attempt each endpoint in order until the request is successful or no more endpoints are available.

### Why envoy.lb namespace as a default?
The `envoy.lb` namespace is a predefined namespace. One common way to use the selected endpoint returned from the server, is [envoy subsets](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/subsets)  where host metadata for subset load balancing must be placed under `envoy.lb`. Note that this is not related to the subsetting feature discussed above, this is an enovy implementation detail.
