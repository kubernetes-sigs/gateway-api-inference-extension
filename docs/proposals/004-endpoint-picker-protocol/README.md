# Endpoint Picker Protocol

The Endpoint Picker, or EPP, is a core component of the inference extension. Ultimately it's
responsible for picking an endpoint from the `InferencePool`. A reference implementation can be
found [here](../../../pkg/epp/).

This doc defines the protocol between the EPP and the proxy (e.g, Envoy).

The EPP MUST implement the Envoy
[external processing service](https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor)protocol.

For each HTTP request, the EPP MUST communicate to the proxy the picked model server endpoint via:

1. Setting the `x-gateway-destination-endpoint` HTTP header to the selected endpoint in <ip:port> format.

2. Set an unstructured entry in the [dynamic_metadata](https://github.com/envoyproxy/go-control-plane/blob/c19bf63a811c90bf9e02f8e0dc1dcef94931ebb4/envoy/service/ext_proc/v3/external_processor.pb.go#L320) field of the ext-proc response. The metadata entry for the picked endpoint MUST be wrapped with an outer key (which represents the metadata namespace) with a default of `envoy.lb`.

The final metadata necessary would look like: 
```go
dynamicMetadata: {
  "envoy.lb": {
    "x-gateway-destination-endpoint": <ip:port>"  
  }
}
```

Note:
- If the EPP did not communicate the server endpoint via these two methods, it MUST return an error.
- The EPP MUST not set two different values in the header and the inner response metadata value. 

## Why envoy.lb namespace as a default? 
The `envoy.lb` namesapce is a predefined namespace used for subsetting. One common way to use the selected endpoint returned from the server, is [envoy subsets](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/subsets) where host metadata for subset load balancing must be placed under `envoy.lb`.

Setting different value leads to unpredictable behavior because proxies aren't guaranteed to support both paths, and so this protocol does not define what takes precedence.

