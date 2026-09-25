# Lightweight Endpoint Picker (LWEPP)

This package provides a minimal, lightweight reference implementation of the Endpoint Picker (EPP).

## Core Functions

- **Envoy Integration**: Implements the Envoy External Processing (ext_proc) protocol to:
  - Receive request headers and set the target endpoint header to guide Envoy's routing decision.
  - Receive response headers and add a header indicating which endpoint served the request.
- **Simple Load Balancing**: Performs basic round-robin load balancing across available pods in the target pool.

## Conformance Testing Support

The LWEPP supports endpoint steering and reports routing decisions to the Gateway API Inference Extension conformance test suite. These test headers do not add requirements to the EPP protocol.

### Header-Based Endpoint Filtering

Conformance tests need to steer individual requests to a specific backend pod in order to verify that routing works correctly. When the `test-epp-endpoint-selection` request header is present, the LWEPP restricts its candidate pool to only the pods whose IP addresses appear in the header (comma-separated). If none of the listed IPs match a known pod, or if the header is absent, the LWEPP falls back to round-robin across all available pods.

> **Note**: This header is only intended for use in test environments. It should not be present in production traffic.

### Routing Response Headers

When the gateway sends response headers to the LWEPP through `ext_proc`, the LWEPP adds these headers to the client response:

| Header | Value and source |
| --- | --- |
| `x-conformance-test-selected-endpoint` | The endpoint selected for this request, as `IP:port` (IPv6 addresses are bracketed). Omitted if no endpoint was selected. |
| `x-conformance-test-epp-pool` | The picker's configured `InferencePool`, as `namespace/name`, from `--pool-namespace` and `--pool-name`. Available before the datastore syncs. |
| `x-conformance-test-served-endpoint` | The endpoint reported by the gateway in `envoy.lb.x-gateway-destination-endpoint-served` response metadata. If the metadata is missing, the value is `fail: missing envoy lb metadata` or `fail: missing destination endpoint served metadata`. |
| `x-went-into-resp-headers` | `true`, a debugging marker that response-header processing ran. |

The selected-endpoint header exposes the picker's decision to the test client. The existing `x-gateway-destination-endpoint` request header and dynamic metadata communicate that decision to the gateway for routing. The served-endpoint header independently reports where the gateway says it routed the request. Comparing the two can detect an ignored selection; comparing the reported pool with the responding backend's pool can detect a request sent to the wrong picker.

The LWEPP does not ask the backend to echo the selection as the served endpoint. It also does not copy backend response headers into its mutations, which would allow a backend's value to overwrite these reports. Other response headers remain unchanged by the gateway's header mutation.

These reports require response-header processing through `ext_proc`; they are absent if that stage is skipped. Consumers of the new selected-endpoint and pool headers must allow for their absence, including when using an older LWEPP image. A `fail:` value in the served-endpoint header indicates missing gateway metadata, not an endpoint address. Tests can also compare the selection with the responding pod identified by the backend's response body and Kubernetes pod data.
