# Lightweight Endpoint Picker (LWEPP)

This package provides a minimal, lightweight reference implementation of the Endpoint Picker (EPP).

## Core Functions

- **Envoy Integration**: Implements the Envoy External Processing (ext_proc) protocol to:
  - Receive request headers and set the target endpoint header to guide Envoy's routing decision.
  - Receive response headers and add a header indicating which endpoint served the request.
- **Simple Load Balancing**: Performs basic round-robin load balancing across available pods in the target pool.

## Conformance Testing Support

The LWEPP includes two behaviors specifically to support the Gateway API Inference Extension conformance test suite.

### Header-Based Endpoint Filtering

Conformance tests need to steer individual requests to a specific backend pod in order to verify that routing works correctly. When the `test-epp-endpoint-selection` request header is present, the LWEPP restricts its candidate pool to only the pods whose IP addresses appear in the header (comma-separated). If none of the listed IPs match a known pod, or if the header is absent, the LWEPP falls back to round-robin across all available pods.

> **Note**: This header is only intended for use in test environments. It should not be present in production traffic.

### Served Endpoint Reporting

The `GatewayDestinationEndpointServed` test sends a unique `x-conformance-request-id`
header. LWEPP logs the served endpoint from response metadata together with that ID
as a JSON `Conformance served endpoint observation` event. After validating the
client response and backend, the test polls EPP pod logs with a bounded timeout.
This permits the gateway to send the client response before EPP processes response
metadata; no response-header mutation is required for this test to pass.

The test runner needs access to list EPP pods and read their `pods/log` subresource.
The conformance manifests already configure LWEPP with the JSON log encoder.
Missing observations time out, while missing or incorrect metadata in a matching
observation fails the test. Unique request IDs prevent old or unrelated records
from satisfying it. Logs use normal container log retention, with no in-memory
observation store to clean up.

LWEPP retains `x-conformance-test-served-endpoint` response headers for existing
users and other tests; this test no longer uses them as its evidence.
