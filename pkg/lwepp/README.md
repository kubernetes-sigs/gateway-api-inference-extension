# Lightweight Endpoint Picker (LWEPP)

This package provides a minimal, lightweight reference implementation of the Endpoint Picker (EPP). It is designed for testing and simple use cases where complex traffic routing policies are not required.

## Core Functions

- **Simple Load Balancing**: Performs basic round-robin load balancing across available pods in the target pool.
- **Endpoint Filtering**: Supports filtering candidate endpoints based on the `test-epp-endpoint-selection` header. If present, the EPP will only select from the list of comma-separated IP addresses provided in the header.
- **Envoy Integration**: Implements the Envoy External Processing (ext_proc) protocol to:
  - Receive request headers and set the target endpoint header to guide Envoy's routing decision.
  - Receive response headers and add a header indicating which endpoint served the request.

## Differences from Full EPP

Unlike the full EPP implementation, this lightweight version:
- Does **not** implement complex scheduling or queueing.
- Does **not** support traffic splitting between different models or adapters.
- Does **not** perform ModelName rewriting.

This implementation is intended as a simple reference or a starting point for environments testing basic connectivity and routing with the Gateway API Inference Extension.
