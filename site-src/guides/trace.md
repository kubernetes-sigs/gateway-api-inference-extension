# Trace

This guide describes the current state of trace and how to use them.

## Requirements

You would need to enable the trace feature with following helm option:

```yaml
inferenceExtension:
  tracing:
    enabled: true
    otelExporterEndpoint: "http://localhost:4317"
    sampling:
      sampler: "parentbased_traceidratio"
      samplerArg: "0.1"
```

- `otelExporterEndpoint`: Points to your OpenTelemetry collector endpoint.
- `sampler`: Currently, only `parentbased_traceidratio` is supported. This sampler respects the parent span's sampling decision and applies the configured ratio for root spans.
- `samplerArg`: Base sampling rate for new traces, range [0.0, 1.0]. For example, "0.1" enables 10% sampling.

## Span Coverage

Currently, the inference gateway covers the entry point of the external processing request.

- **Tracer Name**: `gateway-api-inference-extension`
- **Span Name**: `gateway.request`

This span is the root span the entire lifecycle of an external processing request from Envoy, including header and body processing, scheduling decisions, and response handling.

## Attributes

### Span Attributes

Currently, the `gateway.request` span does not include custom attributes. It primarily serves as a container for the request's execution time and provides context for child spans in downstream services.

## Context Propagation

The inference gateway supports distributed tracing by propagating the trace context to downstream services (e.g., model servers).
