# Cost Reporting Plugin

## Overview

This plugin for the Endpoint Picker (EPP) allows you to report the "cost" of processing a request back to the downstream proxy (e.g., Envoy). The cost is calculated based on data extracted from the model server's response body. This information is returned as dynamic metadata in the `ext_proc` response.

Currently only data from the "usage" object in the response body is supported. See below for details.

The plugin is designed to be flexible, allowing users to define how the cost is calculated using Common Expression Language (CEL) expressions without modifying the EPP binary.

## Purpose

The primary purpose of this plugin is to provide visibility into resource consumption for each request. This data can be used for:

*   **Advanced Load Balancing:** Informing routing decisions based on request cost.
*   **Prefix Sharding:** Optimizing cache utilization and routing in sharded environments.
*   **Observability:** Monitoring token usage and other cost metrics.
*   **Throttling/Billing:** Implementing usage-based quotas or billing.

## Configuration

The plugin is configured within the EPP server's configuration file (provided via `--config-file` or `--config-text`). It uses the `cost-reporter` type.

```yaml
apiVersion: config.apix.gateway-api-inference-extension.sigs.k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - name: total-tokens-cost-reporter
    type: cost-reporter
    parameters:
      # Defines where in dynamic metadata to return the data
      metric:
        namespace: envoy.lb  # Optional: Defaults to envoy.lb if omitted
        # What key to use in the provided namespace for the value from the expression
        name: x-gateway-inference-request-cost # Example key
        # Specifies the source of data for the CEL expression. Currently only "usage" is supported.
        dataSource: usage
        # The CEL expression to calculate the cost. Must return an integer.
        expression: |
          usage.prompt_tokens + usage.completion_tokens
        # Optional: CEL expression to determine if this metric should be calculated/reported.
        # Must return a boolean.
        condition: "has(usage.prompt_tokens) && has(usage.completion_tokens)"
```

## Available Data

When `dataSource` is `usage`, the CEL expressions have access to a `usage` object. This object's structure depends on the format of the response body sent by the model server. Commonly, for OpenAI-compatible APIs, this object might include:

*   `usage.prompt_tokens`: Number of tokens in the prompt.
*   `usage.completion_tokens`: Number of tokens in the generated completion.
*   `usage.total_tokens`: Total number of tokens.

**Example CEL Expressions:**

*   Report total tokens:
    ```cel
    (has(usage.prompt_tokens) ? usage.prompt_tokens : 0) + (has(usage.completion_tokens) ? usage.completion_tokens : 0)
    ```
*   Report prompt tokens only:
    ```cel
    usage.prompt_tokens
    ```
*   Conditional reporting (only if completion tokens are present):
    ```cel
    // condition
    has(usage.completion_tokens)
    // expression
    usage.completion_tokens
    ```

## Dynamic Metadata Output

The plugin sends the calculated cost back to the proxy via `ext_proc` dynamic metadata. For the example configuration above, the proxy will receive instructions to set metadata like this:

```
dynamic_metadata {
  key: "envoy.lb"
  value: {
    fields: {
      key: "x-gateway-inference-request-cost"
      value: { number_value: <calculated_cost> }
    }
  }
}
```
