# Parsers

This directory contains parser plugins used to parse and understand the payloads of requests and responses. This understanding is key for empowering features like prefix-cache aware request scheduling and response usage tracking.

## Supported Parsers

*   **`openai-parser`**: The default parser, supporting the [OpenAI API](https://developers.openai.com/api/reference/overview). This is used when no parser is explicitly specified in the `EndpointPickerConfig`.
*   **`vllmgrpc-parser`**: A parser designed to handle requests specifically for the [vLLM gRPC API](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/grpc_server/).

## Configuration

Parsers are configured via the `parser` section in the `EndpointPickerConfig` YAML file. You must first instantiate the parser plugin in the `plugins` section, and then reference its name in the `parser` section. 

If no parser is specified, `openai-parser` is used as the fallback.

Here is an example configuration using the `vllmgrpc-parser`:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: maxScore
  type: max-score-picker
- name: vllmgrpcParser
  type: vllmgrpc-parser
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
parser:
  pluginRef: vllmgrpcParser
```
