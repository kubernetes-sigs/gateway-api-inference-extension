# Body-based routing

A chart to the body-based routing deployment and service.


## Install

To install a body-based router named `body-based-router`, you can run the following command:

```txt
$ helm install body-based-router ./config/charts/body-based-routing \
    --set provider.name=[gke|istio] \
    --set inferenceGateway.name=inference-gateway
```

Note that the provider name is needed to ensure provider-specific manifests are also applied. If no provider is specified, then only
the deployment and service are deployed.

To install via the latest published chart in staging  (--version v0 indicates latest dev version), you can run the following command:

```txt
$ helm install body-based-router oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/body-based-routing \ 
    --version v0
    --set provider.name=[gke|istio]
```

### Install with Custom Cmd-line Flags

To set cmd-line flags, you can use the `--set` option to set each flag, e.g.,:

```txt
$ helm install body-based-router ./config/charts/body-based-routing \
    --set provider.name=[gke|istio] \
    --set inferenceGateway.name=inference-gateway
    --set bbr.flags.<FLAG_NAME>=<FLAG_VALUE>
```

Alternatively, you can define flags in the `values.yaml` file:

```yaml
bbr:
  flags:
    FLAG_NAME: <FLAG_VALUE>
    v: 3 ## Log verbosity
    ...
```

### Install with Custom BBR Plugins Configuration

To set custom BBR plugin config, you can pass it under plugins section. For example:
```yaml
bbr:
  plugins:
    - type: custom-plugin-type
      name: custom-plugin-name
      json: // optional, can be empty
        custom_param: "example-value"
    - type: ...
```

### Configure ext_proc Events

By default, BBR receives all HTTP lifecycle events (request and response headers, body, trailers). If your plugins only need specific events, you can disable the others to reduce latency:

```bash
# Disable response events if plugins only need request data
$ helm install body-based-router ./config/charts/body-based-routing \
    --set provider.name=istio \
    --set provider.supportedEvents.responseHeaders=false \
    --set provider.supportedEvents.responseBody=false \
    --set provider.supportedEvents.responseTrailers=false
```

Or in `values.yaml`:
```yaml
provider:
  name: istio
  supportedEvents:
    requestHeaders: true
    requestBody: true
    requestTrailers: true
    responseHeaders: false  # Disable if plugins don't need response headers
    responseBody: false     # Disable if plugins don't need response body
    responseTrailers: false
```

> **Tip:** Only enable events your plugins need. Each extra event adds a network hop between the proxy and BBR.

### Uninstall

Run the following command to uninstall the chart:

```txt
$ helm uninstall body-based-router
```

## Configuration

The following table list the configurable parameters of the chart.

| **Parameter Name**               | **Description**                                                                                                                                                                                                                                                                                  |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bbr.name`                   | Name for the deployment and service.                                                                                                                                                                                                                                                             |
| `bbr.replicas`               | Number of replicas for the deployment. Defaults to `1`.                                                                                                                                                                                                                                          |
| `bbr.port`                   | Port serving ext_proc. Defaults to `9004`.                                                                                                                                                                                                                                                       |
| `bbr.healthCheckPort`        | Port for health checks. Defaults to `9005`.                                                                                                                                                                                                                                                      |
| `bbr.multiNamespace`         | Boolean flag to indicate whether BBR should watch cross namesapce configmaps or only within the namespace it is deployed. Defaults to `false`.                                                                                                                                                   |
| `bbr.image.repository`       | Rpository of the container image used.                                                                                                                                                                                                                                                           |
| `bbr.image.registry`         | Registry URL where the image is hosted.                                                                                                                                                                                                                                                          |
| `bbr.image.tag`              | Image tag.                                                                                                                                                                                                                                                                                       |
| `bbr.image.pullPolicy`       | Image pull policy for the container. Possible values: `Always`, `IfNotPresent`, or `Never`. Defaults to `Always`.                                                                                                                                                                                |
| `bbr.flags`                  | map of flags which are passed through to bbr. Refer to [runner.go](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/cmd/bbr/runner/runner.go) for complete list.                                                                                                     |
| `bbr.plugins`   | Custom ordered plugins array to set for BBR. each plugin have fields: type, name and optionally json (which represents parameters of the plugin). If not specified, BBR will use by default the `body-field-to-header` to extract the `model` field, and `base-model-to-header` (in that order). |
| `provider.name`              | Name of the Inference Gateway implementation being used. Possible values: `istio`, `gke`. Defaults to `none`.                                                                                                                                                                                    |
| `provider.supportedEvents.requestHeaders` | Enable Request Headers event. Defaults to `true`. |
| `provider.supportedEvents.requestBody` | Enable Request Body event. Defaults to `true`. |
| `provider.supportedEvents.requestTrailers` | Enable Request Trailers event. Defaults to `true`. |
| `provider.supportedEvents.responseHeaders` | Enable Response Headers event. Defaults to `false`. |
| `provider.supportedEvents.responseBody` | Enable Response Body event. Defaults to `false`. |
| `provider.supportedEvents.responseTrailers` | Enable Response Trailers event. Defaults to `false`. |
| `inferenceGateway.name`      | The name of the Gateway. Defaults to `inference-gateway`.                                                                                                                                                                                                                                       |

## Notes

This chart should only be deployed once per Gateway.
