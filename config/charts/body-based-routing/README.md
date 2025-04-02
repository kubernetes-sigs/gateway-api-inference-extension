# Body-based routing

A chart to the body-based routing deployment and service.


## Install

To install a body-based router named `body-based-router`, you can run the following command:

```txt
$ helm install body-based-router ./config/charts/body-based-routing \
    --set provider.name=[gke|istio] \
    --set inference-gateway.name=inference-gateway
```

Note that the provider name is needed to ensure provider-specific manifests are also applied. If no provider is specified, then only
the deployment and service are deployed.

To install via the latest published chart in staging  (--version v0 indicates latest dev version), you can run the following command:

```txt
$ helm install body-based-router oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/body-based-router \ 
    --version v0
    --set provider.name=[gke|istio]
```

## Uninstall

Run the following command to uninstall the chart:

```txt
$ helm uninstall body-based-router
```

## Configuration

The following table list the configurable parameters of the chart.

| **Parameter Name**                          | **Description**                                                                                    |
|---------------------------------------------|----------------------------------------------------------------------------------------------------|
| `bbr.name`                   | Name for the deployment and service.                                                                              |
| `bbr.replicas`               | Number of replicas for the deployment. Defaults to `1`.                                                           |
| `bbr.port`                   | Port serving ext_proc. Defaults to `9004`.                                                                        |
| `bbr.healthCheckPort`        | Port for health checks. Defaults to `9005`.                                                                       |
| `bbr.image.name`             | Name of the container image used.                                                                                 |
| `bbr.image.hub`              | Registry URL where the image is hosted.                                                                           | 
| `bbr.image.tag`              | Image tag.                                                                                                        |
| `bbr.image.pullPolicy`       | Image pull policy for the container. Possible values: `Always`, `IfNotPresent`, or `Never`. Defaults to `Always`. |
| `provider.name`              | Name of the Inference Gateway implementation being used. Possible values: `istio`, `gke`. Defaults to `none`.     |
| `inferenceGateway.name`     | The name of the Gateway. Defaults to `inference-gateway`.                                                         |                        

## Notes

This chart should only be deployed once per Gateway.