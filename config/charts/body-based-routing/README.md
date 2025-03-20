# Body-based routing

A chart to the body-based routing deployment and service.


## Install

To install a body-based router named `body-based-router`, you can run the following command:

```txt
$ helm install body-based-router ./config/charts/body-based-routing
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
| `bbr.image.name`             | Name of the container image used.                                                                                 |
| `bbr.image.hub`              | Registry URL where the image is hosted.                                                                           | 
| `bbr.image.tag`              | Image tag.                                                                                                        |
| `bbr.image.pullPolicy`       | Image pull policy for the container. Possible values: `Always`, `IfNotPresent`, or `Never`. Defaults to `Always`. |

## Notes

This chart will only deploy the body-based router deployment and service.
Note that this should only be deployed once per Gateway.

Additional configuration is needed to configure a proxy extension that calls
out to the service in the request path. For example, vwith Envoy Gateway, this
would require configuring EnvoyExtensionPolicy.
