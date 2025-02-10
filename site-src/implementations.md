# Implementations

This project has several implementations that are planned or in progress:

* [Envoy Gateway][1]
* [Kgateway][2]
* [Google Kubernetes Engine][3]

[1]:#envoy-gateway
[2]:#kgateway
[3]:#google-kubernetes-engine

## Envoy Gateway

[Envoy Gateway][eg-home] is an [Envoy][envoy-org] subproject for managing
Envoy-based application gateways. The supported APIs and fields of the Gateway
API are outlined [here][eg-supported]. Use the [quickstart][eg-quickstart] to
get Envoy Gateway running with Gateway API in a few simple steps.

Progress towards supporting this project is tracked with a [GitHub
Issue](https://github.com/envoyproxy/gateway/issues/4423).

[eg-home]:https://gateway.envoyproxy.io/
[envoy-org]:https://github.com/envoyproxy
[eg-supported]:https://gateway.envoyproxy.io/docs/tasks/quickstart/
[eg-quickstart]:https://gateway.envoyproxy.io/docs/tasks/quickstart

## Kgateway

[Kgateway](https://kgateway.dev/) is a feature-rich, Kubernetes-native
ingress controller and next-generation API gateway. Kgateway brings the
full power and community support of the Gateway API to its existing control-plane
implementation.

Progress towards supporting this project is tracked with a [GitHub
Issue](https://github.com/kgateway-dev/kgateway/issues/10411).

## Google Kubernetes Engine

[Google Kubernetes Engine (GKE)][gke] is a managed Kubernetes platform offered
by Google Cloud. GKE's implementation of the Gateway API is through the [GKE
Gateway controller][gke-gateway] which provisions Google Cloud Load Balancers
for Pods in GKE clusters.

The GKE Gateway controller supports weighted traffic splitting, mirroring,
advanced routing, multi-cluster load balancing and more. See the docs to deploy
[private or public Gateways][gke-gateway-deploy] and also [multi-cluster
Gateways][gke-multi-cluster-gateway].

Progress towards supporting this project is tracked with a [GitHub
Issue](https://github.com/GoogleCloudPlatform/gke-gateway-api/issues/20).

[gke]:https://cloud.google.com/kubernetes-engine
[gke-gateway]:https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api
[gke-gateway-deploy]:https://cloud.google.com/kubernetes-engine/docs/how-to/deploying-gateways
[gke-multi-cluster-gateway]:https://cloud.google.com/kubernetes-engine/docs/how-to/deploying-multi-cluster-gateways
