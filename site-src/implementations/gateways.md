# Gateway Implementations

This project has several implementations that are planned or in progress:

* [Envoy AI Gateway][1]
* [Kgateway][2]
* [Google Kubernetes Engine][3]
* [Istio][4]

[1]:#envoy-gateway
[2]:#kgateway
[3]:#google-kubernetes-engine
[4]:#istio

## Envoy AI Gateway

[Envoy AI Gateway][aigw-home] is an open source project built on top of 
[Envoy][envoy-org] and [Envoy Gateway][envoy-gateway] to handle request traffic 
from application clients to GenAI services. The features and capabilities are outlined [here][aigw-capabilities]. Use the [quickstart][aigw-quickstart] to get Envoy AI Gateway running with Gateway API in a few simple steps.

Progress towards supporting this project is tracked with a [GitHub
Issue](https://github.com/envoyproxy/ai-gateway/issues/423).

[aigw-home]:https://aigateway.envoyproxy.io/
[envoy-org]:https://github.com/envoyproxy
[envoy-gateway]: https://gateway.envoyproxy.io/
[aigw-capabilities]:https://aigateway.envoyproxy.io/docs/capabilities/
[aigw-quickstart]:https://aigateway.envoyproxy.io/docs/capabilities/gateway-api-inference-extension

## Kgateway

[Kgateway](https://kgateway.dev/) is a feature-rich, Kubernetes-native
ingress controller and next-generation API gateway. Kgateway brings the
full power and community support of Gateway API to its existing control-plane
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

## Istio

[Istio](https://istio.io/) is an open source service mesh and gateway implementation.
It provides a fully compliant implementation of the Kubernetes Gateway API for cluster ingress traffic control. 
For service mesh users, Istio also fully supports east-west (including [GAMMA](https://gateway-api.sigs.k8s.io/mesh/)) traffic management within the mesh.

Gateway API Inference Extension support is being tracked by this [GitHub
Issue](https://github.com/istio/istio/issues/55768).
