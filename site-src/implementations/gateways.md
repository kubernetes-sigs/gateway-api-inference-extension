# Gateway Implementations

This project has several implementations that are planned or in progress:

* [Agentgateway][1]
* [Alibaba Cloud Container Service for Kubernetes][2]
* [Envoy AI Gateway][3]
* [Google Kubernetes Engine][4]
* [Istio][5]
* [Kgateway][6]
* [Kubvernor][7]

[1]:#agentgateway
[2]:#alibaba-cloud-container-service-for-kubernetes
[3]:#envoy-ai-gateway
[4]:#google-kubernetes-engine
[5]:#istio
[6]:#kgateway
[7]:#kubvernor

## Agentgateway

[Agentgateway](https://agentgateway.dev/) is an open source Gateway API and Inference Gateway
[v1.0.0 conformant](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/conformance/reports/v1.0.0/gateway/kgateway/agentgateway)
implementation focusing on AI use cases, including LLM consumption, LLM serving, agent-to-agent ([A2A](https://a2aproject.github.io/A2A/latest/)),
and agent-to-tool ([MCP](https://modelcontextprotocol.io/introduction)). It is the first and only proxy designed specifically for the Kubernetes Gateway API
that is powered by a high performance and scalable Rust dataplane.

Agentgateway can run independently or can be managed by [Kgateway](https://kgateway.dev/).

## Alibaba Cloud Container Service for Kubernetes

[Alibaba Cloud Container Service for Kubernetes (ACK)][ack] is a managed Kubernetes platform 
offered by Alibaba Cloud. The implementation of the Gateway API in ACK is through the 
[ACK Gateway with Inference Extension][ack-gie] component, which introduces model-aware, 
GPU-efficient load balancing for AI workloads beyond basic HTTP routing.

The ACK Gateway with Inference Extension implements the Gateway API Inference Extension 
and provides optimized routing for serving generative AI workloads, 
including weighted traffic splitting, mirroring, advanced routing, etc. 
See the docs for the [usage][ack-gie-usage].

Progress towards supporting Gateway API Inference Extension is being tracked 
by [this Issue](https://github.com/AliyunContainerService/ack-gateway-api/issues/1).

[ack]:https://www.alibabacloud.com/help/en/ack
[ack-gie]:https://www.alibabacloud.com/help/en/ack/product-overview/ack-gateway-with-inference-extension
[ack-gie-usage]:https://www.alibabacloud.com/help/en/ack/ack-managed-and-ack-dedicated/user-guide/intelligent-routing-and-traffic-management-with-ack-gateway-inference-extension

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

## Kgateway

[Kgateway](https://kgateway.dev/) is a Gateway API and Inference Gateway
[v1.0.0 conformant](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/conformance/reports/v1.0.0/gateway/kgateway)
implementation that can run [independently](https://gateway-api-inference-extension.sigs.k8s.io/guides/#__tabbed_3_3), as an
[Istio waypoint](https://kgateway.dev/blog/extend-istio-ambient-kgateway-waypoint/), or within your
[llm-d infrastructure](https://github.com/llm-d-incubation/llm-d-infra) to improve accelerator (GPU) utilization for AI inference workloads.

## Kubvernor

[Kubvernor Rust API Gateway][krg] is an open-source, highly experimental implementation of API controller in Rust programming language. Currently, Kubvernor supports Envoy Proxy. The project aims to be as generic as possible so Kubvernor can be used to manage/deploy different gateways (Envoy, Nginx, HAProxy, etc.). See the docs for the [usage][krgu].

[krg]:https://github.com/kubvernor/kubvernor
[krgu]: https://github.com/kubvernor/kubvernor/blob/main/README.md

