# Gateway Implementations

This project has several implementations that are planned or in progress:

- [Gateway Implementations](#gateway-implementations)
  - [Alibaba Cloud Container Service for Kubernetes](#alibaba-cloud-container-service-for-kubernetes)
  - [Envoy AI Gateway](#envoy-ai-gateway)
  - [Google Kubernetes Engine](#google-kubernetes-engine)
  - [Istio](#istio)
  - [Agentgateway](#agentgateway)
  - [Kubvernor](#kubvernor)
  - [NGINX Gateway Fabric](#nginx-gateway-fabric)

[1]:#alibaba-cloud-container-service-for-kubernetes
[2]:#envoy-ai-gateway
[3]:#google-kubernetes-engine
[4]:#istio
[5]:#agentgateway
[6]:#kubvernor
[7]:#nginx-gateway-fabric

Agentgateway supports both standalone and Kubernetes deployment modes.

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

## Agentgateway

[Agentgateway](https://agentgateway.dev/) is a high-performance, Rust-based AI
gateway for LLM, MCP, and A2A workloads that can also serve as a Gateway API
and Inference Gateway implementation.

It can run as a [standalone binary or in Docker](https://agentgateway.dev/docs/standalone/latest/)
on a local machine or server without Kubernetes, or be deployed on
[Kubernetes](https://agentgateway.dev/docs/kubernetes/latest/) for cluster-based
environments, or within your
[llm-d infrastructure](https://github.com/llm-d-incubation/llm-d-infra) to
improve accelerator (GPU) utilization for AI inference workloads.

## Kubvernor

[Kubvernor Rust API Gateway][krg] is an open-source, highly experimental implementation of API controller in Rust programming language. Currently, Kubvernor supports Envoy Proxy. The project aims to be as generic as possible so Kubvernor can be used to manage/deploy different gateways (Envoy, Nginx, HAProxy, etc.). See the docs for the [usage][krgu].

[krg]:https://github.com/kubvernor/kubvernor
[krgu]: https://github.com/kubvernor/kubvernor/blob/main/README.md

## NGINX Gateway Fabric

[NGINX Gateway Fabric][nginx-gateway-fabric] is an open-source project that provides an implementation of the Gateway API using [NGINX][nginx] as the data plane. The goal of this project is to implement the core Gateway API to configure an HTTP or TCP/UDP load balancer, reverse-proxy, or API gateway for applications running on Kubernetes. You can find the comprehensive NGINX Gateway Fabric user documentation on the [NGINX Documentation][nginx-docs] website.

[nginx-gateway-fabric]: https://github.com/nginx/nginx-gateway-fabric
[nginx]:https://nginx.org/
[nginx-docs]:https://docs.nginx.com/nginx-gateway-fabric/
