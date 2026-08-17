# Gateway Implementations

This project has several conformant Gateway implementations:

- [Gateway Implementations](#gateway-implementations)
  - [Alibaba Cloud Container Service for Kubernetes](#alibaba-cloud-container-service-for-kubernetes)
  - [Istio](#istio)
  - [Agentgateway](#agentgateway)
  - [NGINX Gateway Fabric](#nginx-gateway-fabric)

Agentgateway supports both standalone and Kubernetes deployment modes.

Gateway conformance status is governed by the
[Conformance Report Deprecation Policy](../concepts/conformance.md#conformance-report-deprecation-policy).
Implementations that have not submitted a successful Gateway profile report for
the current minor release or either of the two previous minor releases may be
removed from conformant implementation listings until they submit an accepted
report.

## Alibaba Cloud Container Service for Kubernetes

[Alibaba Cloud Container Service for Kubernetes (ACK)][ack] is a managed Kubernetes platform 
offered by Alibaba Cloud. The implementation of the Gateway API in ACK is through the 
[ACK Gateway with Inference Extension][ack-gie] component, which introduces model-aware, 
GPU-efficient load balancing for AI workloads beyond basic HTTP routing.

The ACK Gateway with Inference Extension implements the Gateway API Inference Extension 
and provides optimized routing for serving generative AI workloads, 
including weighted traffic splitting, mirroring, advanced routing, etc. 
See the docs for the [usage][ack-gie-usage].

ACK Gateway with Inference Extension v1.6.1-apsara.3 passes the Gateway profile for
Gateway API Inference Extension v1.5.0. See the [conformance report][ack-gie-report].

[ack]:https://www.alibabacloud.com/help/en/ack
[ack-gie]:https://www.alibabacloud.com/help/en/ack/product-overview/ack-gateway-with-inference-extension
[ack-gie-usage]:https://www.alibabacloud.com/help/en/ack/ack-managed-and-ack-dedicated/user-guide/intelligent-routing-and-traffic-management-with-ack-gateway-inference-extension
[ack-gie-report]:https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/conformance/reports/v1.5.0/gateway/ack-gateway

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

## NGINX Gateway Fabric

[NGINX Gateway Fabric][nginx-gateway-fabric] is an open-source project that provides an implementation of the Gateway API using [NGINX][nginx] as the data plane. The goal of this project is to implement the core Gateway API to configure an HTTP or TCP/UDP load balancer, reverse-proxy, or API gateway for applications running on Kubernetes. You can find the comprehensive NGINX Gateway Fabric user documentation on the [NGINX Documentation][nginx-docs] website.

[nginx-gateway-fabric]: https://github.com/nginx/nginx-gateway-fabric
[nginx]:https://nginx.org/
[nginx-docs]:https://docs.nginx.com/nginx-gateway-fabric/
