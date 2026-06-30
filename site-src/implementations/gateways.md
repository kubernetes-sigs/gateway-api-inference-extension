# Gateway Implementations

This project has several conformant Gateway implementations:

- [Gateway Implementations](#gateway-implementations)
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
