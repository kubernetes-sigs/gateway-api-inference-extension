A cluster with:

- Support for one of the three most recent Kubernetes minor [releases](https://kubernetes.io/releases/).
- Support for services of type `LoadBalancer`. For [kind](https://kind.sigs.k8s.io/) (Kubernetes IN Docker) clusters, follow [this guide](https://kind.sigs.k8s.io/docs/user/loadbalancer)
  to get services of type LoadBalancer working.
- Support for [sidecar containers](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/) (enabled by default since Kubernetes v1.29)
  to run the model server deployment. Learn more about [sidecar containers in Kubernetes](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/).

Tools:

- [Helm](https://helm.sh/docs/intro/install/)
- [jq](https://jqlang.org/download/)
