# Gateway API Inference Extension 

The Gateway API Inference Extension came out of [wg-serving](https://github.com/kubernetes/community/tree/master/wg-serving) and is sponsored by [SIG Network](https://github.com/kubernetes/community/blob/master/sig-network/README.md#gateway-api-inference-extension). This repo contains: the load balancing algorithm, [ext-proc](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter) code, CRDs, and controllers of the extension.

This extension is intented to provide value to multiplexed LLM services on a shared pool of compute. See the [proposal](https://github.com/kubernetes-sigs/wg-serving/tree/main/proposals/012-llm-instance-gateway) for more info.

## Status

This project is currently in development. 

## Getting Started

Follow this [README](./pkg/README.md) to get the inference-extension up and running on your cluster!

## Website

Detailed documentation is available on our website: https://gateway-api-inference-extension.sigs.k8s.io/

## Contributing

Our community meeting is weekly at Thursday 10AM PDT ([Zoom](https://zoom.us/j/9955436256?pwd=Z2FQWU1jeDZkVC9RRTN4TlZyZTBHZz09), [Meeting Notes](https://www.google.com/url?q=https://docs.google.com/document/d/1frfPE5L1sI3737rdQV04IcDGeOcGJj2ItjMg6z2SRH0/edit?usp%3Dsharing&sa=D&source=calendar&usd=2&usg=AOvVaw1pUVy7UN_2PMj8qJJcFm1U)).

We currently utilize the [#wg-serving](https://kubernetes.slack.com/?redir=%2Fmessages%2Fwg-serving) slack channel for communications.

Contributions are readily welcomed, follow the [dev guide](./docs/dev.md) to start contributing!

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
