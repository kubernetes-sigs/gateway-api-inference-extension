# Frequently Asked Questions (FAQ)

## How can I get involved with this project?
The [contributing](/contributing) page keeps track of how to get involved with
the project.

## Why isn't this project in the main Gateway API repo?
This project builds on top of Gateway API, however, the Gateway API 
[repo](https://github.com/kubernetes-sigs/gateway-api) is, as the name implies,
an API definition, and does not contain deployables. Inference Gateway deviates
from this; with the EPP & BBR deployables, and projects (like 
[llm-d](https://github.com/llm-d)) depending on these binaries, merging this into
the Gateway API repo would deviate from how that repo operates. 

Additionally, this project represents a close collaboration between
[WG-Serving](https://github.com/kubernetes/community/tree/master/wg-serving),
[SIG-Network](https://github.com/kubernetes/community/tree/master/sig-network),
and the [Gateway API](https://gateway-api.sigs.k8s.io/) subproject. These groups
are all well represented within the ownership of this project, and the separate
repo enables this group to iterate more quickly as this project is getting
started.  As the project stabilizes, we'll revisit if it should become part of
the main Gateway API project.

## Will there be a default controller implementation?
No. Although this project will provide a default/reference implementation of an
extension, each individual Gateway controller can support this pattern. The
scope of this project is to define the API extension model, a reference
extension, conformance tests, and overall documentation. 

A default controller for CRDs that do not have a Gateway dependency may be considered in the future.

## Can you add support for my use case?
Definitely. We are always happy to accept proposals to improving inference capabilities! However, the extensible and pluggable architecture of the EPP allows you to add it yourself, benchmark the capabilities, and then share those with the community. If you are looking for a place for more experimental features, [llm-d](https://github.com/llm-d) is a great place for incubation, many ideas and optimizations that start there, graduate to this repo. Additionally, many of us that work on this project, work there as well!
