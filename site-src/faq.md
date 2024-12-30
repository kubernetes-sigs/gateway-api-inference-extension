# Frequently Asked Questions (FAQ)

## How can I get involved with this project?
The [contributing](/contributing) page keeps track of how to get involved with
the project.

## Why isn't this project in the main Gateway API repo?
This project is an extension of Gateway API, and may eventually be merged into
the main Gateway API repo. As we're starting, this project represents a close
collaboration between
[WG-Serving](https://github.com/kubernetes/community/tree/master/wg-serving),
[SIG-Network](https://github.com/kubernetes/community/tree/master/sig-network),
and the [Gateway API](https://gateway-api.sigs.k8s.io/) subproject. These groups
are all well represented within the ownership of this project, and the separate
repo enables this group to iterate more quickly as this project is getting
started. As the project stabilizes, we'll revisit if it should become part of
the main Gateway API project.

## Will there be a default controller implementation?
No. Although this project will provide a default/reference implementation of an
extension, each individual Gateway controller can support this pattern. The
scope of this project is to define the API extension model, a reference
extension, conformance tests, and overall documentation.

## Can you add support for my use case to the reference extension?
Maybe. We're trying to keep the scope of the reference extension fairly narrow
and instead hoping to see an ecosystem of compatible extensions developed in
this space. Unless a use case fits neatly into the existing scope of our
reference extension, it would likely be better to develop a separate extension
focused on your use case.
