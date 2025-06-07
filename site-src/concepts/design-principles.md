# Design Principles

These principles guide our efforts to build flexible [Gateway API] extensions
that empower the development of high-performance [AI Inference] routing
technologies—balancing rapid delivery with long-term growth.

!!! note "Inference Gateways"

    For simplicity, we'll refer to Gateway API Gateways which are
    composed together with AI Inference extensions as "Inference Gateways"
    throughout this document.

[Gateway API]:https://github.com/kubernetes-sigs/gateway-api
[AI Inference]:https://www.arm.com/glossary/ai-inference


## Prioritize stability of the core interfaces

The most critical part of this project is the interfaces between components. To encourage both controller and extension developers to integrate with this project, we need to prioritize the stability of these interfaces.
Although we can extend these interfaces in the future, it’s critical the core is stable as soon as possible.

When describing "core interfaces", we are referring to both of the following:

### 1. Gateway -> Endpoint Picker
At a high level, this defines how a Gateway provides information to an Endpoint Picker, and how the Endpoint Picker selects endpoint(s) that the Gateway should route to.

### 2. Endpoint Picker -> Model Server Framework
This defines what an Endpoint Picker should expect from a compatible Model Server Framework with a focus on health checks and metrics.

## Our presets are finely tuned

We provide APIs and reference implementations for the most common inference requirements. Our defaults for those APIs and implementations—shaped by extensive experience with leading model serving platforms and APIs—are designed to provide the majority of Inference Gateway users with a great default experience without the need for extensive configuration or customization. If you take all of our default extensions and attach them to a compatible `Gateway`, it just "works out of the box".

## Encourage innovation via extensibility

This project is largely based on the idea that extensibility will enable innovation. With that in mind, we should make it as easy as possible for AI researchers to experiment with custom scheduling and routing logic. They should not need to know how to build a Kubernetes controller, or replicate a full networking stack. Instead, all the information needed to make a routing decision should be provided in an accessible format, with clear guidelines and examples of how to customize routing logic.

## Objectives over instructions

The pace of innovation in this ecosystem has been rapid. Focusing too heavily on the specifics of current techniques could result in the API becoming outdated quickly. Instead of making the API too descriptive about _how_ an objective should be achieved, this API should focus on the objectives that a Gateway and/or Endpoint Picker should strive to attain. Overly specific instructions or configuration can start as implementation specific APIs and grow into standards as the concepts become more stable and widespread.

## Composable components and reducing reinvention
While it may be tempting to develop an entirely new AI-focused Gateway, many essential routing capabilities are already well established by Kubernetes. Our focus is on creating a layer of composable components that can be assembled together with other Kubernetes components. This approach empowers engineers to use our solution as a building block—combining established technologies like Gateway API with our extensible model to build higher level solutions. Should you encounter a limitation, consider how existing tooling may be extended or improved first.

## Additions to the API should be carefully prioritized

Every addition to the API should take the principles described above into account. Given that the goal of the API is to encourage a highly extensible ecosystem, each additional feature in the API is raising the barrier for entry to any new controller or extension. Our top priority should be to focus on concepts that we expect to be broadly implementable and useful. The extensible nature of this API will allow each individual implementation to experiment with new features via custom flags or APIs before they become part of the core API surface.
