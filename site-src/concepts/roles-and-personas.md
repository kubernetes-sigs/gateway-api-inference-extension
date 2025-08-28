# Roles and Personas

Before diving into the details of the API, descriptions of the personas these APIs were designed for will help convey the thought process of the API design.

## Inference Platform Admin

The Inference Platform Admin creates and manages the infrastructure necessary to run LLM workloads, including handling Ops for:

  - Hardware
  - Model Server
  - Base Model
  - Resource Allocation for Workloads
  - Gateway configuration
  - etc

## Inference Workload Owner

An Inference Workload Owner persona owns and manages one or many Generative AI Workloads (LLM focused *currently*). This includes:

- Defining priority
- Managing fine-tunes
  - LoRA Adapters
  - System Prompts
  - Prompt Cache
  - etc.
- Managing rollout of adapters
