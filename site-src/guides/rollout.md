# Rollout

The goal of this guide is to show you how to perform incremental roll out operations,
which gradually deploy new versions of your inference infrastructure.
You can update LoRA adapters and Inference Pool with minimal service disruption.
This page also provides guidance on traffic splitting and rollbacks to help ensure reliable deployments.

## Prerequisites
Follow the steps in the [main guide](index.md)

## Use case
The following use cases are supported:
*   [LoRA adapter roll out](adapter-rollout.md)
*   [InferencePool roll out](inferencepool-rollout.md)
