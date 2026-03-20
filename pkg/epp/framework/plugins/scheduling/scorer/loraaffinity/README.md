# LoRA Affinity Scorer Plugin

This plugin scores candidate endpoints based on whether a requested LoRA adapter is already loaded (or likely to load quickly) on each model server.

It is registered as type `lora-affinity-scorer` and runs as a scheduling scorer.

## What it does

For each candidate endpoint, the plugin checks endpoint metrics for the request's `targetModel` and assigns:

- `1.0`: target model is already active on endpoint (`ActiveModels` contains target)
- `0.8`: target model is not active, but endpoint still has capacity to load more models
- `0.6`: target model is already waiting to be loaded (`WaitingModels` contains target)
- `0.0`: endpoint is at capacity and target model is neither active nor waiting

## Scheduling intent

The scorer returns category `Affinity`, preferring endpoints with higher probability of immediate adapter reuse and lower model-load latency.

## Inputs consumed

The plugin consumes:

- `metrics.ActiveModelsKey` (`map[string]int`)
- `metrics.WaitingModelsKey` (`map[string]int`)

It also relies on endpoint metric `MaxActiveModels` to determine remaining adapter capacity.

## Configuration

This scorer currently has no runtime parameters.
