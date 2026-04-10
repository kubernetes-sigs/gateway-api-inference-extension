# Random Picker

Selects endpoint(s) uniformly at random, ignoring any scores calculated by scorer plugins.

It is registered as type `random-picker` and runs as a scheduling picker.

## What it does

1.  Receives a list of `ScoredEndpoint` candidates.
2.  Shuffles the list in-place to randomize the order.
3.  Returns the top `maxNumOfEndpoints` candidates from the shuffled list.

## Behavioral Intent

This picker provides strict uniform distribution of load across all available candidates. It ignores all scoring signals, making it immune to hot-spotting but unable to leverage optimization signals like cache affinity or predictable latency.

## Inputs consumed

- Consumes the list of `ScoredEndpoint` results (but ignores the `Score` field).

## Configuration

The plugin config supports:

- `maxNumOfEndpoints` (default 1)
  - The maximum number of endpoints to pick and return. Must be > 0. If more candidates are available than this limit, only the top subset is returned.

> [!TIP]
> In most production scenarios, `maxNumOfEndpoints` is left at its default value of `1` to select a single target endpoint for the request.
