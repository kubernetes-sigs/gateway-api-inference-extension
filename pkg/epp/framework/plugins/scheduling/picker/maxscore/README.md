# Max Score Picker

Selects the endpoint(s) with the highest score calculated during the scoring phase.

It is registered as type `max-score-picker` and runs as a scheduling picker.

## What it does

1.  Receives a list of `ScoredEndpoint` candidates.
2.  Shuffles the list in-place to ensure random tie-breaking when multiple endpoints share the same maximum score.
3.  Sorts the candidates by score in descending order.
4.  Returns the top `maxNumOfEndpoints` candidates.

## Behavioral Intent

This picker maximizes the adherence to scoring objectives (e.g., cache affinity, lowest load). However, it is susceptible to **hot-spotting** if many concurrent requests produce identical scores for the same endpoint (e.g., identical prompts targeting a specific cache hit).

## Inputs consumed

- Consumes the list of `ScoredEndpoint` results from the scoring phase.

## Configuration

The plugin config supports:

- `maxNumOfEndpoints` (default 1)
  - The maximum number of endpoints to pick and return. Must be > 0. If more candidates are available than this limit, only the top subset is returned.

> [!TIP]
> In most production scenarios, `maxNumOfEndpoints` is left at its default value of `1` to select a single target endpoint for the request.
