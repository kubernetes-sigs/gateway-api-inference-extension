# Weighted Random Picker

Selects endpoint(s) randomly, where the probability of an endpoint being selected is proportional to its score.

It is registered as type `weighted-random-picker` and runs as a scheduling picker.

## What it does

1.  Receives a list of `ScoredEndpoint` candidates.
2.  If all candidates have a score of zero or less, it delegates to the `random-picker` for uniform selection.
3.  Uses the **A-Res (Algorithm for Reservoir Sampling)** algorithm to perform mathematically correct weighted random sampling.
    - Generates a random key for each endpoint based on its score: $key_i = U_i^{(1/w_i)}$ where $U_i$ is a random number in $(0,1)$ and $w_i$ is the endpoint's score.
    - Selects the candidates with the largest keys.
4.  Returns the top `maxNumOfEndpoints` candidates.

## Behavioral Intent

This picker resolves the trade-off between `max-score-picker` and `random-picker`. It prefers higher-scoring endpoints while maintaining exploration and avoiding extreme hot-spotting. It is ideal when paired with scorers that produce continuous scores (e.g., Token Load or Latency).

## Inputs consumed

- Consumes the list of `ScoredEndpoint` results and utilizes the `Score` value.

## Configuration

The plugin config supports:

- `maxNumOfEndpoints` (default 1)
  - The maximum number of endpoints to pick and return. Must be > 0. If more candidates are available than this limit, only the top subset is returned.

> [!TIP]
> In most production scenarios, `maxNumOfEndpoints` is left at its default value of `1` to select a single target endpoint for the request.
