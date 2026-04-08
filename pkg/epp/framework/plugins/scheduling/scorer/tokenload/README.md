# Token Load Scorer Plugin

This plugin scores candidate endpoints based on the estimated number of in-flight tokens currently being processed by each model server.

It is registered as type `token-load-scorer` and runs as a scheduling scorer.

## What it does

For each scheduling cycle, the plugin reads the `InFlightLoad` attribute from each endpoint and computes a normalized score based on the total estimated tokens in flight:

$$
\text{score(endpoint)} = 1.0 - \min\left(1.0, \frac{\text{tokens(endpoint)}}{\text{queueThresholdTokens}}\right)
$$

So:
- 0 tokens in flight → score `1.0`
- `queueThresholdTokens` or more tokens → score `0.0`
- others are linearly scaled between them

This scoring mechanism provides a more granular signal than simple request concurrency by accounting for the size of the prompts being processed.

## Scheduling intent

The scorer returns category `Distribution`, helping spread requests away from endpoints that are processing a high volume of tokens, which often correlates with higher latency and resource utilization.

## Inputs consumed

The plugin consumes:
- `attrconcurrency.InFlightLoadKey` (`*attrconcurrency.InFlightLoad`)

## Configuration

The scorer supports the following runtime parameters:

- `queueThresholdTokens` (integer, default: 4194304): The maximum number of in-flight tokens used for score normalization. Endpoints exceeding this threshold will receive a score of `0.0`. The default (4Mi tokens) is equivalent to 128 requests with an average size of 32K tokens.
