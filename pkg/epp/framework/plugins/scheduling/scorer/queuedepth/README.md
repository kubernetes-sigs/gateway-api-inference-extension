# Queue Depth Scorer Plugin

This plugin scores candidate endpoints by current waiting-queue depth.

It is registered as type `queue-scorer` and runs as a scheduling scorer.

## What it does

For each scheduling cycle, the plugin reads `WaitingQueueSize` from endpoint metrics and computes a normalized score:

\[
\text{score(endpoint)} = \frac{\maxQueue - \text{queue(endpoint)}}{\maxQueue - \minQueue}
\]

So:

- shortest queue gets score `1.0`
- longest queue gets score `0.0`
- others are linearly scaled between them

If all endpoints have the same queue size (`maxQueue = minQueue`), all endpoints receive a neutral score of `1.0`.

## Scheduling intent

The scorer returns category `Distribution`, helping spread requests away from endpoints with deeper backlogs.

## Inputs consumed

The plugin consumes:

- `metrics.WaitingQueueSizeKey` (`int`)

## Configuration

This scorer currently has no runtime parameters.