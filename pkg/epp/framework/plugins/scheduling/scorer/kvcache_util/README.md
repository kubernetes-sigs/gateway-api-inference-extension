# KV Cache Utilization Scorer Plugin

This plugin scores candidate endpoints using each endpoint's current KV-cache utilization.

It is registered as type `kv-cache-utilization-scorer` and runs as a scheduling scorer.

## What it does

For each candidate endpoint, the plugin computes:

\[
\text{score(endpoint)} = 1 - \text{kvCacheUsagePercent}
\]

Where `kvCacheUsagePercent` is read from endpoint metrics.

This means:

- lower KV-cache usage \(\rightarrow\) higher score
- higher KV-cache usage \(\rightarrow\) lower score

## Scheduling intent

The scorer returns category `Distribution`, so it helps spread traffic away from endpoints with high KV-cache pressure.

## Inputs consumed

The plugin consumes:

- `metrics.KVCacheUsagePercentKey` (`float64`)

## Configuration

This scorer currently has no runtime parameters.

## Source files

- `kvcache_utilization.go`: plugin type, factory, and scoring logic.
- `kvcache_utilization_test.go`: table-driven tests for score behavior across utilization levels.
