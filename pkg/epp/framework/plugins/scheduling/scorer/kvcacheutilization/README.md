# KV Cache Utilization Scorer Plugin

This plugin scores candidate endpoints using each endpoint's current KV-cache utilization.

It is registered as type `kv-cache-utilization-scorer` and runs as a scheduling scorer.

## What it does

For each candidate endpoint, the plugin computes:

```
  {score(endpoint)} = 1 - {kvCacheUsagePercent}
```

Where `kvCacheUsagePercent` is read from endpoint metrics.

This means:

- lower KV-cache usage -> higher score
- higher KV-cache usage -> lower score

## Scheduling intent

The scorer returns category `Distribution`, so it helps spread traffic away from endpoints with high KV-cache pressure.

## Inputs consumed

The plugin consumes:

- `metrics.KVCacheUsagePercentKey` (`float64`)

## Configuration

This scorer currently has no runtime parameters.
