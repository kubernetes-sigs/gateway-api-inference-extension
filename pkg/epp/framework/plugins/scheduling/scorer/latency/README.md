# Latency Scorer Plugin (`latency-scorer`)

Scores endpoints based on predicted latency headroom, defined as the gap between the predicted
request latency and the user's SLO. Endpoints with more favorable headroom get higher
scores and are more likely to be selected by the picker. For negative headroom
(all endpoints violate SLO), idle endpoints are preferred.

## Inputs

- `LatencyPredictionInfo` endpoint attribute:
  - `TTFTHeadroom` / `TPOTHeadroom` - `SLO - predicted` (positive = meets SLO, negative = violates)
  - `DispatchedRequestCount` - in-flight requests tracked by the EPP
- `PrefixCacheMatchInfo` endpoint attribute:
  - Prefix cache score (0.0-1.0), used in composite fallback

## Output

A score in [0, 1] per endpoint. Higher = better candidate.

## Scoring

### Positive vs negative headroom

If both positive and negative headroom endpoints are present, only positive
endpoints are scored. Negative endpoints receive score 0. This should not normally happen when SLO filters (e.g. `slo-headroom-tier-filter`)
are configured upstream, but provides safe fallback behavior.

### Idle preference (negative headroom only)

When all endpoints have negative headroom, idle preference is applied: if any
endpoint has zero dispatched requests, only idle endpoints are scored. This
ensures idle pods absorb traffic before adding load to already-struggling pods.

### Deficit bucketing (negative headroom only)

Among non-idle negative endpoints, the scorer groups by which SLOs are violated
and scores only the best (least severe) non-empty bucket:
1. Only TPOT negative (most preferred - TTFT is met)
2. Only TTFT negative (TTFT impacts perceived responsiveness most)
3. Both negative (least preferred - violates both SLOs)

### Normalization and blending

TTFT and TPOT headroom values are normalized to [0, 1] and blended:
`combined = ttftWeight * nTTFT + tpotWeight * nTPOT`.

### No SLO behavior

When no SLO headers are set, headroom = `0 - predicted`, which is always negative
for both dimensions. All endpoints are already homogeneous (all negative), so no
tier filter is needed. The scorer differentiates by relative deficit magnitude,
effectively routing to the endpoint with the lowest predicted latency.

### Strategies

| Strategy | Behavior |
|----------|----------|
| `least` (default) | Prefer endpoints closest to SLO (bin-packing to preserve capacity) |
| `most` | Prefer endpoints with most headroom (conservative, max safety margin) |

`least` applies to both positive and negative headroom (closest to SLO boundary in
either direction). `most` only applies to positive headroom endpoints. For negative
headroom, `least` is always used regardless of config, since `most` would incorrectly
prefer the most overloaded endpoint.

### Range-based weight re-normalization

If all endpoints have identical TTFT headroom (range = 0), the TTFT weight is set to 0
and TPOT weight to 1 (and vice versa). This prevents the zero-range dimension from
compressing all scores to the same value.

### Composite fallback

When no predictions are available (sidecar down or timed out), falls back to a weighted
combination of KV cache utilization, queue depth, and prefix cache score.

## Config

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `ttftWeight` | 0.8 | [0, inf) | TTFT blending weight. Higher = favor lower TTFT |
| `tpotWeight` | 0.2 | [0, inf) | TPOT blending weight. Set to 0 for non-streaming |
| `headroomSelectionStrategy` | "least" | least/most | Scoring strategy |
| `compositeKVWeight` | 1 | [0, inf) | KV cache weight in composite fallback |
| `compositeQueueWeight` | 1 | [0, inf) | Queue depth weight in composite fallback |
| `compositePrefixWeight` | 1 | [0, inf) | Prefix cache weight in composite fallback |
