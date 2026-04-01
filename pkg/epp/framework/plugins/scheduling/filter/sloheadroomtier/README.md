# SLO Headroom Tier Filter (`slo-headroom-tier-filter`)

Probabilistic filter that selects endpoints based on predicted latency headroom
(SLO - predicted). Splits endpoints into positive (meets SLO) and negative (violates SLO)
tiers and probabilistically selects one tier.

This filter is only needed when SLO headers are configured. When no SLO is set,
headroom = 0 - predicted, which is always negative. All endpoints pass through in
the negative tier (the filter is effectively a no-op).

## Behavior

- Positive tier: `TTFTHeadroom >= 0 AND TPOTHeadroom >= 0`
- Negative tier: at least one headroom is negative
- When both tiers exist: 99% keep positive only, 1% keep negative only (epsilon exploration)
- When only one tier exists: keep that tier
- When no predictions: keep all endpoints
- Endpoints without predictions are placed in the negative tier

The 1% exploration ensures endpoints recovering from overload receive occasional traffic
so their state is re-evaluated.

## Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilonExploreNeg` | 0.01 | Probability of selecting negative tier when both exist |

## Inputs

- `LatencyPredictionInfo` endpoint attribute:
  - `TTFTHeadroom` / `TPOTHeadroom` for tier classification
