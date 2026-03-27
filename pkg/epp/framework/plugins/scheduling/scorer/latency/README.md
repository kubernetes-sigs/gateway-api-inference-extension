# Latency Scorer Plugin (`latency-scorer`)

Scores endpoints based on predicted latency headroom - the gap between the predicted
request latency and the user's SLO (Service Level Objective). Endpoints with more favorable
headroom get higher scores and are more likely to be selected by the picker.

## Inputs

- **`LatencyPredictionInfo`** endpoint attribute (produced by `predicted-latency-producer`):
  - `TTFT` / `TPOT` - predicted latency values (ms)
  - `TTFTHeadroom` / `TPOTHeadroom` - `SLO - predicted` (positive = meets SLO, negative = violates)
  - `TTFTValid` / `TPOTValid` - whether predictions are within SLO
  - `DispatchedRequestCount` - EPP-tracked in-flight requests (for idle detection)
- **`PrefixCacheMatchInfo`** endpoint attribute (produced by `prefix-cache-scorer`):
  - Prefix cache score (0.0–1.0) used for affinity gating (sticky routing)

## Output

A score in [0, 1] per endpoint. Higher = better candidate. Endpoints excluded by affinity
gating or tier selection get score = 0.

## How It Works

### Step 1: Affinity Gating (Prefix Cache Stickiness)

"Sticky" endpoints are those with a high prefix cache score for the current request -
meaning the request's prompt is already cached on that endpoint and prefill will be
faster. The scorer narrows the candidate set to sticky endpoints before scoring, so
requests tend to route to endpoints that already have their prefix cached.

**Global gate** (tau=0.99): Before tier split, try to narrow to endpoints with prefix
cache score >= 0.99. If no endpoints pass, all endpoints are kept (the gate is a no-op).
With probability `epsilonExploreSticky` (default 1%), the gate is skipped entirely to
allow exploration of non-sticky endpoints.

**Within-tier gate** (tau=0.80): After tier split, within each tier, try to narrow to
endpoints with prefix cache score >= 0.80. If no endpoints pass, all are kept. This is
a separate, independent gate from the global gate. When the global gate succeeds (all
remaining endpoints have score >= 0.99), this gate has no additional effect. When the
global gate finds no matches and falls back to all endpoints, this tier-level gate
provides a second chance to find "good enough" cache matches within each tier.

**TTFT load gate**: Before accepting the sticky subset, check if sticky endpoints are
significantly slower than non-sticky alternatives. If the best sticky endpoint's
predicted TTFT exceeds the best non-sticky endpoint's TTFT by more than
`affinityMaxTTFTPenaltyMs` (default 5000ms), break stickiness and keep all endpoints.
This prevents hot-spotting where all requests pile onto one cached endpoint that is
already overloaded.

### Step 2: Tier Classification

Endpoints are split into two tiers based on headroom (headroom = SLO - predicted latency):

- **Positive tier**: `TTFTHeadroom >= 0 AND TPOTHeadroom >= 0` - predicted latency meets SLO
- **Negative tier**: at least one headroom dimension is negative - predicted latency exceeds SLO

### Step 3: Tier Selection (Epsilon Exploration)

When both tiers have endpoints:
- **99% of requests**: only the positive tier is scored; negative-tier endpoints get score = 0
- **1% of requests** (`epsilonExploreNeg`): only the negative tier is scored; positive-tier
  endpoints get score = 0

This 1% exploration ensures that endpoints recovering from overload (transitioning from
negative to positive headroom) receive occasional traffic so their state is re-evaluated
rather than being permanently starved.

When only one tier has endpoints, that tier is scored.

### Step 4: Scoring Within a Tier

**Positive tier scoring:**

Within the gated subset, TTFT and TPOT headroom are each normalized to [0, 1]:
- `nTTFT = (ttftHeadroom - min) / (max - min)` across the subset
- Same for nTPOT

Then blended: `combined = alpha * nTTFT + beta * nTPOT` (where alpha and beta are
normalized from `ttftWeight` and `tpotWeight`).

With the **`least` strategy** (default): `score = 1 - combined`. This means endpoints
with the *least* headroom (closest to SLO) get the *highest* scores. The rationale is
bin-packing: by sending requests to the endpoint that can *just barely* meet the SLO,
we keep other endpoints lightly loaded for future requests that may need more capacity.

With the **`most` strategy**: `score = combined`. Endpoints with the most headroom get
the highest scores - a conservative approach that maximizes safety margin for each request.

**Example (least strategy, 3 endpoints):**

| Endpoint | Predicted TTFT | SLO | Headroom | nTTFT | Score |
|----------|---------------|-----|----------|-------|-------|
| A | 80ms | 200ms | 120ms | 1.0 | 0.0 (most headroom, least preferred) |
| B | 150ms | 200ms | 50ms | 0.25 | 0.75 |
| C | 170ms | 200ms | 30ms | 0.0 | 1.0 (least headroom, most preferred) |

**Negative tier scoring:**

All endpoints violate SLO. The scorer uses hierarchical bucketing to prioritize:

1. **Idle endpoints first**: If any endpoint has zero dispatched requests, only idle
   endpoints are scored. Busy endpoints get score = 0. This ensures idle pods absorb
   traffic before adding more load to already-struggling pods.

2. **Deficit bucketing** (among non-idle, or when all are busy):
   - Bucket 1: Both TTFT and TPOT negative (worst - violates both dimensions)
   - Bucket 2: Only TTFT negative (violates time-to-first-token)
   - Bucket 3: Only TPOT negative (violates token generation speed)

   Higher-numbered buckets are preferred. Bucket 3 (only TPOT violation) is preferred
   over Bucket 2 (only TTFT violation) because TTFT directly impacts perceived
   responsiveness - users experience TTFT as the initial wait before any output
   appears, making it the more impactful SLO to meet. Within each bucket,
   deficits are normalized independently using `negHeadroomTTFTWeight` and
   `negHeadroomTPOTWeight`, and the endpoint with the *smallest* deficit (closest to
   meeting SLO) gets the highest score.

**Range-based weight re-normalization**: If all endpoints in the subset have identical
TTFT headroom (range = 0), the TTFT weight is automatically set to 0 and TPOT weight
to 1 (and vice versa). This prevents the zero-range dimension from compressing all
scores to the same value.

### Fallback: No Predictions Available

If the `predicted-latency-producer` is down or timed out (exceeded the PrepareDataPlugin
timeout), no endpoints have `LatencyPredictionInfo`. The scorer falls back to **composite
scoring**: a weighted combination of KV cache utilization, queue depth, and prefix cache
score. This avoids returning all-zero scores which would cause uniform random selection.

The composite fallback exists because the framework doesn't support conditional scorer
execution ("run scorer A; if no results, run B instead"). See `compositeKVWeight`,
`compositeQueueWeight`, `compositePrefixWeight` config parameters.

## Behavior Without SLOs

When no SLO headers (`x-slo-ttft-ms`, `x-slo-tpot-ms`) are set on the request, the SLO
defaults to 0. This means headroom = `0 - predicted_latency`, which is always negative,
so all endpoints land in the negative tier.

This still works because the negative tier scores by *relative* deficit, not absolute.
The endpoint with predicted TTFT = 100ms gets a better score than one with 500ms, even
though both have negative headroom. The scorer effectively becomes "route to the endpoint
with the lowest predicted latency", with idle pod preference on top. The SLO value
doesn't matter for relative ordering - only the differences between predictions matter.

## Strategies

| Strategy | Behavior | When to use |
|----------|----------|-------------|
| `least` (default) | Prefer endpoints closest to SLO (bin-packing) | High throughput: pack requests tightly to keep spare capacity |
| `most` | Prefer endpoints with most headroom (conservative) | Low latency: maximize safety margin per request |
| `composite-only` | Use KV cache + queue + prefix scoring with SLO and affinity gates | When predictions are available but you prefer metric-based scoring with SLO-aware gating |

## Config

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `ttftWeight` | 0.8 | [0, inf) | Positive-tier TTFT blending weight. Higher = favor lower TTFT. Set to 0 for TPOT-only scoring |
| `tpotWeight` | 0.2 | [0, inf) | Positive-tier TPOT blending weight. Set to 0 for non-streaming workloads |
| `negHeadroomTTFTWeight` | 0.8 | [0, inf) | Negative-tier TTFT deficit weight. Higher = favor endpoints with less TTFT violation |
| `negHeadroomTPOTWeight` | 0.2 | [0, inf) | Negative-tier TPOT deficit weight. Set to 0 for non-streaming |
| `affinityGateTauGlobal` | 0.99 | [0, 1] | Global sticky threshold. 0 = disabled. Higher = stricter cache match required |
| `affinityGateTau` | 0.80 | [0, 1] | Within-tier sticky threshold. 0 = disabled |
| `epsilonExploreSticky` | 0.01 | [0, 1] | Probability of skipping affinity gate. Higher = more exploration of non-cached endpoints |
| `epsilonExploreNeg` | 0.01 | [0, 1] | Probability of scoring negative tier. Higher = more traffic to overloaded endpoints |
| `affinityMaxTTFTPenaltyMs` | 5000 | [0, inf) | TTFT penalty threshold (ms) to break stickiness. 0 = always stick |
| `headroomSelectionStrategy` | "least" | least/most/composite-only | Scoring strategy (see Strategies above) |
| `compositeKVWeight` | 1 | [0, inf) | KV cache weight in composite fallback |
| `compositeQueueWeight` | 1 | [0, inf) | Queue depth weight in composite fallback |
| `compositePrefixWeight` | 1 | [0, inf) | Prefix cache weight in composite fallback |

## Dependencies

- Requires `predicted-latency-producer` to populate `LatencyPredictionInfo` on endpoints
- Reads `PrefixCacheMatchInfo` for affinity gating (from `prefix-cache-scorer`)
- Downstream: scores are consumed by the picker (e.g., `weighted-random-picker`)
