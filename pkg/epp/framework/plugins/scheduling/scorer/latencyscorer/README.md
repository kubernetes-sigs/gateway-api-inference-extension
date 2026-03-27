# Latency Scorer Plugin (`latency-scorer`)

Scores endpoints by predicted latency headroom relative to SLO constraints.

## Interface

Scorer

## Scoring Flow

1. **Global affinity gate** (tau=0.99) — narrow to near-perfect cache hits
2. **Tier split** — classify endpoints as positive (meets SLO) or negative headroom
3. **Within-tier affinity gate** (tau=0.80) — prefer sticky endpoints before normalization
4. **Score** — headroom-based weighting within the gated, normalized set
5. **Tier selection** — 99% positive tier, 1% negative (epsilon exploration)

Non-sticky endpoints get score=0 and are excluded by the picker.

## Strategies

| Strategy | Behavior |
|----------|----------|
| `least` (default) | Prefer endpoints with least headroom (tighter packing) |
| `most` | Prefer endpoints with most headroom (safer margin) |
| `composite-only` | Use composite scoring (KV cache + queue + prefix) instead of headroom, still with affinity gates and tier logic |

## Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ttftWeight` | 0.8 | Positive-tier TTFT blending weight |
| `tpotWeight` | 0.2 | Positive-tier TPOT blending weight |
| `negHeadroomTTFTWeight` | 0.8 | Negative-tier TTFT deficit weight |
| `negHeadroomTPOTWeight` | 0.2 | Negative-tier TPOT deficit weight |
| `affinityGateTauGlobal` | 0.99 | Global sticky threshold before tier split |
| `affinityGateTau` | 0.80 | Within-tier sticky threshold before normalization |
| `epsilonExploreSticky` | 0.01 | Probability of skipping affinity gate |
| `epsilonExploreNeg` | 0.01 | Probability of selecting from negative tier |
| `affinityMaxTTFTPenaltyMs` | 5000 | Max TTFT penalty (ms) before breaking stickiness |
| `headroomSelectionStrategy` | "least" | "least", "most", or "composite-only" |
| `compositeKVWeight` | 1 | KV cache weight for composite scoring |
| `compositeQueueWeight` | 1 | Queue depth weight for composite scoring |
| `compositePrefixWeight` | 1 | Prefix cache weight for composite scoring |

## Design Notes

- **Affinity gate before normalization**: The within-tier gate narrows to sticky endpoints
  before normalizing headroom, so scores are relative to the sticky subset. This prevents
  non-sticky endpoints from compressing the score contrast.
- **Range-based re-normalization**: When one dimension (TTFT or TPOT) has zero range across
  all endpoints, alpha/beta weights are automatically re-normalized so the zero-range
  dimension doesn't compress scores.
- **Per-bucket normalization**: In the negative tier, each hierarchical deficit bucket
  (both-neg, ttft-only-neg, tpot-only-neg) normalizes independently.
- **Idle pod preference**: In the negative tier, endpoints with zero dispatched requests are
  strongly preferred (busy endpoints get score=0 when idle ones exist).
