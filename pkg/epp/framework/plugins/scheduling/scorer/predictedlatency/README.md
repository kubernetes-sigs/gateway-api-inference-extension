# Predicted Latency Plugins

Composable plugins for ML-based latency prediction and SLO-aware routing.
The system is organized as 3 independent sub-plugins that communicate via endpoint attributes.

## Sub-plugins

### `requestcontrol/requestdataproducer/latencypredictor/` — `latency-predictor`

Trains XGBoost models via a sidecar and generates per-endpoint TTFT/TPOT predictions.

**Location:** `pkg/epp/framework/plugins/requestcontrol/requestdataproducer/latencypredictor/`

**Interfaces:** PrepareDataPlugin, PreRequest, ResponseHeader, ResponseBody, Producer, Consumer

**Responsibilities:**
- Bulk predictions during `PrepareRequestData` (writes `LatencyPredictionInfo` to endpoint attributes)
- TTFT training data collection on first token / EOS
- TPOT training data collection at EOS (streaming mode)
- Per-endpoint running request queue tracking
- Prefix cache score forwarding from `PrefixCacheMatchInfo` attributes
- E2E latency metrics when `streamingMode=false`

**Config:**
- `samplingMean` — mean interval for decode token sampling (default: 1000)
- `maxDecodeTokenSamplesForPrediction` — max tokens to sample for TPOT prediction (default: 0 = disabled)
- `sloBufferFactor` — multiplier for SLO headroom calculation (default: 1.0)
- `contextTTL` — TTL for per-request context in the cache (default: 5m)
- `streamingMode` — record TTFT on first chunk vs EOS (default: false)
- `endpointRoleLabel` — label key for disaggregated serving roles (default: "")
- `predictInPrepareData` — enable/disable bulk predictions (default: true). Set false for training-only mode.

**Default behavior (no SLO, `streamingMode: false`):** By default, the system assumes no
SLO headers are set and trains for end-to-end request latency. TTFT is recorded at EOS and
represents the full e2e latency (reported as `request_e2e_latency_seconds` in metrics). TPOT
is not trained because there is no per-token streaming to measure inter-token latency. The
scorer routes based on e2e latency predictions only, with TPOT automatically neutralized.

**Streaming mode (`streamingMode: true`):** Set this when clients send `"stream": true` and
you want to train separate TTFT (time to first token) and TPOT (time per output token)
models. TTFT is recorded on the first streaming chunk, and TPOT is sampled across
subsequent tokens. Both metrics are used for scoring and routing.

### `scheduling/scorer/latencyscorer/` — `latency-scorer`

Scores endpoints by predicted latency headroom relative to SLO constraints.

**Location:** `pkg/epp/framework/plugins/scheduling/scorer/latencyscorer/`

**Interface:** Scorer

**Scoring flow:**
1. Global affinity gate (tau=0.99) — narrow to near-perfect cache hits
2. Tier split — classify endpoints as positive (meets SLO) or negative headroom
3. Within-tier affinity gate (tau=0.80) — prefer sticky endpoints before normalization
4. Score — headroom-based weighting within the gated, normalized set
5. Tier selection — 99% positive tier, 1% negative (epsilon exploration)

**Strategies:**
- `least` (default) — prefer endpoints with least headroom (tighter packing)
- `most` — prefer endpoints with most headroom (safer margin)
- `composite-only` — use composite scoring (KV cache + queue + prefix) instead of headroom, still with affinity gates and tier logic

**Config:**
- `ttftWeight` / `tpotWeight` — positive-tier headroom blending weights (default: 0.8 / 0.2)
- `negHeadroomTTFTWeight` / `negHeadroomTPOTWeight` — negative-tier deficit blending weights (default: 0.8 / 0.2)
- `affinityGateTauGlobal` — global sticky threshold before tier split (default: 0.99)
- `affinityGateTau` — within-tier sticky threshold before normalization (default: 0.80)
- `epsilonExploreSticky` — probability of skipping affinity gate (default: 0.01)
- `epsilonExploreNeg` — probability of selecting from negative tier when positive exists (default: 0.01)
- `affinityMaxTTFTPenaltyMs` — max TTFT penalty before breaking stickiness (default: 5000)
- `headroomSelectionStrategy` — "least", "most", or "composite-only" (default: "least")
- `compositeKVWeight` / `compositeQueueWeight` / `compositePrefixWeight` — weights for composite scoring fallback (default: 1 / 1 / 1)

### `requestcontrol/admission/latencyadmission/` — `latency-admission`

Rejects sheddable requests when no endpoint can meet SLO constraints.

**Location:** `pkg/epp/framework/plugins/requestcontrol/admission/latencyadmission/`

**Interface:** AdmissionPlugin

**Admit conditions** (any one is sufficient):
- At least one endpoint has a valid prediction (meets SLO)
- At least one endpoint is idle (0 dispatched requests)
- At least one endpoint is cold (<2% KV cache, predictions unreliable)

**Config:** None — the plugin infers validity from the `LatencyPredictionInfo` attributes
set by the predictor (which already neutralizes TPOT when `streamingMode=false`).

## Pipeline

```
Request
  |
  v
PrepareRequestData (latency-predictor)
  |  bulk predictions -> LatencyPredictionInfo attributes
  v
AdmitRequest (latency-admission)
  |  reject sheddable if no valid/idle/cold endpoint
  v
Score (latency-scorer)
  |  affinity gates -> tier split -> headroom scoring
  v
Pick (weighted-random-picker)
  |  A-Res weighted random from scored endpoints
  v
PreRequest (latency-predictor)
  |  dispatch bookkeeping, token counters
  v
ResponseBody (latency-predictor)
     training data collection (TTFT/TPOT)
```

## Disaggregated Serving

For prefill/decode disaggregation, TPOT is automatically neutralized for prefill
endpoints (set `endpointRoleLabel` to the label distinguishing prefill from decode pods).
This ensures TPOT doesn't affect scoring, admission, or tier classification for prefill pods.

## ConfigMap Example

```yaml
plugins:
- type: latency-predictor
- type: latency-scorer
  parameters:
    headroomSelectionStrategy: "least"
    affinityGateTauGlobal: 0.99
    affinityGateTau: 0.80
- type: latency-admission

schedulingProfiles:
- name: default
  plugins:
  - pluginRef: latency-predictor
  - pluginRef: latency-scorer
    weight: 1
  - pluginRef: weighted-random-picker
```
