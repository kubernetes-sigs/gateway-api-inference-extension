# Predicted Latency Producer (`predicted-latency-producer`)

Trains XGBoost models via a sidecar and generates per-endpoint TTFT/TPOT predictions.

## Interfaces

PrepareDataPlugin, PreRequest, ResponseHeader, ResponseBody, Producer, Consumer

## Responsibilities

- Bulk predictions during `PrepareRequestData` (writes `LatencyPredictionInfo` to endpoint attributes)
- SLO headroom calculation per endpoint: `headroom = SLO - predicted_latency` (used by downstream scorer and admission plugins)
- TTFT training data collection on first token / EOS
- TPOT training data collection at EOS (streaming mode)
- Per-endpoint running request queue tracking (TPOT SLO priority queue)
- Prefix cache score forwarding from `PrefixCacheMatchInfo` attributes
- TPOT neutralization for prefill endpoints in disaggregated serving
- E2E latency metrics when `streamingMode=false`

## Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `samplingMean` | 1000 | Mean interval for decode token sampling |
| `maxDecodeTokenSamplesForPrediction` | 0 | Max tokens to sample for TPOT prediction (0 = disabled) |
| `sloBufferFactor` | 1.0 | Multiplier for SLO headroom calculation |
| `contextTTL` | 5m | TTL for per-request context in the cache |
| `streamingMode` | false | Record TTFT on first chunk (true) vs EOS (false) |
| `endpointRoleLabel` | "" | Label key for disaggregated serving roles |
| `predictInPrepareData` | true | Enable/disable bulk predictions. Set false for training-only mode |

## Default Behavior (`streamingMode: false`)

By default, the system assumes no SLO headers and trains for end-to-end request latency.
TTFT is recorded at EOS and represents the full e2e latency (reported as
`request_e2e_latency_seconds` in metrics). TPOT is not trained because there is no
per-token streaming. The scorer routes based on e2e latency predictions only, with TPOT
automatically neutralized.

## Streaming Mode (`streamingMode: true`)

Set this when clients send `"stream": true` and you want to train separate TTFT (time to
first token) and TPOT (time per output token) models. TTFT is recorded on the first
streaming chunk, and TPOT is sampled across subsequent tokens.

## Disaggregated Serving

Set `endpointRoleLabel` to the label distinguishing prefill from decode pods. TPOT is
automatically neutralized for prefill endpoints (`TPOTValid=true`, `TPOTHeadroom=0`),
ensuring TPOT doesn't affect scoring, admission, or tier classification for prefill pods.

## Files

| File | Purpose |
|------|---------|
| `plugin.go` | Struct, factory, config, per-request context, queue helpers |
| `requestcontrol_hooks.go` | PreRequest, ResponseHeader, ResponseBody hooks |
| `preparedata_hooks.go` | PrepareRequestData, Produces, Consumes |
| `training.go` | buildTrainingEntry, buildPredictionRequest, bulkPredict |
| `prediction.go` | generatePredictions, validatePrediction, TPOT neutralization |
| `decode_token_sampler.go` | Poisson-distributed token sampling for TPOT |
| `running_request_queue.go` | Per-pod request priority queue |
