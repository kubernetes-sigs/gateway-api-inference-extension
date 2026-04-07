/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package latency provides a scorer that scores endpoints based on predicted
// latency headroom. Designed to run after prefix-cache-affinity and
// headroom-tier filters have narrowed the candidate set.
package latency

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrlatency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	LatencyScorerType = "latency-scorer"
	wMax              = 100
	minWeight         = 0
	eps               = 1e-9
)

// HeadroomSelectionStrategy controls how headroom is mapped to scores.
type HeadroomSelectionStrategy string

const (
	// StrategyLeast prefers endpoints closest to SLO (bin-packing to preserve
	// capacity on other endpoints for future requests).
	StrategyLeast HeadroomSelectionStrategy = "least"
	// StrategyMost prefers endpoints with the most headroom (conservative
	// approach for max safety margin).
	StrategyMost HeadroomSelectionStrategy = "most"
)

// compile-time validation
var _ framework.Scorer = &Plugin{}

type Config struct {
	// TTFTWeight controls the relative importance of TTFT headroom when scoring.
	// Headroom = SLO - predicted. When no SLO is set, headroom is always negative
	// but relative ordering still differentiates endpoints.
	// Range: [0, inf). Higher = favor endpoints with lower predicted TTFT.
	// Default: 0.8.
	TTFTWeight float64 `json:"ttftWeight,omitempty"`

	// TPOTWeight controls the relative importance of TPOT headroom when scoring.
	// Range: [0, inf). Set to 0 for non-streaming workloads.
	// Default: 0.2.
	TPOTWeight float64 `json:"tpotWeight,omitempty"`

	// HeadroomSelectionStrategy controls how headroom is mapped to scores.
	// See StrategyLeast and StrategyMost constants. Default: "least".
	HeadroomSelectionStrategy HeadroomSelectionStrategy `json:"headroomSelectionStrategy,omitempty"`

	// Composite scoring weights used as fallback when no predictions are
	// available (sidecar down or timed out).
	CompositeKVWeight     float64 `json:"compositeKVWeight,omitempty"`
	CompositeQueueWeight  float64 `json:"compositeQueueWeight,omitempty"`
	CompositePrefixWeight float64 `json:"compositePrefixWeight,omitempty"`
}

var DefaultConfig = Config{
	TTFTWeight:                0.8,
	TPOTWeight:                0.2,
	HeadroomSelectionStrategy: StrategyLeast,
	CompositeKVWeight:         1,
	CompositeQueueWeight:      1,
	CompositePrefixWeight:     1,
}

// Plugin scores endpoints based on predicted latency headroom.
//
// It expects prefix-cache-affinity and headroom-tier filters to have already
// narrowed the candidate set. The scorer handles:
//   - Idle pod preference (zero dispatched requests preferred)
//   - Hierarchical deficit bucketing (both-neg > ttft-only-neg > tpot-only-neg)
//   - Headroom normalization and blending
//   - Range-based weight re-normalization when one dimension has zero range
//   - Composite fallback when no predictions available
type Plugin struct {
	typedName fwkplugin.TypedName
	config    Config
}

// NewPlugin creates a Plugin with the given config. Used for testing.
func NewPlugin(config Config) *Plugin {
	return &Plugin{
		typedName: fwkplugin.TypedName{Type: LatencyScorerType, Name: LatencyScorerType},
		config:    config,
	}
}

func Factory(name string, rawParameters json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := DefaultConfig
	if len(rawParameters) > 0 {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config: %w", err)
		}
	}
	return &Plugin{
		typedName: fwkplugin.TypedName{Type: LatencyScorerType, Name: name},
		config:    config,
	}, nil
}

func (s *Plugin) TypedName() fwkplugin.TypedName {
	return s.typedName
}

func (s *Plugin) Category() framework.ScorerCategory {
	return framework.Balance
}

// epData holds per-endpoint data gathered from attributes.
type epData struct {
	endpoint     framework.Endpoint
	info         *attrlatency.LatencyPredictionInfo
	ttftHeadroom float64 // cached from info.TTFTHeadroom()
	tpotHeadroom float64 // cached from info.TPOTHeadroom()
}

// Score returns a float64 score in [0,1] for each endpoint.
func (s *Plugin) Score(ctx context.Context, _ *framework.CycleState, _ *framework.InferenceRequest, endpoints []framework.Endpoint) map[framework.Endpoint]float64 {
	logger := log.FromContext(ctx)
	scores := make(map[framework.Endpoint]float64, len(endpoints))
	for _, ep := range endpoints {
		scores[ep] = 0
	}

	// Gather prediction data from endpoint attributes.
	data := make([]epData, 0, len(endpoints))
	hasPredictions := false
	for _, ep := range endpoints {
		d := epData{endpoint: ep}
		if raw, ok := ep.Get(attrlatency.LatencyPredictionInfoKey); ok {
			info := raw.(*attrlatency.LatencyPredictionInfo)
			d.info = info
			d.ttftHeadroom = info.TTFTHeadroom()
			d.tpotHeadroom = info.TPOTHeadroom()
			hasPredictions = true
		}
		data = append(data, d)
	}

	// No predictions: composite fallback.
	if !hasPredictions {
		logger.V(logutil.DEBUG).Info("LatencyScorer: no predictions, using composite fallback")
		return s.compositeScores(ctx, endpoints)
	}

	// Separate positive and negative headroom endpoints.
	// The scorer assumes homogeneous input from upstream filters, but if a mix
	// is present, only positive endpoints are scored (negative get score 0).
	var positive, negative []epData
	for _, d := range data {
		if d.info != nil && (d.ttftHeadroom < 0 || d.tpotHeadroom < 0) {
			negative = append(negative, d)
		} else {
			positive = append(positive, d)
		}
	}

	if len(positive) > 0 {
		s.scoreBucket(ctx, positive, scores, false)
		return scores
	}

	// All negative: apply idle preference.
	// If any endpoint has zero dispatched requests, only score idle endpoints.
	var idle []epData
	for _, d := range negative {
		if d.info != nil && d.info.DispatchedRequestCount() == 0 {
			idle = append(idle, d)
		}
	}
	if len(idle) > 0 {
		logger.V(logutil.DEBUG).Info("LatencyScorer: idle endpoints found, scoring only idle",
			"idle", len(idle), "total", len(negative))
		s.scoreBucket(ctx, idle, scores, true)
		return scores
	}

	// Deficit bucketing: group by which SLOs are violated, score the best
	// (least severe) non-empty bucket. TTFT violations are treated as more
	// severe because TTFT directly impacts perceived responsiveness.
	var negTPOTonly, negTTFTonly, bothNeg []epData
	for _, d := range negative {
		if d.info == nil {
			bothNeg = append(bothNeg, d)
			continue
		}
		ttftNeg := d.ttftHeadroom < 0
		tpotNeg := d.tpotHeadroom < 0
		switch {
		case ttftNeg && tpotNeg:
			bothNeg = append(bothNeg, d)
		case ttftNeg:
			negTTFTonly = append(negTTFTonly, d)
		case tpotNeg:
			negTPOTonly = append(negTPOTonly, d)
		default:
			bothNeg = append(bothNeg, d)
		}
	}

	// Score the best non-empty bucket. Force "least" strategy for negative
	// headroom: "least" deficit = closest to meeting SLO.
	buckets := [][]epData{negTPOTonly, negTTFTonly, bothNeg}
	for _, bucket := range buckets {
		if len(bucket) > 0 {
			s.scoreBucket(ctx, bucket, scores, true)
			return scores
		}
	}

	// Fallback: score all negative.
	s.scoreBucket(ctx, negative, scores, true)

	return scores
}

// scoreBucket normalizes headroom within a bucket and assigns scores.
// If forceLeast is true, the "least" strategy is used regardless of config
// (needed for negative headroom where "most" would incorrectly prefer the most overloaded).
func (s *Plugin) scoreBucket(ctx context.Context, data []epData, scores map[framework.Endpoint]float64, forceLeast bool) {
	logger := log.FromContext(ctx)

	alpha, beta := normalizedWeights(s.config.TTFTWeight, s.config.TPOTWeight)

	// Compute min/max for normalization.
	minTTFT, maxTTFT := math.MaxFloat64, -math.MaxFloat64
	minTPOT, maxTPOT := math.MaxFloat64, -math.MaxFloat64
	for _, d := range data {
		h := math.Abs(d.ttftHeadroom)
		if h < minTTFT {
			minTTFT = h
		}
		if h > maxTTFT {
			maxTTFT = h
		}
		h = math.Abs(d.tpotHeadroom)
		if h < minTPOT {
			minTPOT = h
		}
		if h > maxTPOT {
			maxTPOT = h
		}
	}

	ttftRange := maxTTFT - minTTFT
	tpotRange := maxTPOT - minTPOT

	// Range-based weight re-normalization.
	// If one dimension has zero range (all values identical), set its weight
	// to 0 so it doesn't compress scores to a single value.
	if ttftRange <= eps && tpotRange > eps {
		alpha, beta = 0.0, 1.0
	} else if tpotRange <= eps && ttftRange > eps {
		alpha, beta = 1.0, 0.0
	}

	for _, d := range data {
		var nTTFT, nTPOT float64
		if ttftRange > eps {
			nTTFT = (math.Abs(d.ttftHeadroom) - minTTFT) / ttftRange
		} else {
			nTTFT = 0.5
		}
		if tpotRange > eps {
			nTPOT = (math.Abs(d.tpotHeadroom) - minTPOT) / tpotRange
		} else {
			nTPOT = 0.5
		}

		combined := alpha*nTTFT + beta*nTPOT

		var w float64
		strategy := s.config.HeadroomSelectionStrategy
		if forceLeast {
			strategy = StrategyLeast
		}
		switch strategy {
		case StrategyMost:
			// Prefer endpoints with the most headroom (conservative approach for max safety margin).
			w = float64(int(combined*float64(wMax-minWeight)) + minWeight + 1)
		default: // "least"
			// Prefer endpoints closest to the SLO (bin-packing approach to preserve fully idle nodes).
			w = float64(int((1.0-combined)*float64(wMax-minWeight)) + minWeight + 1)
		}
		scores[d.endpoint] = w / float64(wMax)

		logger.V(logutil.TRACE).Info("LatencyScorer: scored endpoint",
			"endpoint", d.endpoint.GetMetadata().NamespacedName.Name,
			"ttftHeadroom", d.ttftHeadroom, "tpotHeadroom", d.tpotHeadroom,
			"nTTFT", nTTFT, "nTPOT", nTPOT, "combined", combined, "score", scores[d.endpoint])
	}
}

// compositeScores returns scores based on KV cache, queue, and prefix cache.
// This is a fallback for when latency predictions are unavailable (sidecar down
// or timed out).
func (s *Plugin) compositeScores(ctx context.Context, endpoints []framework.Endpoint) map[framework.Endpoint]float64 {
	scores := make(map[framework.Endpoint]float64, len(endpoints))

	wkv, wq, wpref := s.config.CompositeKVWeight, s.config.CompositeQueueWeight, s.config.CompositePrefixWeight
	sumw := wkv + wq + wpref
	if sumw <= 0 {
		wkv, wq, wpref = 1, 0, 0
		sumw = 1
	}
	wkv /= sumw
	wq /= sumw
	wpref /= sumw

	// Find max queue for relative scoring.
	maxQ := 0
	for _, ep := range endpoints {
		if q := ep.GetMetrics().WaitingQueueSize; q > maxQ {
			maxQ = q
		}
	}
	qRange := float64(maxQ)

	logger := log.FromContext(ctx)
	for _, ep := range endpoints {
		q := ep.GetMetrics().WaitingQueueSize
		relQueue := 1.0
		if qRange > 0 {
			relQueue = float64(maxQ-q) / qRange
		}

		kvFree := 1.0 - ep.GetMetrics().KVCacheUsagePercent
		prefix := prefixCacheScore(ep)

		composite := wkv*kvFree + wq*relQueue + wpref*prefix
		w := int(math.Round(float64(minWeight) + float64(wMax-minWeight)*composite))
		score := float64(w) / float64(wMax)

		scores[ep] = score
		logger.V(logutil.TRACE).Info("LatencyScorer: composite",
			"endpoint", ep.GetMetadata().NamespacedName.Name,
			"kvFree", kvFree, "relQueue", relQueue, "prefix", prefix, "score", score)
	}

	return scores
}

func (s *Plugin) Consumes() map[string]any {
	return map[string]any{
		attrlatency.LatencyPredictionInfoKey: attrlatency.LatencyPredictionInfo{},
		attrprefix.PrefixCacheMatchInfoKey:   attrprefix.PrefixCacheMatchInfo{},
	}
}

func normalizedWeights(a, b float64) (float64, float64) {
	sum := a + b
	if sum <= 0 {
		return 1.0, 0.0
	}
	return a / sum, b / sum
}

func prefixCacheScore(ep framework.Endpoint) float64 {
	if raw, ok := ep.Get(attrprefix.PrefixCacheMatchInfoKey); ok {
		info := raw.(*attrprefix.PrefixCacheMatchInfo)
		if info.TotalBlocks() > 0 {
			score := float64(info.MatchBlocks()) / float64(info.TotalBlocks())
			if !math.IsNaN(score) {
				return score
			}
		}
	}
	return 0
}
