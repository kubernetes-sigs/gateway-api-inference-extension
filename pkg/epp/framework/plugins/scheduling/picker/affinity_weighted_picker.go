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

package picker

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrlatency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	AffinityWeightedPickerType = "affinity-weighted-picker"
)

// compile-time type validation
var _ framework.Picker = &AffinityWeightedPicker{}

// AffinityWeightedPickerConfig holds configuration for the affinity weighted picker.
type AffinityWeightedPickerConfig struct {
	MaxNumOfEndpoints int `json:"maxNumOfEndpoints"`

	// GlobalTau is the strict affinity threshold (session/conversation affinity).
	// Applied first. Default: 0.99.
	GlobalTau float64 `json:"globalTau,omitempty"`

	// LocalTau is the relaxed affinity threshold (prefix affinity).
	// Applied if global gate finds no matches. Default: 0.80.
	LocalTau float64 `json:"localTau,omitempty"`

	// EpsilonExplore is the probability of ignoring the affinity gate
	// and using all candidates. Range: [0, 1]. Default: 0.01.
	EpsilonExplore float64 `json:"epsilonExplore,omitempty"`

	// MaxTTFTPenaltyMs is the maximum TTFT penalty (ms) tolerated for sticking
	// to a high-affinity endpoint. If the best sticky endpoint's predicted TTFT
	// exceeds the best overall by more than this, stickiness is broken.
	// Set to 0 to disable. Only applies when LatencyPredictionInfo is available
	// on endpoints — no hard dependency on any scorer. Default: 5000.
	MaxTTFTPenaltyMs float64 `json:"maxTTFTPenaltyMs,omitempty"`

	// SelectionMode controls how endpoints are selected after affinity gating.
	// "linear" (default): linear CDF weighted random selection (probability proportional to score).
	// "max": deterministic selection of the highest-scored endpoint.
	SelectionMode string `json:"selectionMode,omitempty"`
}

var defaultAffinityWeightedPickerConfig = AffinityWeightedPickerConfig{
	MaxNumOfEndpoints: DefaultMaxNumOfEndpoints,
	GlobalTau:         0.99,
	LocalTau:          0.80,
	EpsilonExplore:    0.01,
	MaxTTFTPenaltyMs:  5000.0,
	SelectionMode:     "linear",
}

// AffinityWeightedPickerFactory creates a new AffinityWeightedPicker.
func AffinityWeightedPickerFactory(name string, rawParameters json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := defaultAffinityWeightedPickerConfig
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for '%s': %w", AffinityWeightedPickerType, err)
		}
	}
	if config.MaxNumOfEndpoints <= 0 {
		config.MaxNumOfEndpoints = DefaultMaxNumOfEndpoints
	}
	return NewAffinityWeightedPicker(config).WithName(name), nil
}

// NewAffinityWeightedPicker creates a new AffinityWeightedPicker.
func NewAffinityWeightedPicker(config AffinityWeightedPickerConfig) *AffinityWeightedPicker {
	return &AffinityWeightedPicker{
		typedName: fwkplugin.TypedName{Type: AffinityWeightedPickerType, Name: AffinityWeightedPickerType},
		config:    config,
	}
}

// AffinityWeightedPicker combines two-tier prefix cache affinity gating with
// configurable endpoint selection. Scorer-agnostic — works with any scorer.
//
// Pick flow:
//  1. Try global gate (tau=0.99): filter to endpoints with very high prefix match
//  2. If no match, try local gate (tau=0.80): filter to good prefix matches
//  3. Each gate has epsilon-greedy exploration
//  4. Select from the resulting set using configured mode (linear or max)
type AffinityWeightedPicker struct {
	typedName fwkplugin.TypedName
	config    AffinityWeightedPickerConfig
}

func (p *AffinityWeightedPicker) WithName(name string) *AffinityWeightedPicker {
	p.typedName.Name = name
	return p
}

func (p *AffinityWeightedPicker) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Consumes declares that this picker reads prefix cache match info for affinity gating.
func (p *AffinityWeightedPicker) Consumes() map[string]any {
	return map[string]any{
		attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{},
	}
}

// Pick applies two-tier affinity gating then selects using the configured mode.
func (p *AffinityWeightedPicker) Pick(ctx context.Context, _ *framework.CycleState, scoredEndpoints []*framework.ScoredEndpoint) *framework.ProfileRunResult {
	logger := log.FromContext(ctx)

	if len(scoredEndpoints) == 0 {
		return &framework.ProfileRunResult{}
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Apply two-tier affinity gating.
	candidates := p.affinityGate(ctx, scoredEndpoints, rng)

	logger.V(logutil.DEBUG).Info("AffinityWeightedPicker: after gating",
		"candidates", len(candidates), "total", len(scoredEndpoints),
		"selectionMode", p.config.SelectionMode)

	// Select from gated candidates.
	if p.config.SelectionMode == "max" {
		return maxSelect(candidates, p.config.MaxNumOfEndpoints)
	}
	return linearWeightedSelect(candidates, p.config.MaxNumOfEndpoints, rng)
}

// affinityGate applies two-tier prefix cache affinity gating.
// Returns the filtered set, or all endpoints if no gate applies.
func (p *AffinityWeightedPicker) affinityGate(
	ctx context.Context,
	endpoints []*framework.ScoredEndpoint,
	rng *rand.Rand,
) []*framework.ScoredEndpoint {
	logger := log.FromContext(ctx)

	// Try global gate first (strict, tau=0.99).
	if p.config.GlobalTau > 0 {
		if result := p.tryGate(ctx, endpoints, rng, p.config.GlobalTau, "global"); result != nil {
			logger.V(logutil.DEBUG).Info("AffinityWeightedPicker: global gate applied",
				"tau", p.config.GlobalTau, "eligible", len(result))
			return result
		}
	}

	// Try local gate (relaxed, tau=0.80).
	if p.config.LocalTau > 0 {
		if result := p.tryGate(ctx, endpoints, rng, p.config.LocalTau, "local"); result != nil {
			logger.V(logutil.DEBUG).Info("AffinityWeightedPicker: local gate applied",
				"tau", p.config.LocalTau, "eligible", len(result))
			return result
		}
	}

	// No gate applied — use all endpoints.
	return endpoints
}

// tryGate attempts to apply an affinity gate at the given tau threshold.
// Returns the filtered set if the gate applies, nil otherwise.
func (p *AffinityWeightedPicker) tryGate(
	ctx context.Context,
	endpoints []*framework.ScoredEndpoint,
	rng *rand.Rand,
	tau float64,
	label string,
) []*framework.ScoredEndpoint {
	logger := log.FromContext(ctx)

	// Filter to endpoints with prefix cache score >= tau.
	eligible := make([]*framework.ScoredEndpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		if prefixScore(ep) >= tau {
			eligible = append(eligible, ep)
		}
	}

	if len(eligible) == 0 {
		return nil // no match at this threshold
	}

	// Epsilon exploration: ignore gate with some probability.
	if rng.Float64() < p.config.EpsilonExplore {
		logger.V(logutil.DEBUG).Info("AffinityWeightedPicker: epsilon explore, ignoring gate",
			"gate", label, "epsilon", p.config.EpsilonExplore)
		return nil // fall through to next gate or all endpoints
	}

	// Opportunistic TTFT load gate: if LatencyPredictionInfo is available on
	// endpoints, check whether the best sticky endpoint's TTFT is too much
	// worse than the best overall. No hard dependency — if predictions aren't
	// present, the gate is simply skipped.
	if p.config.MaxTTFTPenaltyMs > 0 {
		bestAll := bestTTFT(endpoints)
		bestSticky := bestTTFT(eligible)

		if bestAll < math.MaxFloat64 && bestSticky < math.MaxFloat64 {
			penalty := bestSticky - bestAll
			if penalty > p.config.MaxTTFTPenaltyMs {
				logger.V(logutil.DEBUG).Info("AffinityWeightedPicker: TTFT penalty too high, breaking stickiness",
					"gate", label,
					"bestStickyTTFT", bestSticky,
					"bestOverallTTFT", bestAll,
					"penaltyMs", penalty,
					"maxPenaltyMs", p.config.MaxTTFTPenaltyMs)
				return nil // fall through
			}
		}
	}

	return eligible
}

// maxSelect picks the top-k endpoints by score.
func maxSelect(endpoints []*framework.ScoredEndpoint, maxCount int) *framework.ProfileRunResult {
	candidates := make([]*framework.ScoredEndpoint, len(endpoints))
	copy(candidates, endpoints)
	shuffleScoredEndpoints(candidates) // random tie-break

	slices.SortStableFunc(candidates, func(i, j *framework.ScoredEndpoint) int {
		if i.Score > j.Score {
			return -1
		}
		if i.Score < j.Score {
			return 1
		}
		return 0
	})

	if maxCount < len(candidates) {
		candidates = candidates[:maxCount]
	}

	targetEndpoints := make([]framework.Endpoint, len(candidates))
	for i, se := range candidates {
		targetEndpoints[i] = se
	}
	return &framework.ProfileRunResult{TargetEndpoints: targetEndpoints}
}

// linearWeightedSelect performs linear CDF weighted random selection.
func linearWeightedSelect(endpoints []*framework.ScoredEndpoint, maxCount int, rng *rand.Rand) *framework.ProfileRunResult {
	if len(endpoints) == 0 {
		return &framework.ProfileRunResult{}
	}

	// Work on a copy to avoid mutating the caller's slice.
	candidates := make([]*framework.ScoredEndpoint, len(endpoints))
	copy(candidates, endpoints)
	shuffleScoredEndpoints(candidates)

	targetEndpoints := make([]framework.Endpoint, 0, maxCount)

	for range maxCount {
		if len(candidates) == 0 {
			break
		}

		total := 0.0
		for _, se := range candidates {
			if se.Score > 0 {
				total += se.Score
			}
		}

		if total <= 0 {
			idx := rng.Intn(len(candidates))
			targetEndpoints = append(targetEndpoints, candidates[idx])
			candidates = removeIdx(candidates, idx)
			continue
		}

		val := rng.Float64() * total
		selected := len(candidates) - 1
		for i, se := range candidates {
			if se.Score <= 0 {
				continue
			}
			val -= se.Score
			if val < 0 {
				selected = i
				break
			}
		}

		targetEndpoints = append(targetEndpoints, candidates[selected])
		candidates = removeIdx(candidates, selected)
	}

	return &framework.ProfileRunResult{TargetEndpoints: targetEndpoints}
}

// removeIdx removes element at index i without preserving order.
func removeIdx(s []*framework.ScoredEndpoint, i int) []*framework.ScoredEndpoint {
	s[i] = s[len(s)-1]
	return s[:len(s)-1]
}

// bestTTFT returns the lowest positive predicted TTFT across endpoints.
// Returns math.MaxFloat64 if no endpoint has LatencyPredictionInfo.
func bestTTFT(endpoints []*framework.ScoredEndpoint) float64 {
	best := math.MaxFloat64
	for _, ep := range endpoints {
		raw, ok := ep.Get(attrlatency.LatencyPredictionInfoKey)
		if !ok {
			continue
		}
		info := raw.(*attrlatency.LatencyPredictionInfo)
		ttft := info.TTFT()
		if ttft > 0 && ttft < best {
			best = ttft
		}
	}
	return best
}

// prefixScore reads the prefix cache score from an endpoint's attributes.
func prefixScore(se *framework.ScoredEndpoint) float64 {
	raw, ok := se.Get(attrprefix.PrefixCacheMatchInfoKey)
	if !ok {
		return 0
	}
	info := raw.(*attrprefix.PrefixCacheMatchInfo)
	total := info.TotalBlocks()
	if total == 0 {
		return 0
	}
	score := float64(info.MatchBlocks()) / float64(total)
	if math.IsNaN(score) {
		return 0
	}
	return score
}
