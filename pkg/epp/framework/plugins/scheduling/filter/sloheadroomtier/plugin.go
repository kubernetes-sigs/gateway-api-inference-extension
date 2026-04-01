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

// Package headroomtier provides a probabilistic filter that selects endpoints
// based on predicted latency headroom (SLO - predicted). It splits endpoints
// into positive (meets SLO) and negative (violates SLO) tiers, and
// probabilistically selects one tier to pass through.
package sloheadroomtier

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrlatency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
)

const (
	PluginType = "slo-headroom-tier-filter"
)

var _ framework.Filter = &Plugin{}

type Config struct {
	// EpsilonExploreNeg is the probability of selecting the negative tier
	// when both tiers have endpoints. This ensures overloaded endpoints get
	// occasional traffic for recovery. Range: [0, 1]. Default: 0.01 (1%).
	EpsilonExploreNeg float64 `json:"epsilonExploreNeg,omitempty"`
}

var DefaultConfig = Config{
	EpsilonExploreNeg: 0.01,
}

type Plugin struct {
	typedName fwkplugin.TypedName
	config    Config
}

func Factory(name string, rawParameters json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := DefaultConfig
	if len(rawParameters) > 0 {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config: %w", err)
		}
	}
	if config.EpsilonExploreNeg < 0 || config.EpsilonExploreNeg > 1.0 {
		return nil, fmt.Errorf("epsilonExploreNeg must be in [0, 1], got %f", config.EpsilonExploreNeg)
	}
	return &Plugin{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		config:    config,
	}, nil
}

func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Filter splits endpoints into positive headroom (meets SLO) and negative
// headroom (violates SLO) tiers. 99% of the time, only positive-tier
// endpoints are kept. 1% of the time, only negative-tier endpoints are kept
// (epsilon exploration for recovery). If only one tier has endpoints, that
// tier is returned. If no endpoints have predictions, all are kept.
func (p *Plugin) Filter(ctx context.Context, _ *framework.CycleState, _ *framework.LLMRequest, endpoints []framework.Endpoint) []framework.Endpoint {
	logger := log.FromContext(ctx)

	if len(endpoints) <= 1 {
		return endpoints
	}

	var positive, negative, noPrediction []framework.Endpoint
	for _, ep := range endpoints {
		raw, ok := ep.Get(attrlatency.LatencyPredictionInfoKey)
		if !ok {
			noPrediction = append(noPrediction, ep)
			continue
		}
		info := raw.(*attrlatency.LatencyPredictionInfo)
		// Validity flags are not checked; invalid predictions are expected to
		// have negative headroom by convention (the predictor sets headroom =
		// SLO - predicted, which is negative when the prediction exceeds SLO).
		if info.TTFTHeadroom() >= 0 && info.TPOTHeadroom() >= 0 {
			positive = append(positive, ep)
		} else {
			negative = append(negative, ep)
		}
	}

	// No predictions available, keep all.
	if len(positive) == 0 && len(negative) == 0 {
		logger.V(logutil.DEBUG).Info("SLOHeadroomTierFilter: no predictions, keeping all",
			"total", len(endpoints))
		return endpoints
	}

	// Endpoints without predictions go to negative tier (can't confirm SLO).
	negative = append(negative, noPrediction...)

	switch {
	case len(positive) > 0 && len(negative) > 0:
		if rand.Float64() < p.config.EpsilonExploreNeg {
			logger.V(logutil.DEBUG).Info("SLOHeadroomTierFilter: epsilon explore, selecting negative tier",
				"positive", len(positive), "negative", len(negative))
			return negative
		}
		logger.V(logutil.DEBUG).Info("SLOHeadroomTierFilter: selecting positive tier",
			"positive", len(positive), "negative", len(negative))
		return positive
	case len(positive) > 0:
		logger.V(logutil.DEBUG).Info("SLOHeadroomTierFilter: only positive tier",
			"positive", len(positive))
		return positive
	default:
		logger.V(logutil.DEBUG).Info("SLOHeadroomTierFilter: only negative tier",
			"negative", len(negative))
		return negative
	}
}

func (p *Plugin) Consumes() map[string]any {
	return map[string]any{
		attrlatency.LatencyPredictionInfoKey: attrlatency.LatencyPredictionInfo{},
	}
}
