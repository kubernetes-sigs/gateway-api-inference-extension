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

// Package vtc implements a fairness policy based on Virtual Token Counting (VTC).
//
// Unlike round-robin (which gives each flow one turn regardless of request size), VTC tracks the
// cumulative "virtual work" dispatched per flow. The flow with the lowest virtual counter gets the
// next dispatch opportunity. This ensures that a tenant sending large prompts does not receive the
// same number of dispatch turns as one sending small prompts.
//
// For detailed documentation, see README.md.
package vtc

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// VTCFairnessPolicyType is the registration type for the VTC fairness policy.
const VTCFairnessPolicyType = "vtc-fairness-policy"

// normalizationThreshold is the value at which counters are normalized to prevent unbounded growth and float64
// precision loss. When any counter exceeds this, the global minimum is subtracted from all counters.
const normalizationThreshold = 1e12

// VTCFairnessPolicyFactory creates a VTC fairness policy from optional JSON parameters.
//
// Supported parameters:
//
//	{
//	  "weights": {"tenant-a": 2.0, "tenant-b": 1.0},
//	  "defaultWeight": 1.0
//	}
//
// - weights: per-flow-ID weight overrides. Higher weight = larger share of dispatch opportunities.
// - defaultWeight: weight for flows not listed in weights. Defaults to 1.0 if omitted or <= 0.
func VTCFairnessPolicyFactory(name string, params json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return newVTC(name, params)
}

// vtcConfig holds the JSON-parsed configuration for the VTC policy.
type vtcConfig struct {
	Weights       map[string]float64 `json:"weights"`
	DefaultWeight float64            `json:"defaultWeight"`
}

// vtc implements FairnessPolicy using Virtual Token Counting.
// The struct is immutable after construction and shared across all priority bands (Singleton).
type vtc struct {
	name          string
	weights       map[string]float64
	defaultWeight float64
}

func newVTC(name string, params json.RawMessage) (*vtc, error) {
	if name == "" {
		name = VTCFairnessPolicyType
	}

	cfg := vtcConfig{
		DefaultWeight: 1.0,
	}

	if len(params) > 0 {
		if err := json.Unmarshal(params, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse VTC parameters: %w", err)
		}
	}

	if cfg.DefaultWeight <= 0 {
		cfg.DefaultWeight = 1.0
	}

	for id, w := range cfg.Weights {
		if w <= 0 {
			return nil, fmt.Errorf("weight for flow %q must be positive, got %f", id, w)
		}
	}

	return &vtc{
		name:          name,
		weights:       cfg.Weights,
		defaultWeight: cfg.DefaultWeight,
	}, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *vtc) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: VTCFairnessPolicyType,
		Name: p.name,
	}
}

// vtcState holds the mutable per-band state for the VTC policy (Flyweight pattern).
type vtcState struct {
	mu       sync.Mutex
	counters map[string]float64 // flow ID -> cumulative virtual work
}

// NewState initializes the policy state for a specific priority band.
func (p *vtc) NewState(_ context.Context) any {
	return &vtcState{
		counters: make(map[string]float64),
	}
}

// Pick selects the flow with the lowest virtual counter from the given priority band.
//
// Algorithm (Weighted Fair Queuing):
//  1. For each non-empty flow queue, look up (or initialize) its virtual counter.
//  2. Select the flow with the smallest counter. Ties are broken by deterministic FlowKey ordering.
//  3. Advance the winner's counter by cost/weight, where cost is the head item's ByteSize.
//  4. Prune counters for flows no longer in the active set and normalize if needed.
func (p *vtc) Pick(
	_ context.Context,
	flowGroup flowcontrol.PriorityBandAccessor,
) (flowcontrol.FlowQueueAccessor, error) {
	if flowGroup == nil {
		return nil, nil
	}

	v := flowGroup.PolicyState()
	s, ok := v.(*vtcState)
	if !ok {
		return nil, fmt.Errorf("invalid state type for VTC policy: expected *vtcState, got %T", v)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	keys := flowGroup.FlowKeys()
	if len(keys) == 0 {
		return nil, nil
	}

	// Sort for deterministic tie-breaking.
	slices.SortFunc(keys, func(a, b flowcontrol.FlowKey) int { return a.Compare(b) })

	// Find the current global minimum counter among tracked flows.
	globalMin := math.MaxFloat64
	for _, c := range s.counters {
		if c < globalMin {
			globalMin = c
		}
	}
	if globalMin == math.MaxFloat64 {
		globalMin = 0
	}

	// Find the non-empty flow with the lowest virtual counter.
	var bestQueue flowcontrol.FlowQueueAccessor
	bestCounter := math.MaxFloat64
	activeIDs := make(map[string]struct{}, len(keys))

	for _, key := range keys {
		activeIDs[key.ID] = struct{}{}

		queue := flowGroup.Queue(key.ID)
		if queue == nil || queue.Len() == 0 {
			continue
		}

		// Initialize counter for new flows to the current global minimum.
		if _, exists := s.counters[key.ID]; !exists {
			s.counters[key.ID] = globalMin
		}

		counter := s.counters[key.ID]
		if counter < bestCounter {
			bestCounter = counter
			bestQueue = queue
		}
	}

	if bestQueue == nil {
		return nil, nil
	}

	// Advance the winner's counter: counter += cost / weight.
	winnerID := bestQueue.FlowKey().ID
	var cost float64
	if head := bestQueue.PeekHead(); head != nil {
		if req := head.OriginalRequest(); req != nil {
			cost = float64(req.ByteSize())
		}
	}
	s.counters[winnerID] += cost / p.weightFor(winnerID)

	// Prune counters for flows no longer in the active set.
	for id := range s.counters {
		if _, active := activeIDs[id]; !active {
			delete(s.counters, id)
		}
	}

	// Normalize counters if any value exceeds the threshold.
	normalizeCounters(s)

	return bestQueue, nil
}

// weightFor returns the configured weight for the given flow ID, or the default weight.
func (p *vtc) weightFor(flowID string) float64 {
	if w, ok := p.weights[flowID]; ok {
		return w
	}
	return p.defaultWeight
}

// normalizeCounters subtracts the global minimum from all counters when any counter exceeds the threshold.
// This preserves relative differences while preventing unbounded growth.
func normalizeCounters(s *vtcState) {
	var maxCounter float64
	for _, c := range s.counters {
		if c > maxCounter {
			maxCounter = c
		}
	}

	if maxCounter <= normalizationThreshold {
		return
	}

	minCounter := math.MaxFloat64
	for _, c := range s.counters {
		if c < minCounter {
			minCounter = c
		}
	}

	for id := range s.counters {
		s.counters[id] -= minCounter
	}
}
