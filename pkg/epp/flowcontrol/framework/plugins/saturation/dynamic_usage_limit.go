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

package saturation

import (
	"context"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"k8s.io/utils/clock"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

const (
	// DynamicUsageLimitPolicyType is the type of the dynamic usage limit policy plugin.
	DynamicUsageLimitPolicyType = "dynamic-usage-limit-policy"
)

// Private const values. These might evolve into config.
const (
	// defaultTargetSaturation is the saturation threshold above which we start throttling.
	// When saturation exceeds this value, limits decrease proportionally to the overshoot.
	defaultTargetSaturation = 0.8

	// lowPriorityThrottleScaleFactor multiplies the throttle adjustment for low-priority traffic (priority < 0).
	// A value of 2.0 means low-priority traffic is throttled twice as aggressively as high-priority traffic.
	lowPriorityThrottleScaleFactor = 2.0

	// idleTimeThreshold is the duration after which we start applying decay to limits
	idleTimeThreshold = 5 * time.Second

	// decayRate is how much we recover toward 1.0 per idle period
	decayRate = 0.05

	// sampleWindowDuration is how long we keep saturation samples for trend analysis
	sampleWindowDuration = 10 * time.Second

	// trendProjectionSeconds projects per-second saturation rate to this time horizon
	// for evaluating trend severity. A 10-second projection means we ask: "if this rate
	// continues for 10 seconds, how much would saturation change?"
	trendProjectionSeconds = 10.0
)

// DynamicUsagePolicy dynamically adjusts usage limits per priority level based on saturation levels and trends.
// It attempts to maintain saturation near a target threshold by throttling low-priority traffic more aggressively
// when saturation is high or rising, and recovering limits when saturation is low or falling.
type DynamicUsagePolicy struct {
	name  string
	clock clock.Clock

	// mu protects the saturations map.
	mu          sync.Mutex
	saturations map[string][]saturationSample

	// mul protects the usageLimits map.
	mul         sync.Mutex
	usageLimits map[string]usageLimitEntry
}

// usageLimitEntry tracks the limit and when it was last updated for decay purposes.
type usageLimitEntry struct {
	limit      float64
	lastUpdate time.Time
}

// saturationSample represents a single saturation measurement at a point in time.
type saturationSample struct {
	saturation float64
	timestamp  time.Time
}

var _ flowcontrol.UsageLimitPolicy = &DynamicUsagePolicy{}

// NewDynamicUsagePolicy creates a new dynamic usage limit policy.
// The clock parameter is used for time-based operations (decay, trend tracking).
func NewDynamicUsagePolicy(clk clock.Clock) *DynamicUsagePolicy {
	return &DynamicUsagePolicy{
		name:        DynamicUsageLimitPolicyType,
		clock:       clk,
		saturations: make(map[string][]saturationSample),
		usageLimits: make(map[string]usageLimitEntry),
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *DynamicUsagePolicy) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: DynamicUsageLimitPolicyType,
		Name: p.name,
	}
}

// ComputeLimit computes the dynamic usage limit for a given priority level based on current saturation
// and saturation trends. The limit adjusts over time to maintain saturation near the target threshold
// while being more aggressive with low-priority (priority < 0) traffic.
//
// The algorithm:
//  1. Applies decay to recover limits that have been idle (no requests for > idleTimeThreshold)
//  2. Computes proportional adjustment based on distance from target saturation
//  3. Scales adjustment by saturation trend (rising = more aggressive throttle, falling = gentler)
//  4. Applies priority-based multiplier (low priority gets throttled more aggressively)
//  5. Clamps adjustment and final limit to safe bounds
//
// Returns a limit in [0.0, 1.0] where 1.0 means no gating and 0.0 means fully gated.
func (p *DynamicUsagePolicy) ComputeLimit(
	ctx context.Context, priority int, saturation float64, requestMetadata map[string]any) (limit float64) {
	saturationTrend := p.saturationTrend(requestMetadata, saturation)
	p.mul.Lock()
	defer p.mul.Unlock()

	now := p.clock.Now()
	cacheKey := generateUsageLimitCacheKey(requestMetadata, priority)
	entry, ok := p.usageLimits[cacheKey]
	u := 1.0 // Default usage limit
	if ok {
		u = entry.limit
		// If this priority has been idle, and we are below saturation,
		// then apply decay.
		idleTime := now.Sub(entry.lastUpdate)
		if idleTime > idleTimeThreshold && saturation < defaultTargetSaturation {
			// Gradually recover toward 1.0
			u = min(1.0, u+decayRate)
		}
	}

	var adjustment float64
	if saturation > defaultTargetSaturation {
		// We need to throttle, how far over target are we?
		overshoot := saturation - defaultTargetSaturation
		// Base adjustment proportional to overshoot
		baseAdjustment := -0.5 * overshoot

		// Scale by trend: if rising, more aggressive
		// saturationTrend is per-second rate; project to configured time horizon for comparison
		projectedDelta := saturationTrend * trendProjectionSeconds

		var trendMultiplier float64
		if projectedDelta > 0 {
			// getting worse, let's throttle more aggressively
			trendMultiplier = 1.0 + min(projectedDelta, 0.5) // up to 1.5x if rising fast
		} else {
			trendMultiplier = 0.5 // gentler if improving
		}

		adjustment = baseAdjustment * trendMultiplier

		// low priority gets throttled more
		if priority < 0 {
			adjustment *= lowPriorityThrottleScaleFactor
		}
	} else {
		// Under target - proportional recovery
		headroom := defaultTargetSaturation - saturation
		adjustment = 0.3 * headroom // faster recovery when further under target
	}

	// Clamp adjustment to avoid wild swings
	adjustment = max(-0.3, min(0.3, adjustment))
	u += adjustment
	// clamp within [0,1]
	u = max(0.0, min(1.0, u))

	p.usageLimits[cacheKey] = usageLimitEntry{
		limit:      u,
		lastUpdate: now,
	}
	return u
}

// saturationTrend computes the rate of change of saturation over a time window for the given endpoint subset.
// It maintains a rolling window of saturation samples (configured by sampleWindowDuration) and calculates
// the rate of change per second.
//
// Returns:
//   - 0.0 if less than 2 samples exist or the time distance within samples is too small
//   - Slope of the fitted line in saturation units per second (can be positive, negative, or zero)
func (p *DynamicUsagePolicy) saturationTrend(requestMetadata map[string]any, saturation float64) float64 {
	key := generateCacheKey(requestMetadata)

	p.mu.Lock()
	defer p.mu.Unlock()

	now := p.clock.Now()
	cutoff := now.Add(-sampleWindowDuration)

	samples, found := p.saturations[key]
	if found {
		// Filter out old samples
		validSamples := make([]saturationSample, 0, len(samples)+1)
		for _, s := range samples {
			if s.timestamp.After(cutoff) {
				validSamples = append(validSamples, s)
			}
		}
		samples = validSamples
	} else {
		samples = make([]saturationSample, 0, 10)
	}

	// Add current sample
	samples = append(samples, saturationSample{
		saturation: saturation,
		timestamp:  now,
	})

	// Update saturations
	p.saturations[key] = samples

	// Compute delta/trend over the time window
	if len(samples) < 2 {
		// Not enough history - assume no trend
		return 0.0
	}

	// slope is in saturation units per second
	return slope(samples)
}

// slope computes a linear regression using the least squares method
func slope(samples []saturationSample) float64 {
	// slope = (n*Σ(t*s) - Σ(t)*Σ(s)) / (n*Σ(t²) - (Σ(t))²)

	n := float64(len(samples))
	var tSum, sSum, tsSum, tsqrSum float64

	// Use the oldest sample timestamp as t=0 reference point
	t0 := samples[0].timestamp

	for _, sample := range samples {
		t := sample.timestamp.Sub(t0).Seconds() // time in seconds since oldest sample
		s := sample.saturation
		tSum += t
		sSum += s
		tsSum += t * s
		tsqrSum += t * t
	}

	denominator := n*tsqrSum - tSum*tSum
	if denominator < 1e-10 {
		// All samples at same time (or very close)
		return 0.0
	}

	return (n*tsSum - tSum*sSum) / denominator
}

// generateUsageLimitCacheKey creates a composite cache key from endpoint subset and priority.
// This ensures that usage limits are tracked independently for each (subset, priority) pair.
// FIXME: should also implement cache eviction
func generateUsageLimitCacheKey(reqMetadata map[string]any, priority int) string {
	subsetKey := generateCacheKey(reqMetadata)
	// Use a separator that won't appear in endpoint addresses to avoid collisions
	return subsetKey + "::" + strconv.Itoa(priority)
}

// following: lifted from locator.go
// FIXME: should also implement cache eviction
const (
	// defaultCacheKey is used when no subset filter is present (Return All Pods).
	defaultCacheKey = "__default__"

	// emptySubsetCacheKey is used when a subset filter is present but empty (Return No Pods).
	emptySubsetCacheKey = "__explicit_empty__"
)

// generateCacheKey creates a deterministic string key representing the pod selection criteria.
// It handles the "x-gateway-destination-endpoint-subset" structure specifically.
func generateCacheKey(reqMetadata map[string]any) string {
	// No Metadata -> All Pods
	if reqMetadata == nil {
		return defaultCacheKey
	}

	subsetMap, found := reqMetadata[metadata.SubsetFilterNamespace].(map[string]any)
	if !found {
		return defaultCacheKey
	}

	// The subset filter key contains a list of endpoint strings (e.g., "10.0.0.1:8080").
	// We must treat this list as a set (order independent).
	endpointSubsetList, found := subsetMap[metadata.SubsetFilterKey].([]any)

	// Namespace exists, but "subset" key is missing -> All Pods
	if !found {
		return defaultCacheKey
	}

	// "subset" key exists, but is empty list -> No Pods
	if len(endpointSubsetList) == 0 {
		return emptySubsetCacheKey
	}

	// Optimization: If there's only one endpoint, return it directly to avoid allocation.
	if len(endpointSubsetList) == 1 {
		if s, ok := endpointSubsetList[0].(string); ok {
			return s
		}
		return defaultCacheKey // Fallback for malformed data.
	}

	// Copy and sort to ensure determinism ( [A, B] must equal [B, A] ).
	endpoints := make([]string, 0, len(endpointSubsetList))
	for _, ep := range endpointSubsetList {
		if s, ok := ep.(string); ok {
			endpoints = append(endpoints, s)
		}
	}

	sort.Strings(endpoints)
	return strings.Join(endpoints, "|")
}
