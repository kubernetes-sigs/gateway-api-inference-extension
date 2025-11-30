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

// Package requestcontrol contains helpers to decouple latency-predictor logic.
package slo_aware_router

import (
	"os"
	"strconv"
	"strings"
)

var DefaultSamplingMean = func() float64 {
	if value, exists := os.LookupEnv("SAMPLING_MEAN"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue > 0 {
			return parsedValue
		}
	}
	return 100.0 // default value
}()

var MaxSampledTokens = func() int {
	if value, exists := os.LookupEnv("MAX_SAMPLED_TOKENS"); exists {
		if parsedValue, err := strconv.Atoi(value); err == nil && parsedValue > 0 {
			return parsedValue
		}
	}
	return 20 // default value
}()

var SLOBufferFactor = func() float64 {
	if value, exists := os.LookupEnv("SLO_BUFFER_FACTOR"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil {
			return parsedValue
		}
	}
	return 1.0 // default value
}()

var NegHeadroomTTFTWeight = func() float64 {
	if value, exists := os.LookupEnv("NEG_HEADROOM_TTFT_WEIGHT"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue >= 0 {
			return parsedValue
		}
	}
	return 0.8 // default: TTFT dominates when violating SLOs
}()

var NegHeadroomTPOTWeight = func() float64 {
	if value, exists := os.LookupEnv("NEG_HEADROOM_TPOT_WEIGHT"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue >= 0 {
			return parsedValue
		}
	}
	return 0.2 // default: TPOT less important in your tiny-output scenario
}()

var HeadroomTTFTWeight = func() float64 {
	if value, exists := os.LookupEnv("HEADROOM_TTFT_WEIGHT"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue >= 0 {
			return parsedValue
		}
	}
	return 0.8 // default
}()

var HeadroomTPOTWeight = func() float64 {
	if value, exists := os.LookupEnv("HEADROOM_TPOT_WEIGHT"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue >= 0 {
			return parsedValue
		}
	}
	return 0.2 // default
}()

var HeadroomSelectionStrategy = func() headroomStrategy {
	if value, exists := os.LookupEnv("HEADROOM_SELECTION_STRATEGY"); exists {
		switch strings.ToLower(value) {
		case "least":
			return headroomStrategyLeast
		case "most":
			return headroomStrategyMost
		case "composite-least":
			return headroomStrategyCompositeLeast
		case "composite-most":
			return headroomStrategyCompositeMost
		case "composite-only":
			return headroomStrategyCompositeOnly
		}
	}
	return headroomStrategyLeast // default to least (better packing)
}()

// If using composite headroom, weights for each component. Not used by default
var CompositeKVWeight = func() float64 {
	if v, ok := os.LookupEnv("COMPOSITE_KV_WEIGHT"); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
			return f
		}
	}
	return 1
}()

var CompositeQueueWeight = func() float64 {
	if v, ok := os.LookupEnv("COMPOSITE_QUEUE_WEIGHT"); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
			return f
		}
	}
	return 1
}()

var CompositePrefixWeight = func() float64 {
	if v, ok := os.LookupEnv("COMPOSITE_PREFIX_WEIGHT"); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
			return f
		}
	}
	return 1
}()

// With probability ε, explore (ignore affinity gate); otherwise exploit.
var EpsilonExploreSticky = func() float64 {
	// Prefer new env; fall back to old for compatibility.
	if v, ok := os.LookupEnv("STICKY_EPSILON"); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f <= 1 {
			return f
		}
	}
	return 0.01 // default 1% exploration
}()

var EpsilonExploreNeg = func() float64 {
	// Prefer new env; fall back to old for compatibility.
	if v, ok := os.LookupEnv("NEG_HEADROOM_EPSILON"); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f <= 1 {
			return f
		}
	}
	return 0.01 // default 1% exploration
}()

// τ for per-path affinity gate (aka "stickiness" threshold).
var AffinityGateTau = func() float64 {
	// Prefer new env; fall back to old for compatibility.
	if v, ok := os.LookupEnv("AFFINITY_GATE_TAU"); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f <= 1 {
			return f
		}
	}
	return 0.80
}()

// Global τ for the overall candidate set (previously "overall stickiness").
var AffinityGateTauGlobal = func() float64 {
	// Prefer new env; fall back to old for compatibility.
	if v, ok := os.LookupEnv("AFFINITY_GATE_TAU_GLOBAL"); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f <= 1 {
			return f
		}
	}
	return 0.99
}()

// Read once at init. Values: "linear" (default) or "max".
var SelectionMode = func() podSelectionMode {
	if v, ok := os.LookupEnv("POD_SELECTION_MODE"); ok {
		switch strings.ToLower(v) {
		case "max":
			return podSelectionMax
		case "linear":
			fallthrough
		default:
			return podSelectionLinear
		}
	}
	return podSelectionLinear
}()
