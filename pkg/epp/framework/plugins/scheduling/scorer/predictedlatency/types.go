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
package predictedlatency

import (
	"context"
	"strings"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

type headroomStrategy string

type choice struct {
	endpointName schedulingtypes.Endpoint
	weight       int
}

const (
	// headroomStrategyLeast prioritizes pods with least positive headroom (better packing)
	headroomStrategyLeast headroomStrategy = "least"
	// headroomStrategyMost prioritizes pods with most positive headroom (more conservative)
	headroomStrategyMost headroomStrategy = "most"

	headroomStrategyCompositeLeast headroomStrategy = "composite-least"
	headroomStrategyCompositeMost  headroomStrategy = "composite-most"
	headroomStrategyCompositeOnly  headroomStrategy = "composite-only"

	// TTFT header string
	ttftSLOHeaderKey = "x-slo-ttft-ms"
	// TPOT header string
	tpotSLOHeaderKey = "x-slo-tpot-ms"
)

const (
	PredictedLatencyPluginType = "predicted-latency-scorer"
	eps                        = 1e-9
	wMax                       = 100
	minWeight                  = 1
)

type podSelectionMode string

const (
	podSelectionLinear podSelectionMode = "linear" // weighted-random (current behavior)
	podSelectionMax    podSelectionMode = "max"    // pick argmax weight
)

// PredictionRequestBuilder constructs prediction and training requests with optional customization.
// This interface allows different implementations to customize how prediction requests are built,
// for example to add pod type information for disaggregated serving scenarios.
type PredictionRequestBuilder interface {
	// BuildPredictionRequest constructs a prediction request for a pod
	BuildPredictionRequest(
		ctx context.Context,
		pod schedulingtypes.Endpoint,
		metrics *datalayer.Metrics,
		prompt string,
		generatedTokens int,
		prefixCacheScore float64,
	) latencypredictor.PredictionRequest

	// BuildTrainingEntry constructs a training entry for a pod
	BuildTrainingEntry(
		ctx context.Context,
		pod schedulingtypes.Endpoint,
		metrics *datalayer.Metrics,
		prompt string,
		actualTTFT float64,
		actualTPOT float64,
		timestamp time.Time,
		generatedTokens int,
		prefixCacheScore float64,
	) latencypredictor.TrainingEntry
}

// DefaultPredictionRequestBuilder provides the default monolithic behavior for building prediction requests.
// This implementation leaves PodType empty, suitable for monolithic (non-disaggregated) deployments.
type DefaultPredictionRequestBuilder struct{}

// BuildPredictionRequest constructs a standard prediction request without pod type information
func (b *DefaultPredictionRequestBuilder) BuildPredictionRequest(
	ctx context.Context,
	pod schedulingtypes.Endpoint,
	metrics *datalayer.Metrics,
	prompt string,
	generatedTokens int,
	prefixCacheScore float64,
) latencypredictor.PredictionRequest {
	return latencypredictor.PredictionRequest{
		KVCachePercentage:  metrics.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(prompt)), // Simple word-based tokenization
		NumRequestWaiting:  metrics.WaitingQueueSize,
		NumRequestRunning:  metrics.RunningRequestsSize,
		NumTokensGenerated: generatedTokens,
		PrefixCacheScore:   prefixCacheScore,
		PodType:            "", // Empty for monolithic deployments
	}
}

// BuildTrainingEntry constructs a standard training entry without pod type information
func (b *DefaultPredictionRequestBuilder) BuildTrainingEntry(
	ctx context.Context,
	pod schedulingtypes.Endpoint,
	metrics *datalayer.Metrics,
	prompt string,
	actualTTFT float64,
	actualTPOT float64,
	timestamp time.Time,
	generatedTokens int,
	prefixCacheScore float64,
) latencypredictor.TrainingEntry {
	return latencypredictor.TrainingEntry{
		KVCachePercentage:  metrics.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(prompt)), // Simple word-based tokenization
		ActualTTFT:         actualTTFT,
		ActualTPOT:         actualTPOT,
		Timestamp:          timestamp,
		NumRequestWaiting:  metrics.WaitingQueueSize,
		NumRequestRunning:  metrics.RunningRequestsSize,
		NumTokensGenerated: generatedTokens,
		PrefixCacheScore:   prefixCacheScore,
		PodType:            "", // Empty for monolithic deployments
	}
}
