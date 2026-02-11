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
	"strings"
	"time"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

type headroomStrategy string

type choice struct {
	endpointName fwksched.Endpoint
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

// buildPredictionRequest constructs a prediction request from endpoint metrics and request data.
// If endpointRoleLabel is configured, it extracts the role from the endpoint's labels and
// populates the PodType field, enabling role-aware predictions (e.g., prefill vs decode).
func buildPredictionRequest(
	endpointRoleLabel string,
	targetEndpointMetadata *fwkdl.EndpointMetadata,
	metrics *fwkdl.Metrics,
	prompt string,
	generatedTokens int,
	prefixCacheScore float64,
) latencypredictor.PredictionRequest {
	podType := ""
	if endpointRoleLabel != "" && targetEndpointMetadata != nil && targetEndpointMetadata.Labels != nil {
		podType = targetEndpointMetadata.Labels[endpointRoleLabel]
	}

	return latencypredictor.PredictionRequest{
		KVCachePercentage:  metrics.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(prompt)), // Simple word-based tokenization
		NumRequestWaiting:  metrics.WaitingQueueSize,
		NumRequestRunning:  metrics.RunningRequestsSize,
		NumTokensGenerated: generatedTokens,
		PrefixCacheScore:   prefixCacheScore,
		PodType:            podType,
	}
}

// buildTrainingEntry constructs a training entry from actual latency measurements.
// If endpointRoleLabel is configured, it extracts the role from the endpoint's labels and
// populates the PodType field, enabling role-specific model training.
func buildTrainingEntry(
	endpointRoleLabel string,
	targetEndpointMetadata *fwkdl.EndpointMetadata,
	metrics *fwkdl.Metrics,
	prompt string,
	actualTTFT float64,
	actualTPOT float64,
	timestamp time.Time,
	generatedTokens int,
	prefixCacheScore float64,
) latencypredictor.TrainingEntry {
	podType := ""
	if endpointRoleLabel != "" && targetEndpointMetadata != nil && targetEndpointMetadata.Labels != nil {
		podType = targetEndpointMetadata.Labels[endpointRoleLabel]
	}

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
		PodType:            podType,
	}
}
