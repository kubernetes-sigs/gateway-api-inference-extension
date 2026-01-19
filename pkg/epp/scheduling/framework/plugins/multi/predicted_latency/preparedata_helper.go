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

package predicted_latency

import (
	"context"

	"sigs.k8s.io/controller-runtime/pkg/log"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// hasColdEndpoint checks if any endpoint has KV cache usage less than 2%
func (s *PredictedLatency) hasColdPod(pods []schedulingtypes.Endpoint) bool {
	for _, p := range pods {
		if p.GetMetrics().KVCacheUsagePercent < 0.02 {
			return true
		}
	}
	return false
}

// allEndpointsAreInvalid checks if all endpoint predictions indicate SLO violations
func (s *PredictedLatency) allEndpointsAreInvalid(predictions []endpointPredictionResult, predictedLatencyCtx *predictedLatencyCtx) bool {
	// Only check validity if we have SLO constraints
	if !(predictedLatencyCtx.ttftSLO > 0 && (predictedLatencyCtx.avgITLSLO > 0 || !s.config.StreamingMode)) {
		return false
	}

	for _, pred := range predictions {
		if pred.IsValid {
			return false
		}
	}
	return true
}

// allEndpointsHaveRunningRequests checks if every endpoint currently has at least one running request
func (s *PredictedLatency) allEndpointsHaveRunningRequests(predictions []endpointPredictionResult) bool {
	for _, pred := range predictions {
		runningRequestCount := s.getEndpointRunningRequestCount(pred.Endpoint)
		if runningRequestCount == 0 {
			return false
		}
	}
	return true
}

// updateHasValidEndpoint determines and updates the hasValidEndpoint field in the SLO context
// based on endpoint validity, running requests, and cold endpoint status
func (s *PredictedLatency) updateHasValidEndpoint(
	ctx context.Context,
	predictedLatencyCtx *predictedLatencyCtx,
	endpoints []schedulingtypes.Endpoint,
) {
	logger := log.FromContext(ctx)
	predictions := predictedLatencyCtx.predictionsForScheduling
	allInvalid := s.allEndpointsAreInvalid(predictions, predictedLatencyCtx)
	allHaveRunningRequests := s.allEndpointsHaveRunningRequests(predictions)
	hasCold := s.hasColdPod(endpoints)

	// Set HasValidEndpoint to false if all endpoints are invalid, all have running requests,
	// and there are no cold endpoints available
	if allInvalid && allHaveRunningRequests && !hasCold {
		predictedLatencyCtx.hasValidEndpoint = false
		logger.V(logutil.DEBUG).Info("All endpoints are invalid and have running requests, setting HasValidEndpoint to false")
	}
}