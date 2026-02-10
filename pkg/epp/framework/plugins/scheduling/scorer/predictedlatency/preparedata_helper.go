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

package predictedlatency

import (
	"context"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// hasColdPod checks if any pod has KV cache usage less than 2%
func (s *PredictedLatency) hasColdPod(pods []schedulingtypes.Endpoint) bool {
	for _, p := range pods {
		if p.GetMetrics().KVCacheUsagePercent < 0.02 {
			return true
		}
	}
	return false
}

// allPodsAreInvalid checks if all pod predictions indicate SLO violations
func (s *PredictedLatency) allPodsAreInvalid(predictions []endpointPredictionResult, predictedLatencyCtx *predictedLatencyCtx) bool {
	// Only check validity if we have SLO constraints
	if !(predictedLatencyCtx.ttftSLO > 0 && (predictedLatencyCtx.avgTPOTSLO > 0 || !s.config.StreamingMode)) {
		return false
	}

	for _, pred := range predictions {
		if pred.IsValid {
			return false
		}
	}
	return true
}

// allPodsHaveRunningRequests checks if every pod currently has at least one running request
func (s *PredictedLatency) allPodsHaveRunningRequests(predictions []endpointPredictionResult) bool {
	for _, pred := range predictions {
		runningRequestCount := s.getEndpointRunningRequestCount(pred.Endpoint)
		if runningRequestCount == 0 {
			return false
		}
	}
	return true
}

// hasStickyCandidateEndpoint checks if any endpoint has a prefix cache score
// high enough to pass the global affinity gate. If so, the scoring phase may
// retain it regardless of prediction validity, so we should not reject the request.
func (s *PredictedLatency) hasStickyCandidateEndpoint(predictions []endpointPredictionResult) bool {
	if s.config.AffinityGateTauGlobal <= 0 {
		return false
	}
	for _, pred := range predictions {
		if pred.PrefixCacheScore >= s.config.AffinityGateTauGlobal {
			return true
		}
	}
	return false
}

// updateHasValidPod determines and updates the hasValidPod field in the SLO context
// based on pod validity, running requests, cold pod status, and sticky affinity.
func (s *PredictedLatency) updateHasValidPod(
	ctx context.Context,
	predictedLatencyCtx *predictedLatencyCtx,
	endpoints []schedulingtypes.Endpoint,
) {
	logger := log.FromContext(ctx)
	predictions := predictedLatencyCtx.predictionsForScheduling
	allInvalid := s.allPodsAreInvalid(predictions, predictedLatencyCtx)
	allHaveRunningRequests := s.allPodsHaveRunningRequests(predictions)
	hasCold := s.hasColdPod(endpoints)
	hasSticky := s.hasStickyCandidateEndpoint(predictions)

	// Set HasValidPod to false only if all pods are invalid, all have running requests,
	// there are no cold pods available, and no sticky (high prefix-affinity) candidate exists.
	// The sticky check mirrors the old !sticky guard from Score(), ensuring we don't
	// reject requests when the affinity gate would have kept a valid candidate.
	if allInvalid && allHaveRunningRequests && !hasCold && !hasSticky {
		predictedLatencyCtx.hasValidEndpoint = false
		logger.V(logutil.DEBUG).Info("All pods are invalid and have running requests with no sticky candidate, setting HasValidPod to false")
	}
}
