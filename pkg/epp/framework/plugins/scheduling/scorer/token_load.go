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

package scorer

import (
	"context"
	"encoding/json"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector/framework/plugins/concurrencydetector"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/env"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	TokenLoadScorerType = "token-load-scorer"
	tokenQueueThresholdEnvName = "TOKEN_LOAD_AWARE_SCORER_QUEUE_THRESHOLD"
	tokenQueueThresholdDefault = 4194304 // 128 requests @ 32K per request
)

// compile-time type assertion
var _ framework.Scorer = &TokenLoadScorer{}

type TokenLoadScorer struct {
	typedName      fwkplugin.TypedName
	handle         fwkplugin.Handle
	queueThreshold float64
}

func TokenLoadScorerFactory(name string, _ json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return &TokenLoadScorer{
		typedName: fwkplugin.TypedName{Type: TokenLoadScorerType, Name: name},
		handle:    handle,
		queueThreshold: float64(env.GetEnvInt(tokenQueueThresholdEnvName, tokenQueueThresholdDefault, log.Log)),
	}, nil
}

func (s *TokenLoadScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

func (s *TokenLoadScorer) Category() framework.ScorerCategory {
	return framework.Distribution
}

func (s *TokenLoadScorer) Consumes() map[string]any {
	return nil
}

func (s *TokenLoadScorer) Score(ctx context.Context, _ *framework.CycleState, _ *framework.LLMRequest, endpoints []framework.Endpoint) map[framework.Endpoint]float64 {
	scores := make(map[framework.Endpoint]float64, len(endpoints))
	logger := log.FromContext(ctx)

	detectorPlugin := s.handle.Plugin(concurrencydetector.ConcurrencyDetectorType)
	if detectorPlugin == nil {
		logger.V(1).Info("TokenLoadScorer: ConcurrencyDetector plugin not found, returning neutral scores")
		for _, endpoint := range endpoints {
			scores[endpoint] = 1.0
		}
		return scores
	}

	detector, ok := detectorPlugin.(*concurrencydetector.Detector)
	if !ok {
		logger.V(1).Info("TokenLoadScorer: Plugin is not a ConcurrencyDetector, returning neutral scores")
		for _, endpoint := range endpoints {
			scores[endpoint] = 1.0
		}
		return scores
	}

	for _, endpoint := range endpoints {
		endpointID := endpoint.GetMetadata().NamespacedName.String()
		tokenLoad := float64(detector.GetInFlightTokens(endpointID))

		score := 0.0
		if tokenLoad == 0 {
			score = 1.0
		} else {
			if tokenLoad > s.queueThreshold {
				tokenLoad = s.queueThreshold
			}
			score = 1.0 - (tokenLoad / s.queueThreshold)
		}
		scores[endpoint] = score
		logger.V(1).Info("TokenLoadScorer scoring", "endpoint", endpointID, "tokenLoad", tokenLoad, "score", score)
	}

	return scores
}
