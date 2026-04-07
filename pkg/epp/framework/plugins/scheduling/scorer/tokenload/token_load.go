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

package tokenload

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrconcurrency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

const (
	TokenLoadScorerType        = "token-load-scorer"
	tokenQueueThresholdDefault = 4194304 // 128 requests @ 32K per request
)

// Config holds the configuration for the TokenLoadScorer.
type Config struct {
	// QueueThresholdTokens defines the maximum number of in-flight tokens used for scoring normalization.
	// Defaults to 4194304 if unset.
	QueueThresholdTokens int64 `json:"queueThresholdTokens"`
}

// compile-time type assertion
var _ framework.Scorer = &TokenLoadScorer{}

type TokenLoadScorer struct {
	typedName            fwkplugin.TypedName
	queueThresholdTokens float64
}

func TokenLoadScorerFactory(name string, params json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := Config{
		QueueThresholdTokens: tokenQueueThresholdDefault,
	}
	if len(params) > 0 {
		if err := json.Unmarshal(params, &cfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal token load scorer config: %w", err)
		}
	}
	if cfg.QueueThresholdTokens <= 0 {
		cfg.QueueThresholdTokens = tokenQueueThresholdDefault
	}

	return &TokenLoadScorer{
		typedName:            fwkplugin.TypedName{Type: TokenLoadScorerType, Name: name},
		queueThresholdTokens: float64(cfg.QueueThresholdTokens),
	}, nil
}

func (s *TokenLoadScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

func (s *TokenLoadScorer) Category() framework.ScorerCategory {
	return framework.Distribution
}

func (s *TokenLoadScorer) Consumes() map[string]any {
	return map[string]any{
		attrconcurrency.InFlightLoadKey: &attrconcurrency.InFlightLoad{},
	}
}

func (s *TokenLoadScorer) Score(ctx context.Context, _ *framework.CycleState, _ *framework.InferenceRequest, endpoints []framework.Endpoint) map[framework.Endpoint]float64 {
	scores := make(map[framework.Endpoint]float64, len(endpoints))
	logger := log.FromContext(ctx)

	for _, endpoint := range endpoints {
		endpointID := endpoint.GetMetadata().NamespacedName.String()
		tokenLoad := 0.0

		if val, ok := endpoint.Get(attrconcurrency.InFlightLoadKey); ok {
			if load, ok := val.(*attrconcurrency.InFlightLoad); ok {
				tokenLoad = float64(load.Tokens)
			}
		}

		score := 0.0
		if tokenLoad <= 0 {
			score = 1.0
		} else {
			if tokenLoad > s.queueThresholdTokens {
				tokenLoad = s.queueThresholdTokens
			}
			score = 1.0 - (tokenLoad / s.queueThresholdTokens)
		}
		scores[endpoint] = score
		logger.V(1).Info("TokenLoadScorer scoring", "endpoint", endpointID, "tokenLoad", tokenLoad, "score", score)
	}

	return scores
}
