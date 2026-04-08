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
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrconcurrency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

func TestTokenLoadScorer(t *testing.T) {
	threshold := 1000.0

	scorer := &TokenLoadScorer{
		typedName:            fwkplugin.TypedName{Type: TokenLoadScorerType, Name: TokenLoadScorerType},
		queueThresholdTokens: threshold,
	}

	pod1NN := types.NamespacedName{Namespace: "default", Name: "pod1"}
	pod2NN := types.NamespacedName{Namespace: "default", Name: "pod2"}
	pod3NN := types.NamespacedName{Namespace: "default", Name: "pod3"}

	endpoints := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod1NN}, &fwkdl.Metrics{}, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod2NN}, &fwkdl.Metrics{}, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod3NN}, &fwkdl.Metrics{}, nil),
	}

	// pod1: 0 tokens (default)
	// pod2: 500 tokens
	endpoints[1].Put(attrconcurrency.InFlightLoadKey, &attrconcurrency.InFlightLoad{Tokens: 500})
	// pod3: 1000 tokens
	endpoints[2].Put(attrconcurrency.InFlightLoadKey, &attrconcurrency.InFlightLoad{Tokens: 1000})

	scores := scorer.Score(context.Background(), fwksched.NewCycleState(), &fwksched.LLMRequest{}, endpoints)

	assert.InDelta(t, 1.0, scores[endpoints[0]], 0.0001, "Pod1 (0 tokens) should have score 1.0")
	assert.InDelta(t, 0.5, scores[endpoints[1]], 0.0001, "Pod2 (500 tokens) should have score 0.5")
	assert.InDelta(t, 0.0, scores[endpoints[2]], 0.0001, "Pod3 (1000 tokens) should have score 0.0")
}
