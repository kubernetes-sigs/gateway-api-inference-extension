/*
Copyright 2026 The Kubernetes Authors.

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
	"testing"

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestPrefixMatchScorer_Score(t *testing.T) {
	epWithMatch := framework.NewEndpoint(&datalayer.EndpointMetadata{}, &datalayer.Metrics{}, nil)
	epWithMatch.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(5, 10, 1))

	epWithNoMatch := framework.NewEndpoint(&datalayer.EndpointMetadata{}, &datalayer.Metrics{}, nil)
	epWithNoMatch.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(0, 10, 1))

	epWithZeroTotal := framework.NewEndpoint(&datalayer.EndpointMetadata{}, &datalayer.Metrics{}, nil)
	epWithZeroTotal.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(0, 0, 1))
	tests := []struct {
		name          string
		endpoints     []framework.Endpoint
		expectedSores map[framework.Endpoint]float64
	}{
		{
			name:      "single endpoint with matched prefix",
			endpoints: []framework.Endpoint{epWithMatch},
			expectedSores: map[framework.Endpoint]float64{
				epWithMatch: 0.5,
			},
		},
		{
			name:      "single endpoint with no matched prefix",
			endpoints: []framework.Endpoint{epWithNoMatch},
			expectedSores: map[framework.Endpoint]float64{
				epWithNoMatch: 0.0,
			},
		},
		{
			name:      "single endpoint with total blocks is 0",
			endpoints: []framework.Endpoint{epWithZeroTotal},
			expectedSores: map[framework.Endpoint]float64{
				epWithZeroTotal: 0.0,
			},
		},
	}
	scorer := &PrefixMatchScorer{}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			scores := scorer.Score(context.Background(), nil, &framework.LLMRequest{}, tc.endpoints)
			if diff := cmp.Diff(tc.expectedSores, scores, cmp.Comparer(framework.EndpointComparer)); diff != "" {
				t.Errorf("Score() returned diff (-want +got):\n%s", diff)
			}
		})
	}
}
