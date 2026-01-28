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
	"testing"

	"github.com/stretchr/testify/assert"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestKvCacheUtilizationScorer(t *testing.T) {
	tests := []struct {
		name                   string
		endpoints              []fwksched.Endpoint
		expectedScoresEndpoint map[int]float64 // Map of endpoint index to expected score
	}{
		{
			name: "Different KV cache utilization",
			endpoints: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &datalayer.Metrics{KVCacheUsagePercent: 0.8}, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &datalayer.Metrics{KVCacheUsagePercent: 0.5}, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &datalayer.Metrics{KVCacheUsagePercent: 0.0}, nil),
			},
			expectedScoresEndpoint: map[int]float64{
				0: 0.2, // Highest KV cache usage (0.8) gets lowest score (1-0.8=0.2)
				1: 0.5, // Medium KV cache usage (0.5) gets medium score (1-0.5=0.5)
				2: 1.0, // No KV cache usage (0.0) gets highest score (1-0=1.0)
			},
		},
		{
			name: "Same KV cache utilization",
			endpoints: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &datalayer.Metrics{KVCacheUsagePercent: 0.6}, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &datalayer.Metrics{KVCacheUsagePercent: 0.6}, nil),
			},
			expectedScoresEndpoint: map[int]float64{
				0: 0.4, // Both get same score (1-0.6=0.4)
				1: 0.4,
			},
		},
		{
			name: "Zero KV cache utilization",
			endpoints: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &datalayer.Metrics{KVCacheUsagePercent: 0.0}, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &datalayer.Metrics{KVCacheUsagePercent: 0.0}, nil),
			},
			expectedScoresEndpoint: map[int]float64{
				0: 1.0, // No KV cache usage gets highest score
				1: 1.0,
			},
		},
		{
			name: "Full KV cache utilization",
			endpoints: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &datalayer.Metrics{KVCacheUsagePercent: 1.0}, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &datalayer.Metrics{KVCacheUsagePercent: 0.5}, nil),
			},
			expectedScoresEndpoint: map[int]float64{
				0: 0.0, // Full KV cache (1.0) gets lowest score (1-1=0)
				1: 0.5, // Half KV cache (0.5) gets medium score (1-0.5=0.5)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scores := NewKVCacheUtilizationScorer().Score(context.Background(), fwksched.NewCycleState(), &fwksched.LLMRequest{}, test.endpoints)

			for i, endpoint := range test.endpoints {
				expectedScore := test.expectedScoresEndpoint[i]
				assert.InDelta(t, expectedScore, scores[endpoint], 0.0001, "Endpoint %d should have score %f", i, expectedScore)
			}
		})
	}
}
