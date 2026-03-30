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

package epp

import (
	"context"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/testing/protocmp"

	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

// TestFullDuplexStreamed_DataLayer runs integration tests through the datalayer metrics pipeline.
// It mirrors a representative subset of hermetic_test.go test cases but uses the data layer
// (mock DataSource) instead of the standard FakePodMetricsClient.
func TestFullDuplexStreamed_DataLayer(t *testing.T) {
	// Datalayer tests always run in standard mode with base resources (priority=2 in InferenceObjectives).
	tests := commonTestCases(func(p int) int { return p })

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			h := NewTestHarness(t, ctx, WithStandardMode(), WithDataLayer())
			h.WithBaseResources().WithPods(tc.pods).WaitForSync(len(tc.pods), modelMyModel)
			if len(tc.pods) > 0 {
				h.WaitForReadyPodsMetric(len(tc.pods))
			}

			responses, err := integration.StreamedRequest(t, h.Client, tc.requests, len(tc.wantResponses))
			require.NoError(t, err)

			if diff := cmp.Diff(tc.wantResponses, responses,
				protocmp.Transform(),
				protocmp.SortRepeated(func(a, b *configPb.HeaderValueOption) bool {
					return a.GetHeader().GetKey() < b.GetHeader().GetKey()
				}),
			); diff != "" {
				t.Errorf("Response mismatch (-want +got): %v", diff)
			}

			if len(tc.wantMetrics) > 0 {
				h.ExpectMetrics(tc.wantMetrics)
			}
		})
	}
}
