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

package latencyadmission

import (
	"context"
	"testing"

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrlatency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
)

func makeLatencyAdmissionEndpoint(name string, kvCache float64, runningRequests int) schedulingtypes.Endpoint {
	return schedulingtypes.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{
			KVCacheUsagePercent: kvCache,
			RunningRequestsSize: runningRequests,
		},
		nil,
	)
}

func makeSheddableRequest(ttftSLO, tpotSLO string) *schedulingtypes.LLMRequest {
	return &schedulingtypes.LLMRequest{
		Headers: map[string]string{
			ttftSLOHeaderKey: ttftSLO,
			tpotSLOHeaderKey: tpotSLO,
		},
		Objectives: schedulingtypes.RequestObjectives{Priority: -1},
	}
}

func makeNonSheddableRequest(ttftSLO, tpotSLO string) *schedulingtypes.LLMRequest {
	return &schedulingtypes.LLMRequest{
		Headers: map[string]string{
			ttftSLOHeaderKey: ttftSLO,
			tpotSLOHeaderKey: tpotSLO,
		},
		Objectives: schedulingtypes.RequestObjectives{Priority: 1},
	}
}

func TestAdmitRequest(t *testing.T) {
	plugin := NewLatencyAdmission(LatencyAdmissionDefaultConfig)

	tests := []struct {
		name      string
		request   *schedulingtypes.LLMRequest
		endpoints []schedulingtypes.Endpoint
		setupFn   func(endpoints []schedulingtypes.Endpoint) // set endpoint attributes
		wantErr   bool
	}{
		{
			name:    "nil request — admit",
			request: nil,
			wantErr: false,
		},
		{
			name:    "non-sheddable request — always admit",
			request: makeNonSheddableRequest("100", "30"),
			endpoints: []schedulingtypes.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
			},
			setupFn: func(endpoints []schedulingtypes.Endpoint) {
				// All invalid predictions
				endpoints[0].Put(attrlatency.LatencyPredictionInfoKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40))
			},
			wantErr: false,
		},
		{
			name:    "no SLO headers — admit",
			request: makeSheddableRequest("", ""),
			endpoints: []schedulingtypes.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
			},
			wantErr: false,
		},
		{
			name:    "sheddable, all invalid, all busy, no cold — reject",
			request: makeSheddableRequest("100", "30"),
			endpoints: []schedulingtypes.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
				makeLatencyAdmissionEndpoint("pod2", 0.4, 3),
			},
			setupFn: func(endpoints []schedulingtypes.Endpoint) {
				endpoints[0].Put(attrlatency.LatencyPredictionInfoKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40))
				endpoints[1].Put(attrlatency.LatencyPredictionInfoKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -30, -5, 130, 35))
			},
			wantErr: true,
		},
		{
			name:    "sheddable, all invalid, but one pod idle — admit",
			request: makeSheddableRequest("100", "30"),
			endpoints: []schedulingtypes.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
				makeLatencyAdmissionEndpoint("pod2", 0.4, 0), // idle
			},
			setupFn: func(endpoints []schedulingtypes.Endpoint) {
				endpoints[0].Put(attrlatency.LatencyPredictionInfoKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40))
				endpoints[1].Put(attrlatency.LatencyPredictionInfoKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -30, -5, 130, 35))
			},
			wantErr: false,
		},
		{
			name:    "sheddable, all invalid, but cold pod exists — admit",
			request: makeSheddableRequest("100", "30"),
			endpoints: []schedulingtypes.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
				makeLatencyAdmissionEndpoint("pod2", 0.01, 3), // cold
			},
			setupFn: func(endpoints []schedulingtypes.Endpoint) {
				endpoints[0].Put(attrlatency.LatencyPredictionInfoKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40))
				endpoints[1].Put(attrlatency.LatencyPredictionInfoKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -30, -5, 130, 35))
			},
			wantErr: false,
		},
		{
			name:    "sheddable, one valid endpoint — admit",
			request: makeSheddableRequest("100", "30"),
			endpoints: []schedulingtypes.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
				makeLatencyAdmissionEndpoint("pod2", 0.4, 3),
			},
			setupFn: func(endpoints []schedulingtypes.Endpoint) {
				endpoints[0].Put(attrlatency.LatencyPredictionInfoKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40))
				endpoints[1].Put(attrlatency.LatencyPredictionInfoKey,
					attrlatency.NewLatencyPredictionInfo(true, true, 20, 5, 80, 25)) // valid
			},
			wantErr: false,
		},
		{
			name:    "sheddable, no prediction data on endpoints — admit (fail-open)",
			request: makeSheddableRequest("100", "30"),
			endpoints: []schedulingtypes.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
			},
			// no setupFn — no latency attributes set
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setupFn != nil {
				tt.setupFn(tt.endpoints)
			}
			err := plugin.AdmitRequest(context.Background(), tt.request, tt.endpoints)
			if (err != nil) != tt.wantErr {
				t.Errorf("AdmitRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
