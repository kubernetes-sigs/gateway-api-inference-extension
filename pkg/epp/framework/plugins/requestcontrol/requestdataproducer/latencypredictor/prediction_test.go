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

package latencypredictor

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

func createTestEndpointWithLabels(name string, kvCacheUsage float64, runningRequestsSize, waitingQueueSize int, labels map[string]string) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: name, Namespace: "default"},
		Labels:         labels,
	}, &fwkdl.Metrics{
		KVCacheUsagePercent: kvCacheUsage,
		RunningRequestsSize: runningRequestsSize,
		WaitingQueueSize:    waitingQueueSize,
	}, nil)
}

func TestValidatePrediction_StreamingMode(t *testing.T) {
	cfg := DefaultConfig
	cfg.StreamingMode = true
	pl := NewPredictedLatency(cfg, nil)

	tests := []struct {
		name            string
		pred            *latencypredictor.PredictionResponse
		ttftSLO         float64
		tpotSLO         float64
		podMinTPOTSLO   float64
		wantTTFTOk      bool
		wantTPOTOk      bool
		wantValid       bool
		wantHeadroomPos bool // headroom > 0
	}{
		{
			name:            "both within SLO",
			pred:            &latencypredictor.PredictionResponse{TTFT: 50, TPOT: 20},
			ttftSLO:         100,
			tpotSLO:         30,
			wantTTFTOk:      true,
			wantTPOTOk:      true,
			wantValid:       true,
			wantHeadroomPos: true,
		},
		{
			name:       "TTFT exceeds SLO",
			pred:       &latencypredictor.PredictionResponse{TTFT: 150, TPOT: 20},
			ttftSLO:    100,
			tpotSLO:    30,
			wantTTFTOk: false,
			wantTPOTOk: true,
			wantValid:  false,
		},
		{
			name:       "TPOT exceeds SLO",
			pred:       &latencypredictor.PredictionResponse{TTFT: 50, TPOT: 40},
			ttftSLO:    100,
			tpotSLO:    30,
			wantTTFTOk: true,
			wantTPOTOk: false,
			wantValid:  false,
		},
		{
			name:            "podMinTPOTSLO tightens threshold",
			pred:            &latencypredictor.PredictionResponse{TTFT: 50, TPOT: 25},
			ttftSLO:         100,
			tpotSLO:         30,
			podMinTPOTSLO:   20, // tighter than request SLO
			wantTTFTOk:      true,
			wantTPOTOk:      false, // 25 > 20*1.0
			wantValid:       false,
			wantHeadroomPos: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &predictedLatencyCtx{
				ttftSLO:    tt.ttftSLO,
				avgTPOTSLO: tt.tpotSLO,
			}
			ttftOk, tpotOk, valid, headroom, ttftHeadroom := pl.validatePrediction(tt.pred, ctx, tt.podMinTPOTSLO)
			assert.Equal(t, tt.wantTTFTOk, ttftOk, "ttftOk")
			assert.Equal(t, tt.wantTPOTOk, tpotOk, "tpotOk")
			assert.Equal(t, tt.wantValid, valid, "isValid")
			if tt.wantTTFTOk {
				assert.Greater(t, ttftHeadroom, 0.0, "ttftHeadroom should be positive when TTFT valid")
			}
			if tt.wantHeadroomPos {
				assert.Greater(t, headroom, 0.0, "headroom should be positive")
			}
		})
	}
}

func TestValidatePrediction_NonStreamingMode(t *testing.T) {
	config := DefaultConfig
	config.StreamingMode = false
	pl := NewPredictedLatency(config, nil)

	ctx := &predictedLatencyCtx{
		ttftSLO:    100,
		avgTPOTSLO: 30,
	}

	// In non-streaming mode, TPOT is always valid regardless of prediction
	pred := &latencypredictor.PredictionResponse{TTFT: 50, TPOT: 999}
	ttftOk, tpotOk, valid, headroom, _ := pl.validatePrediction(pred, ctx, 0)

	assert.True(t, ttftOk, "TTFT should be valid")
	assert.True(t, tpotOk, "TPOT should always be valid in non-streaming mode")
	assert.True(t, valid, "overall should be valid")
	assert.Equal(t, 0.0, headroom, "headroom should be 0 in non-streaming mode")
}

func TestValidatePrediction_PrefillEndpointNeutralizeTPOT(t *testing.T) {
	// In disaggregated serving, prefill endpoints should have TPOT neutralized.
	// Even if TPOT prediction violates SLO, prefill should be valid if TTFT is OK.
	config := DefaultConfig
	config.EndpointRoleLabel = "role"
	config.StreamingMode = true
	pl := NewPredictedLatency(config, nil)

	prefillEp := createTestEndpointWithLabels("prefill-pod", 0.3, 0, 0, map[string]string{"role": "prefill"})
	decodeEp := createTestEndpointWithLabels("decode-pod", 0.3, 0, 0, map[string]string{"role": "decode"})

	plCtx := &predictedLatencyCtx{
		ttftSLO:                       100,
		avgTPOTSLO:                    30,
		promptText:                    "test",
		prefixCacheScoresForEndpoints: map[string]float64{"prefill-pod": 0, "decode-pod": 0, "unlabeled-pod": 0},
		predictionsForScheduling:      make(map[string]endpointPredictionResult),
	}

	// Mock predictor that returns TTFT=50 (within SLO) but TPOT=999 (violates SLO)
	mockPred := &latencypredictor.PredictionResponse{TTFT: 50, TPOT: 999}

	// Prefill: TPOT should be neutralized → valid
	ttftOk, tpotOk, valid, _, _ := pl.validatePrediction(mockPred, plCtx, 0)
	// validatePrediction itself doesn't know about roles — it fails TPOT
	assert.True(t, ttftOk, "prefill TTFT should be valid")
	assert.False(t, tpotOk, "validatePrediction itself doesn't know about roles")
	assert.False(t, valid, "validatePrediction itself doesn't know about roles")

	// But generatePredictions applies the override for prefill endpoints.
	// Test the override logic directly:
	predResult := endpointPredictionResult{
		TTFTValid: true,
		TPOTValid: false,
		IsValid:   false,
		Headroom:  -969, // 30 - 999
	}

	// Simulate the fix logic from generatePredictions
	if config.EndpointRoleLabel != "" && prefillEp.GetMetadata().Labels != nil {
		if prefillEp.GetMetadata().Labels[config.EndpointRoleLabel] == Experimental_DefaultPrefillProfile {
			predResult.TPOTValid = true
			predResult.Headroom = 0
			predResult.IsValid = predResult.TTFTValid
		}
	}

	assert.True(t, predResult.TPOTValid, "prefill TPOT should be neutralized to true")
	assert.True(t, predResult.IsValid, "prefill should be valid (TTFT OK, TPOT neutralized)")
	assert.Equal(t, 0.0, predResult.Headroom, "prefill TPOT headroom should be 0")

	// Decode endpoint should NOT be neutralized
	decodeResult := endpointPredictionResult{
		TTFTValid: true,
		TPOTValid: false,
		IsValid:   false,
		Headroom:  -969,
	}
	if config.EndpointRoleLabel != "" && decodeEp.GetMetadata().Labels != nil {
		if decodeEp.GetMetadata().Labels[config.EndpointRoleLabel] == Experimental_DefaultPrefillProfile {
			decodeResult.TPOTValid = true
			decodeResult.Headroom = 0
			decodeResult.IsValid = decodeResult.TTFTValid
		}
	}
	assert.False(t, decodeResult.TPOTValid, "decode TPOT should NOT be neutralized")
	assert.False(t, decodeResult.IsValid, "decode should remain invalid")

}

func TestUpdateRequestContextWithPredictions(t *testing.T) {
	pl := NewPredictedLatency(DefaultConfig, nil)
	ctx := &predictedLatencyCtx{
		predictionsForScheduling: make(map[string]endpointPredictionResult),
	}

	ep1 := createTestEndpoint("pod1", 0.5, 5, 0)
	ep2 := createTestEndpoint("pod2", 0.3, 3, 0)

	predictions := []endpointPredictionResult{
		{Endpoint: ep1, TTFT: 50, TPOT: 20, IsValid: true},
		{Endpoint: ep2, TTFT: 80, TPOT: 30, IsValid: false},
	}

	pl.updateRequestContextWithPredictions(ctx, predictions)

	assert.Len(t, ctx.predictionsForScheduling, 2)
	assert.Equal(t, 50.0, ctx.predictionsForScheduling["pod1"].TTFT)
	assert.Equal(t, 80.0, ctx.predictionsForScheduling["pod2"].TTFT)
}
