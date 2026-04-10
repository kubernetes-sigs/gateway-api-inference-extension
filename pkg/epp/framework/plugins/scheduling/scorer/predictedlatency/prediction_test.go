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
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

func TestValidatePrediction(t *testing.T) {
	tests := []struct {
		name            string
		prediction      *latencypredictor.PredictionResponse
		ttftSLO         float64
		avgTPOTSLO      float64
		sloBufferFactor float64
		streamingMode   bool
		podMinTPOTSLO   float64
		wantTTFTOk      bool
		wantTPOTOk      bool
		wantIsValid     bool
		wantHeadroom    float64
		wantTTFTHead    float64
	}{
		{
			name:            "Both TTFT and TPOT within SLO (streaming)",
			prediction:      &latencypredictor.PredictionResponse{TTFT: 500, TPOT: 30},
			ttftSLO:         1000,
			avgTPOTSLO:      50,
			sloBufferFactor: 1.0,
			streamingMode:   true,
			podMinTPOTSLO:   0,
			wantTTFTOk:      true,
			wantTPOTOk:      true,
			wantIsValid:     true,
			wantHeadroom:    20,  // 50*1.0 - 30
			wantTTFTHead:    500, // 1000 - 500
		},
		{
			name:            "TTFT exceeds SLO",
			prediction:      &latencypredictor.PredictionResponse{TTFT: 1500, TPOT: 30},
			ttftSLO:         1000,
			avgTPOTSLO:      50,
			sloBufferFactor: 1.0,
			streamingMode:   true,
			podMinTPOTSLO:   0,
			wantTTFTOk:      false,
			wantTPOTOk:      true,
			wantIsValid:     false,
			wantHeadroom:    20,   // 50*1.0 - 30
			wantTTFTHead:    -500, // 1000 - 1500
		},
		{
			name:            "TPOT exceeds SLO (streaming)",
			prediction:      &latencypredictor.PredictionResponse{TTFT: 500, TPOT: 60},
			ttftSLO:         1000,
			avgTPOTSLO:      50,
			sloBufferFactor: 1.0,
			streamingMode:   true,
			podMinTPOTSLO:   0,
			wantTTFTOk:      true,
			wantTPOTOk:      false,
			wantIsValid:     false,
			wantHeadroom:    -10, // 50*1.0 - 60
			wantTTFTHead:    500,
		},
		{
			name:            "Both exceed SLO",
			prediction:      &latencypredictor.PredictionResponse{TTFT: 1500, TPOT: 60},
			ttftSLO:         1000,
			avgTPOTSLO:      50,
			sloBufferFactor: 1.0,
			streamingMode:   true,
			podMinTPOTSLO:   0,
			wantTTFTOk:      false,
			wantTPOTOk:      false,
			wantIsValid:     false,
			wantHeadroom:    -10,
			wantTTFTHead:    -500,
		},
		{
			name:            "Non-streaming mode - TPOT always OK",
			prediction:      &latencypredictor.PredictionResponse{TTFT: 500, TPOT: 100},
			ttftSLO:         1000,
			avgTPOTSLO:      50,
			sloBufferFactor: 1.0,
			streamingMode:   false,
			podMinTPOTSLO:   0,
			wantTTFTOk:      true,
			wantTPOTOk:      true,
			wantIsValid:     true,
			wantHeadroom:    0, // non-streaming: headroom stays 0
			wantTTFTHead:    500,
		},
		{
			name:            "SLO buffer factor applied",
			prediction:      &latencypredictor.PredictionResponse{TTFT: 500, TPOT: 55},
			ttftSLO:         1000,
			avgTPOTSLO:      50,
			sloBufferFactor: 1.2,
			streamingMode:   true,
			podMinTPOTSLO:   0,
			wantTTFTOk:      true,
			wantTPOTOk:      true,
			wantIsValid:     true,
			wantHeadroom:    5, // 50*1.2 - 55 = 5
			wantTTFTHead:    500,
		},
		{
			name:            "podMinTPOTSLO constrains buffered TPOT",
			prediction:      &latencypredictor.PredictionResponse{TTFT: 500, TPOT: 35},
			ttftSLO:         1000,
			avgTPOTSLO:      50,
			sloBufferFactor: 1.0,
			streamingMode:   true,
			podMinTPOTSLO:   30, // min(50*1.0, 30*1.0) = 30
			wantTTFTOk:      true,
			wantTPOTOk:      false,
			wantIsValid:     false,
			wantHeadroom:    -5, // 30*1.0 - 35
			wantTTFTHead:    500,
		},
		{
			name:            "podMinTPOTSLO larger than avgTPOTSLO - avgTPOTSLO used",
			prediction:      &latencypredictor.PredictionResponse{TTFT: 500, TPOT: 45},
			ttftSLO:         1000,
			avgTPOTSLO:      50,
			sloBufferFactor: 1.0,
			streamingMode:   true,
			podMinTPOTSLO:   60, // min(50*1.0, 60*1.0) = 50
			wantTTFTOk:      true,
			wantTPOTOk:      true,
			wantIsValid:     true,
			wantHeadroom:    5, // 50*1.0 - 45
			wantTTFTHead:    500,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig
			cfg.SLOBufferFactor = tt.sloBufferFactor
			cfg.StreamingMode = tt.streamingMode
			s := NewPredictedLatency(cfg, nil)

			plCtx := &predictedLatencyCtx{
				ttftSLO:    tt.ttftSLO,
				avgTPOTSLO: tt.avgTPOTSLO,
			}

			ttftOk, tpotOk, isValid, headroom, ttftHeadroom := s.validatePrediction(tt.prediction, plCtx, tt.podMinTPOTSLO)

			assert.Equal(t, tt.wantTTFTOk, ttftOk, "ttftOk mismatch")
			assert.Equal(t, tt.wantTPOTOk, tpotOk, "tpotOk mismatch")
			assert.Equal(t, tt.wantIsValid, isValid, "isValid mismatch")
			assert.InDelta(t, tt.wantHeadroom, headroom, 0.001, "headroom mismatch")
			assert.InDelta(t, tt.wantTTFTHead, ttftHeadroom, 0.001, "ttftHeadroom mismatch")
		})
	}
}

func TestGeneratePredictions(t *testing.T) {
	tests := []struct {
		name             string
		predictor        *mockPredictor
		endpoints        []fwksched.Endpoint
		ttftSLO          float64
		avgTPOTSLO       float64
		prefixScores     map[string]float64
		wantErr          bool
		wantLen          int
		checkPredictions func(t *testing.T, predictions []endpointPredictionResult)
	}{
		{
			name: "Successful predictions for multiple endpoints",
			predictor: &mockPredictor{
				predictions: map[string]*latencypredictor.PredictionResponse{
					"0.3": {TTFT: 400, TPOT: 20},
					"0.6": {TTFT: 600, TPOT: 40},
				},
			},
			endpoints: []fwksched.Endpoint{
				createTestEndpoint("pod1", 0.3, 1, 0),
				createTestEndpoint("pod2", 0.6, 3, 2),
			},
			ttftSLO:    1000,
			avgTPOTSLO: 50,
			wantErr:    false,
			wantLen:    2,
			checkPredictions: func(t *testing.T, predictions []endpointPredictionResult) {
				assert.Equal(t, 400.0, predictions[0].TTFT)
				assert.Equal(t, 20.0, predictions[0].TPOT)
				assert.Equal(t, 600.0, predictions[1].TTFT)
				assert.Equal(t, 40.0, predictions[1].TPOT)
			},
		},
		{
			name: "Predictions with prefix cache scores",
			predictor: &mockPredictor{
				predictions: map[string]*latencypredictor.PredictionResponse{
					"0.5": {TTFT: 500, TPOT: 30},
				},
			},
			endpoints: []fwksched.Endpoint{
				createTestEndpoint("pod1", 0.5, 2, 1),
			},
			ttftSLO:    1000,
			avgTPOTSLO: 50,
			prefixScores: map[string]float64{
				"pod1": 0.75,
			},
			wantErr: false,
			wantLen: 1,
			checkPredictions: func(t *testing.T, predictions []endpointPredictionResult) {
				assert.InDelta(t, 0.75, predictions[0].PrefixCacheScore, 0.001)
			},
		},
		{
			name: "Predictor returns error",
			predictor: &mockPredictor{
				err: errors.New("prediction service unavailable"),
			},
			endpoints: []fwksched.Endpoint{
				createTestEndpoint("pod1", 0.5, 2, 1),
			},
			ttftSLO:    1000,
			avgTPOTSLO: 50,
			wantErr:    true,
		},
		{
			name: "Empty endpoints list",
			predictor: &mockPredictor{
				predictions: map[string]*latencypredictor.PredictionResponse{},
			},
			endpoints:  []fwksched.Endpoint{},
			ttftSLO:    1000,
			avgTPOTSLO: 50,
			wantErr:    false,
			wantLen:    0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig
			cfg.StreamingMode = true
			s := NewPredictedLatency(cfg, tt.predictor)

			request := createTestInferenceRequest("test-gen-pred", tt.ttftSLO, tt.avgTPOTSLO)
			plCtx := newPredictedLatencyContext(request)
			plCtx.ttftSLO = tt.ttftSLO
			plCtx.avgTPOTSLO = tt.avgTPOTSLO

			// Set prefix cache scores
			if tt.prefixScores != nil {
				for k, v := range tt.prefixScores {
					plCtx.prefixCacheScoresForEndpoints[k] = v
				}
			}

			predictions, err := s.generatePredictions(context.Background(), plCtx, tt.endpoints)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Len(t, predictions, tt.wantLen)

			if tt.checkPredictions != nil {
				tt.checkPredictions(t, predictions)
			}
		})
	}
}

func TestUpdateRequestContextWithPredictions(t *testing.T) {
	tests := []struct {
		name        string
		predictions []endpointPredictionResult
		wantKeys    []string
	}{
		{
			name: "Multiple predictions stored correctly",
			predictions: []endpointPredictionResult{
				{
					Endpoint: createTestEndpoint("pod1", 0.5, 2, 1),
					TTFT:     500,
					TPOT:     30,
					IsValid:  true,
				},
				{
					Endpoint: createTestEndpoint("pod2", 0.6, 3, 2),
					TTFT:     600,
					TPOT:     40,
					IsValid:  false,
				},
			},
			wantKeys: []string{"pod1", "pod2"},
		},
		{
			name:        "Empty predictions",
			predictions: []endpointPredictionResult{},
			wantKeys:    []string{},
		},
		{
			name: "Prediction with nil endpoint is skipped",
			predictions: []endpointPredictionResult{
				{
					Endpoint: createTestEndpoint("pod1", 0.5, 2, 1),
					TTFT:     500,
					TPOT:     30,
					IsValid:  true,
				},
				{
					Endpoint: nil,
					TTFT:     600,
					TPOT:     40,
				},
			},
			wantKeys: []string{"pod1"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig
			s := NewPredictedLatency(cfg, nil)

			request := createTestInferenceRequest("test-update-ctx", 1000, 50)
			plCtx := newPredictedLatencyContext(request)

			s.updateRequestContextWithPredictions(plCtx, tt.predictions)

			assert.Len(t, plCtx.predictionsForScheduling, len(tt.wantKeys))
			for _, key := range tt.wantKeys {
				pred, ok := plCtx.predictionsForScheduling[key]
				assert.True(t, ok, "expected key %s in predictionsForScheduling", key)
				assert.NotNil(t, pred.Endpoint)
			}
		})
	}
}

func TestGeneratePredictions_ValidityFlags(t *testing.T) {
	predictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.3": {TTFT: 400, TPOT: 20},  // within SLO
			"0.9": {TTFT: 1500, TPOT: 80}, // exceeds SLO
		},
	}

	cfg := DefaultConfig
	cfg.StreamingMode = true
	cfg.SLOBufferFactor = 1.0
	s := NewPredictedLatency(cfg, predictor)

	request := createTestInferenceRequest("test-validity", 1000, 50)
	plCtx := newPredictedLatencyContext(request)
	plCtx.ttftSLO = 1000
	plCtx.avgTPOTSLO = 50

	endpoints := []fwksched.Endpoint{
		createTestEndpoint("good-pod", 0.3, 1, 0),
		createTestEndpoint("bad-pod", 0.9, 6, 4),
	}

	predictions, err := s.generatePredictions(context.Background(), plCtx, endpoints)
	require.NoError(t, err)
	require.Len(t, predictions, 2)

	// good-pod: TTFT=400 < 1000, TPOT=20 < 50 => valid
	assert.True(t, predictions[0].TTFTValid, "good-pod TTFT should be valid")
	assert.True(t, predictions[0].TPOTValid, "good-pod TPOT should be valid")
	assert.True(t, predictions[0].IsValid, "good-pod should be valid")
	assert.InDelta(t, 600, predictions[0].TTFTHeadroom, 0.001) // 1000 - 400
	assert.InDelta(t, 30, predictions[0].Headroom, 0.001)      // 50 - 20

	// bad-pod: TTFT=1500 > 1000, TPOT=80 > 50 => invalid
	assert.False(t, predictions[1].TTFTValid, "bad-pod TTFT should be invalid")
	assert.False(t, predictions[1].TPOTValid, "bad-pod TPOT should be invalid")
	assert.False(t, predictions[1].IsValid, "bad-pod should be invalid")
	assert.InDelta(t, -500, predictions[1].TTFTHeadroom, 0.001) // 1000 - 1500
	assert.InDelta(t, -30, predictions[1].Headroom, 0.001)      // 50 - 80
}
