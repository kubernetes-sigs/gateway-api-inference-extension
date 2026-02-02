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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestLinearNormalizer(t *testing.T) {
	tests := []struct {
		name     string
		min, max float64
		inputs   []float64
		expected []float64
	}{
		{
			name:     "Standard range 0-100",
			min:      0,
			max:      100,
			inputs:   []float64{0, 50, 100},
			expected: []float64{0.0, 0.5, 1.0},
		},
		{
			name:     "Clamping values outside range",
			min:      0,
			max:      100,
			inputs:   []float64{-10, 110},
			expected: []float64{0.0, 1.0},
		},
		{
			name:     "Zero range (avoid divide by zero)",
			min:      10,
			max:      10,
			inputs:   []float64{5, 10, 15},
			expected: []float64{0.0, 0.0, 0.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &linearNormalizer{min: tt.min, max: tt.max}
			metrics := make(map[fwksched.Endpoint]float64)
			eps := make([]fwksched.Endpoint, len(tt.inputs))
			for i, v := range tt.inputs {
				ep := newEndpointWithMetric(v, "metric")
				metrics[ep] = v
				eps[i] = ep
			}

			scores := n.Normalize(metrics)

			for i, ep := range eps {
				assert.InDelta(t, tt.expected[i], scores[ep], 1e-6)
			}
		})
	}
}

func TestSoftmaxNormalizer(t *testing.T) {
	tests := []struct {
		name     string
		inputs   []float64
		expected []float64
	}{
		{
			name:     "Basic Softmax",
			inputs:   []float64{1.0, 2.0, 3.0},
			expected: []float64{0.09003057, 0.24472847, 0.66524096}, // exp(1)/sum, exp(2)/sum, exp(3)/sum
		},
		{
			name:     "Shift Invariance (adding constant doesn't change prob)",
			inputs:   []float64{101.0, 102.0, 103.0},
			expected: []float64{0.09003057, 0.24472847, 0.66524096},
		},
		{
			name:     "All Equal",
			inputs:   []float64{5.0, 5.0},
			expected: []float64{0.5, 0.5},
		},
		{
			name:     "Empty",
			inputs:   []float64{},
			expected: []float64{},
		},
	}

	n := &softmaxNormalizer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics := make(map[fwksched.Endpoint]float64)
			eps := make([]fwksched.Endpoint, len(tt.inputs))
			for i, v := range tt.inputs {
				ep := newEndpointWithMetric(0, fmt.Sprintf("metric-%d", i))
				metrics[ep] = v
				eps[i] = ep
			}

			scores := n.Normalize(metrics)

			sum := 0.0
			for i, ep := range eps {
				if len(tt.expected) > 0 {
					assert.InDelta(t, tt.expected[i], scores[ep], 1e-6)
					sum += scores[ep]
				}
			}
			if len(tt.inputs) > 0 {
				assert.InDelta(t, 1.0, sum, 1e-6, "Sum of softmax scores should be 1.0")
			}
		})
	}
}

func TestMetricScorer(t *testing.T) {
	metricName := "custom_metric"

	tests := []struct {
		name           string
		config         MetricScorerConfig
		inputs         []float64
		expectedScores map[int]float64 // expected map from input index to score
	}{
		{
			name: "Linear + Minimize (Default)",
			config: MetricScorerConfig{
				MetricName: metricName,
				Min:        0,
				Max:        100,
				// Defaults: Minimize, Linear
			},
			inputs: []float64{0, 50, 100},
			expectedScores: map[int]float64{
				0: 1.0, // Best
				1: 0.5,
				2: 0.0, // Worst
			},
		},
		{
			name: "Linear + Maximize",
			config: MetricScorerConfig{
				MetricName:       metricName,
				Min:              0,
				Max:              100,
				OptimizationMode: OptimizationModeMaximize,
			},
			inputs: []float64{0, 50, 100},
			expectedScores: map[int]float64{
				0: 0.0, // Worst
				1: 0.5,
				2: 1.0, // Best
			},
		},
		{
			name: "Softmax + Maximize",
			config: MetricScorerConfig{
				MetricName:        metricName,
				NormalizationAlgo: NormalizationAlgoSoftmax,
				OptimizationMode:  OptimizationModeMaximize,
			},
			inputs: []float64{1, 2, 3},
			expectedScores: map[int]float64{
				0: 0.09003057,
				1: 0.24472847,
				2: 0.66524096,
			},
		},
		{
			name: "Softmax + Minimize (Inverted)",
			config: MetricScorerConfig{
				MetricName:        metricName,
				NormalizationAlgo: NormalizationAlgoSoftmax,
				OptimizationMode:  OptimizationModeMinimize,
			},
			inputs: []float64{1, 2, 3},
			expectedScores: map[int]float64{
				0: 0.66524096, // exp(-1) / sum(exp(-1), exp(-2), exp(-3))
				1: 0.24472847, // exp(-2) / sum
				2: 0.09003057, // exp(-3) / sum
			},
		},
		{
			name: "Missing Metric Handling",
			config: MetricScorerConfig{
				MetricName: metricName,
				Min:        0, Max: 100,
				OptimizationMode: OptimizationModeMinimize, // Default
			},
			// Special case: input -1 means missing metric for helper
			inputs: []float64{50, -1},
			expectedScores: map[int]float64{
				0: 0.5, // 50 -> 0.5 -> 1-0.5=0.5
				1: 0.0, // Missing -> Assumes Max (100) -> 1.0 -> 1-1=0
			},
		},
		{
			name: "Missing Metric + Maximize Mode",
			config: MetricScorerConfig{
				MetricName:       metricName,
				Min:              0,
				Max:              100,
				OptimizationMode: OptimizationModeMaximize,
			},
			inputs: []float64{50, -1}, // -1 means missing
			expectedScores: map[int]float64{
				0: 0.5, // 50 -> 0.5
				1: 0.0, // Missing -> Assumes Min (0) -> 0.0
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Apply defaults to simulate factory behavior or explicit setup.
			test.config.SetDefaults()
			scorer := newMetricScorer(test.config)

			endpoints := make([]fwksched.Endpoint, len(test.inputs))
			for i, v := range test.inputs {
				if v == -1 {
					endpoints[i] = newEndpointWithoutMetric()
				} else {
					endpoints[i] = newEndpointWithMetric(v, metricName)
				}
			}

			scores := scorer.Score(context.Background(), fwksched.NewCycleState(), &fwksched.LLMRequest{}, endpoints)

			for i, endpoint := range endpoints {
				expected := test.expectedScores[i]
				assert.InDelta(t, expected, scores[endpoint], 1e-5, "Endpoint %d", i)
			}
		})
	}
}

func TestMetricScorerFactory_Validation(t *testing.T) {
	tests := []struct {
		name      string
		rawJson   string
		expectErr bool
	}{
		{
			name:      "Valid Config",
			rawJson:   `{"metricName": "foo", "min": 0, "max": 100}`,
			expectErr: false,
		},
		{
			name:      "Missing MetricName",
			rawJson:   `{"min": 0, "max": 100}`,
			expectErr: true,
		},
		{
			name:      "Invalid OptimizationMode",
			rawJson:   `{"metricName": "foo", "optimizationMode": "Random"}`,
			expectErr: true,
		},
		{
			name:      "Invalid NormalizationAlgo",
			rawJson:   `{"metricName": "foo", "normalizationAlgo": "Magic"}`,
			expectErr: true,
		},
		{
			name:      "Linear: Max <= Min",
			rawJson:   `{"metricName": "foo", "normalizationAlgo": "Linear", "min": 100, "max": 0}`,
			expectErr: true,
		},
		{
			name:      "Softmax: Min/Max Ignored (should be valid even if weird)",
			rawJson:   `{"metricName": "foo", "normalizationAlgo": "Softmax"}`,
			expectErr: false,
		},
	}

	factory := MetricScorerFactory
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Plugin, err := factory("test", json.RawMessage(tt.rawJson), nil)
			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, Plugin)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, Plugin)
			}
		})
	}
}

func newEndpointWithMetric(val float64, name string) fwksched.Endpoint {
	m := fwkdl.NewMetrics()
	m.Custom[name] = val
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, m, nil)
}

func newEndpointWithoutMetric() fwksched.Endpoint {
	m := fwkdl.NewMetrics()
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, m, nil)
}
