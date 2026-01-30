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
	"errors"
	"fmt"
	"math"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	MetricScorerType = "metric-scorer"
)

type OptimizationMode string

const (
	OptimizationModeMinimize OptimizationMode = "Minimize"
	OptimizationModeMaximize OptimizationMode = "Maximize"
)

type NormalizationAlgo string

const (
	NormalizationAlgoLinear  NormalizationAlgo = "Linear"
	NormalizationAlgoSoftmax NormalizationAlgo = "Softmax"
)

// normalizer transforms raw metrics into scores.
type normalizer interface {
	Normalize(metrics map[framework.Endpoint]float64) map[framework.Endpoint]float64
}

type linearNormalizer struct {
	min  float64
	max  float64
	mode OptimizationMode
}

func (n *linearNormalizer) Normalize(rawMetrics map[framework.Endpoint]float64) map[framework.Endpoint]float64 {
	scores := make(map[framework.Endpoint]float64, len(rawMetrics))
	rangeVal := n.max - n.min
	if rangeVal == 0 {
		// Avoid division by zero; treat all equal.
		for ep := range rawMetrics {
			scores[ep] = 0
		}
		return scores
	}

	for ep, val := range rawMetrics {
		clampedVal := min(max(n.min, val), n.max)
		score := (clampedVal - n.min) / rangeVal
		if n.mode == OptimizationModeMinimize {
			score = 1.0 - score
		}
		scores[ep] = score
	}
	return scores
}

type softmaxNormalizer struct {
	mode OptimizationMode
}

func (n *softmaxNormalizer) Normalize(rawMetrics map[framework.Endpoint]float64) map[framework.Endpoint]float64 {
	scores := make(map[framework.Endpoint]float64, len(rawMetrics))
	if len(rawMetrics) == 0 {
		return scores
	}

	// Create inputs for softmax.
	// If minimizing, negate values so that lower values become higher in the exponent.
	inputs := make(map[framework.Endpoint]float64, len(rawMetrics))
	for ep, val := range rawMetrics {
		if n.mode == OptimizationModeMinimize {
			inputs[ep] = -val
		} else {
			inputs[ep] = val
		}
	}

	// Find max for numerical stability (prevents overflow).
	maxVal := -1e18
	first := true
	for _, val := range inputs {
		if first || val > maxVal {
			maxVal = val
			first = false
		}
	}

	// Compute exponentials and sum.
	sumExp := 0.0
	expValues := make(map[framework.Endpoint]float64, len(rawMetrics))
	for ep, val := range inputs {
		v := math.Exp(val - maxVal)
		expValues[ep] = v
		sumExp += v
	}

	// Normalize.
	for ep, v := range expValues {
		if sumExp == 0 {
			scores[ep] = 1.0 / float64(len(rawMetrics))
		} else {
			scores[ep] = v / sumExp
		}
	}

	return scores
}

// MetricScorerConfig defines the configuration for MetricScorer.
type MetricScorerConfig struct {
	// MetricName is the name of the metric to use for scoring.
	MetricName string `json:"metricName"`
	// Min is the minimum expected value for the metric (used for normalization).
	Min float64 `json:"min"`
	// Max is the maximum expected value for the metric (used for normalization).
	Max float64 `json:"max"`
	// OptimizationMode defines how the metric value correlates to the score.
	// Defaults to Minimize if unspecified.
	OptimizationMode OptimizationMode `json:"optimizationMode"`
	// NormalizationAlgo defines the algorithm used for normalization.
	// Defaults to Linear if unspecified.
	NormalizationAlgo NormalizationAlgo `json:"normalizationAlgo"`
}

// SetDefaults sets the default values for the configuration.
func (c *MetricScorerConfig) SetDefaults() {
	if c.OptimizationMode == "" {
		c.OptimizationMode = OptimizationModeMinimize
	}
	if c.NormalizationAlgo == "" {
		c.NormalizationAlgo = NormalizationAlgoLinear
	}
}

// Validate validates the configuration.
func (c *MetricScorerConfig) Validate() error {
	if c.MetricName == "" {
		return errors.New("metricName is required")
	}

	switch c.OptimizationMode {
	case OptimizationModeMinimize, OptimizationModeMaximize:
	default:
		return fmt.Errorf("invalid optimizationMode %q, allowed: %s, %s",
			c.OptimizationMode, OptimizationModeMinimize, OptimizationModeMaximize)
	}

	switch c.NormalizationAlgo {
	case NormalizationAlgoLinear:
		if c.Max <= c.Min {
			return fmt.Errorf("max must be strictly greater than min found min: %v max: %v", c.Min, c.Max)
		}
	case NormalizationAlgoSoftmax:
	default:
		return fmt.Errorf("invalid normalizationAlgo %q, allowed: %s, %s",
			c.NormalizationAlgo, NormalizationAlgoLinear, NormalizationAlgoSoftmax)
	}
	return nil
}

// DefaultMetricScorerConfig holds the default configuration.
var DefaultMetricScorerConfig = MetricScorerConfig{
	OptimizationMode:  OptimizationModeMinimize,
	NormalizationAlgo: NormalizationAlgoLinear,
}

var (
	_ framework.Scorer         = &metricScorer{}
	_ fwkplugin.ConsumerPlugin = &metricScorer{}
)

// MetricScorerFactory defines the factory function for MetricScorer.
func MetricScorerFactory(name string, rawParameters json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := DefaultMetricScorerConfig
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to parse configuration for %s: %w", name, err)
		}
	}

	config.SetDefaults()
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration for %s: %w", name, err)
	}

	return newMetricScorer(config).withName(name), nil
}

// newMetricScorer initializes a new MetricScorer.
func newMetricScorer(config MetricScorerConfig) *metricScorer {
	var normalizer normalizer
	switch config.NormalizationAlgo {
	case NormalizationAlgoSoftmax:
		normalizer = &softmaxNormalizer{
			mode: config.OptimizationMode,
		}
	case NormalizationAlgoLinear:
		fallthrough
	default:
		normalizer = &linearNormalizer{
			min:  config.Min,
			max:  config.Max,
			mode: config.OptimizationMode,
		}
	}

	return &metricScorer{
		typedName:  fwkplugin.TypedName{Type: MetricScorerType, Name: MetricScorerType},
		config:     config,
		normalizer: normalizer,
	}
}

// metricScorer scores endpoints based on a generic metric.
type metricScorer struct {
	typedName  fwkplugin.TypedName
	config     MetricScorerConfig
	normalizer normalizer
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *metricScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *metricScorer) Category() framework.ScorerCategory {
	return framework.Distribution
}

// Consumes returns the list of data that is consumed by the plugin.
func (s *metricScorer) Consumes() map[string]any {
	return map[string]any{
		s.config.MetricName: float64(0),
	}
}

// withName sets the name of the scorer.
func (s *metricScorer) withName(name string) *metricScorer {
	s.typedName.Name = name
	return s
}

// Score returns the scoring result for the given list of pods based on context.
func (s *metricScorer) Score(
	ctx context.Context,
	_ *framework.CycleState,
	_ *framework.LLMRequest,
	endpoints []framework.Endpoint,
) map[framework.Endpoint]float64 {
	rawMetrics := make(map[framework.Endpoint]float64, len(endpoints))
	logger := log.FromContext(ctx).V(logutil.TRACE)

	minVal := s.config.Min
	maxVal := s.config.Max

	for _, endpoint := range endpoints {
		val := 0.0
		if v, ok := endpoint.GetMetrics().Custom[s.config.MetricName]; ok {
			val = v
		} else {
			logger.Info("Missing custom metric for endpoint", "endpoint", endpoint.GetMetadata().NamespacedName, "metric", s.config.MetricName)
			// Apply worst-case value for missing metrics.
			if s.config.OptimizationMode == OptimizationModeMinimize {
				val = maxVal
			} else {
				val = minVal
			}
		}
		rawMetrics[endpoint] = val
	}

	scores := s.normalizer.Normalize(rawMetrics)
	return scores
}
