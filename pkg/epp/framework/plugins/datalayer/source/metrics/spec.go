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

package metrics

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/prometheus/model/labels"
	"github.com/prometheus/prometheus/promql/parser"
)

// Spec represents a single metric's specification.
type Spec struct {
	Name   string            // the metric's name
	Labels map[string]string // maps metric's label name to value
}

// parseStringToSpec converts a string to a metrics.Spec.
// Inputs are expected in PromQL Instant Vector Selector syntax:
// metric_name{label1=value1,label2=value2}, where labels are optional.
func parseStringToSpec(spec string) (*Spec, error) {
	if spec == "" {
		return nil, nil // allow empty string to represent the nil Spec
	}

	// Be liberal in accepting inputs that are missing quotes around label values,
	// allowing both {label=value} and {label=\"value\"} inputs
	quoted := addQuotesToLabelValues(spec)
	expr, err := parser.ParseExpr(quoted)
	if err != nil {
		return nil, err
	}

	// cast to prometheus' VectorSelector to extract metric name and labels
	if vs, ok := expr.(*parser.VectorSelector); ok {
		metricLabels := make(map[string]string)
		for _, matcher := range vs.LabelMatchers {
			// do not insert pseudo labels (such as __name__, __meta_*, etc.)
			if matcher.Type == labels.MatchEqual && !strings.HasPrefix(matcher.Name, "__") {
				metricLabels[matcher.Name] = matcher.Value
			}
		}

		if vs.Name == "" {
			return nil, fmt.Errorf("empty metric name in specification: %q", spec)
		}
		return &Spec{
			Name:   vs.Name,
			Labels: metricLabels,
		}, nil
	}

	return nil, errors.New("not a valid metric specification")
}

// addQuotesToLabelValues wraps label values with quotes, if missing.
// TODO: compile regexp once as file scoped var? Seems that this won't be
// a performance hot spot.
func addQuotesToLabelValues(input string) string {
	re := regexp.MustCompile(`(\w+)\s*=\s*([^",}\s]+)`)
	return re.ReplaceAllString(input, `$1="$2"`)
}

// extract the metric family is common to standard and LoRA spec's.
func extractFamily(spec *Spec, families PrometheusMetricMap) (*dto.MetricFamily, error) {
	if spec == nil {
		return nil, errors.New("metric specification is nil")
	}

	family, exists := families[spec.Name]
	if !exists {
		return nil, fmt.Errorf("metric family %q not found", spec.Name)
	}

	if len(family.GetMetric()) == 0 {
		return nil, fmt.Errorf("no metrics found for %q", spec.Name)
	}
	return family, nil
}

// getLatestMetric retrieves the latest metric based on Spec.
func (spec *Spec) getLatestMetric(families PrometheusMetricMap) (*dto.Metric, error) {
	family, err := extractFamily(spec, families)
	if err != nil {
		return nil, err
	}

	var latest *dto.Metric
	var recent int64 = -1

	for _, metric := range family.GetMetric() {
		if spec.labelsMatch(metric.GetLabel()) {
			ts := metric.GetTimestampMs()
			if ts > recent {
				recent = ts
				latest = metric
			}
		}
	}

	if latest == nil {
		return nil, fmt.Errorf("no matching metric found for %q with labels %v", spec.Name, spec.Labels)
	}

	return latest, nil
}

// labelsMatch checks if metric labels match the specification labels.
func (spec *Spec) labelsMatch(metricLabels []*dto.LabelPair) bool {
	if len(spec.Labels) == 0 {
		return true // no label requirements
	}

	metricLabelMap := make(map[string]string)
	for _, label := range metricLabels {
		metricLabelMap[label.GetName()] = label.GetValue()
	}

	// check if all spec labels match
	for name, value := range spec.Labels {
		if metricValue, exists := metricLabelMap[name]; !exists || metricValue != value {
			return false
		}
	}

	return true
}

// extractValue gets the numeric value from different metric types.
// Currently only Gauge and Counter are supported.
func extractValue(metric *dto.Metric) float64 {
	if metric == nil {
		return 0
	}
	if gauge := metric.GetGauge(); gauge != nil {
		return gauge.GetValue()
	}
	if counter := metric.GetCounter(); counter != nil {
		return counter.GetValue()
	}
	return 0
}
