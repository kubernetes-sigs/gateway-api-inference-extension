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
	"fmt"
	"strings"
)

// MetricSpec represents a single metric's specification.
type MetricSpec struct {
	MetricName string
	Labels     map[string]string // Label name -> Label value
}

// MetricMapping holds named MetricSpecs.
type MetricMapping struct {
	TotalQueuedRequests *MetricSpec
	KVCacheUtilization  *MetricSpec
	LoraRequestInfo     *MetricSpec
}

// stringToMetricSpec converts a string to a MetricSpec.
// Example inputs:
//
//	"metric_name"
//	"metric_name{label1=value1}"
//	"metric_name{label1=value1,label2=value2}"
func stringToMetricSpec(specStr string) (*MetricSpec, error) {
	if specStr == "" {
		return nil, nil // Allow empty strings to represent nil MetricSpecs
	}
	specStr = strings.TrimSpace(specStr)
	metricName := specStr
	labels := make(map[string]string)

	// Check for labels enclosed in curly braces
	start := strings.Index(specStr, "{")
	end := strings.Index(specStr, "}")

	if start != -1 || end != -1 { // If *either* brace is present...
		if start == -1 || end == -1 || end <= start+1 { // ...check that *both* are present and correctly placed.
			return nil, fmt.Errorf("invalid metric spec string: %q, missing or malformed label block", specStr)
		}

		metricName = strings.TrimSpace(specStr[:start])
		labelStr := specStr[start+1 : end]

		// Split into individual label pairs
		labelPairs := strings.Split(labelStr, ",")
		for _, pair := range labelPairs {
			pair = strings.TrimSpace(pair)
			parts := strings.Split(pair, "=")
			if len(parts) != 2 {
				return nil, fmt.Errorf("invalid label pair: %q in metric spec: %q", pair, specStr)
			}
			labelName := strings.TrimSpace(parts[0])
			labelValue := strings.TrimSpace(parts[1])
			if labelName == "" || labelValue == "" {
				return nil, fmt.Errorf("empty label name or value in pair: %q in metric spec: %q", pair, specStr)
			}
			labels[labelName] = labelValue
		}
		// Check for extra characters after labels
		if end != len(specStr)-1 {
			return nil, fmt.Errorf("invalid characters after label section in: %q", specStr)
		}

	}

	if metricName == "" { // Metric name cannot be empty
		return nil, fmt.Errorf("empty metric name in spec: %q", specStr)
	}

	return &MetricSpec{
		MetricName: metricName,
		Labels:     labels,
	}, nil
}

// NewMetricMapping creates a MetricMapping from string values.
func NewMetricMapping(queuedStr, kvUsageStr, loraReqInfoStr string) (*MetricMapping, error) {
	queuedSpec, err := stringToMetricSpec(queuedStr)
	if err != nil {
		return nil, fmt.Errorf("error parsing WaitingRequests: %w", err)
	}
	kvUsageSpec, err := stringToMetricSpec(kvUsageStr)
	if err != nil {
		return nil, fmt.Errorf("error parsing KVCacheUsage: %w", err)
	}
	loraReqInfoSpec, err := stringToMetricSpec(loraReqInfoStr)
	if err != nil {
		return nil, fmt.Errorf("error parsing loraReqInfoStr: %w", err)
	}
	mapping := &MetricMapping{
		TotalQueuedRequests: queuedSpec,
		KVCacheUtilization:  kvUsageSpec,
		LoraRequestInfo:     loraReqInfoSpec,
	}

	return mapping, nil
}
