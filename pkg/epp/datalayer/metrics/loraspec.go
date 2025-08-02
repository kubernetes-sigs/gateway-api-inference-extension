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

	dto "github.com/prometheus/client_model/go"
)

// LoRASpec extends the standard Spec to allow special case
// handling for retrieving the latest metrics value for LoRAs.
type LoRASpec struct {
	*Spec
}

// parseStringToLoRASpec parses the metric specification but
// wraps the return in a LoRASpec.
func parseStringToLoRASpec(spec string) (*LoRASpec, error) {
	baseSpec, err := parseStringToSpec(spec)
	if err != nil {
		return nil, err
	}
	return &LoRASpec{
		Spec: baseSpec,
	}, nil
}

// getLatestMetric retrieves the latest LoRA metric based on Spec.
// We can't use the standard Spec method since, in the case of
// LoRA (i.e., `vllm:lora_requests_info`), each label key-value pair permutation
// generates new series and only most recent should be used. The value of each
// series is its creation timestamp so we can retrieve the latest by sorting on
// that the value first.
func (spec *LoRASpec) getLatestMetric(families PrometheusMetricMap) (*dto.Metric, error) {
	family, err := extractFamily(spec.Spec, families)
	if err != nil {
		return nil, err
	}

	var latest *dto.Metric
	var recent float64 = -1

	for _, metric := range family.GetMetric() {
		if spec.labelsMatch(metric.GetLabel()) {
			value := extractValue(metric) // metric value is its creation timestamp
			if value > recent {
				recent = value
				latest = metric
			}
		}
	}

	if latest == nil {
		return nil, fmt.Errorf("no matching lora metric found for %q with labels %v", spec.Name, spec.Labels)
	}

	return latest, nil
}
