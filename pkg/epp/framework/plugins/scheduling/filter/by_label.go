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

package filter

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	// ByLabelType is the type of the ByLabel filter
	ByLabelType = "by-label"
)

type byLabelParameters struct {
	Label         string   `json:"label"`
	ValidValues   []string `json:"validValues"`
	AllowsNoLabel bool     `json:"allowsNoLabel"`
}

var _ scheduling.Filter = &ByLabel{} // validate interface conformance

// ByLabelFactory defines the factory function for the ByLabel filter.
func ByLabelFactory(name string, rawParameters json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	parameters := byLabelParameters{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", ByLabelType, err)
		}
	}
	if name == "" {
		return nil, fmt.Errorf("invalid configuration for '%s' filter: name cannot be empty", ByLabelType)
	}
	if parameters.Label == "" {
		return nil, fmt.Errorf("invalid configuration for '%s' filter: 'label' must be specified", ByLabelType)
	}
	if len(parameters.ValidValues) == 0 && !parameters.AllowsNoLabel {
		return nil, fmt.Errorf("invalid configuration for '%s' "+
			"filter: either 'validValues' must be non-empty or 'allowsNoLabel' must be true", ByLabelType)
	}
	return NewByLabel(name, parameters.Label, parameters.AllowsNoLabel, parameters.ValidValues...), nil
}

// NewByLabel creates and returns an instance of the RoleBasedFilter based on the input parameters
// name - the filter name
// labelName - the name of the label to use
// allowsNoLabel - if true endpoints without given label will be considered as valid (not filtered out)
// validValuesApp - list of valid values
func NewByLabel(name string, labelName string, allowsNoLabel bool, validValues ...string) *ByLabel {
	validValuesMap := map[string]struct{}{}

	for _, v := range validValues {
		validValuesMap[v] = struct{}{}
	}

	return &ByLabel{
		typedName:     plugin.TypedName{Type: ByLabelType, Name: name},
		labelName:     labelName,
		allowsNoLabel: allowsNoLabel,
		validValues:   validValuesMap,
	}
}

// ByLabel - filters out endpoints based on the values defined by the given label
type ByLabel struct {
	// name defines the filter typed name
	typedName plugin.TypedName
	// labelName defines the name of the label to be checked
	labelName string
	// validValues defines list of valid label values
	validValues map[string]struct{}
	// allowsNoLabel - if true endpoints without given label will be considered as valid (not filtered out)
	allowsNoLabel bool
}

// TypedName returns the typed name of the plugin
func (f *ByLabel) TypedName() plugin.TypedName {
	return f.typedName
}

// WithName sets the name of the plugin.
func (f *ByLabel) WithName(name string) *ByLabel {
	f.typedName.Name = name
	return f
}

// Filter filters out all endpoints that are not marked with one of roles from the validRoles collection
// or has no role label in case allowsNoRolesLabel is true
func (f *ByLabel) Filter(_ context.Context, _ *scheduling.CycleState, _ *scheduling.LLMRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	filteredEndpoints := []scheduling.Endpoint{}

	for _, endpoint := range endpoints {
		val, labelDefined := endpoint.GetMetadata().Labels[f.labelName]
		_, valueExists := f.validValues[val]

		if (!labelDefined && f.allowsNoLabel) || valueExists {
			filteredEndpoints = append(filteredEndpoints, endpoint)
		}
	}

	return filteredEndpoints
}
