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
	"strings"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	// headerTestEppEndPointSelectionKey is the header used for testing purposes to make EPP behavior controllable.
	// The header value should be a comma-separated list of endpoint IP addresses.
	// E.g., "test-epp-endpoint-selection": "10.0.0.7,10.0.0.8"
	// The returned order is the same as the order provided in the header.
	headerTestEppEndPointSelectionKey = "test-epp-endpoint-selection"
)

// compile-time type assertion
var _ framework.Filter = &HeaderBasedTestingFilter{}

// NewHeaderBasedTestingFilter initializes a new HeaderBasedTestingFilter.
// This should only be used for testing purposes.
func NewHeaderBasedTestingFilter() *HeaderBasedTestingFilter {
	return &HeaderBasedTestingFilter{
		typedName: plugins.TypedName{Type: "header-based-testing", Name: "header-based-testing-filter"},
	}
}

// HeaderBasedTestingFilter filters Pods based on an address specified in the "test-epp-endpoint-selection" request header.
type HeaderBasedTestingFilter struct {
	typedName plugins.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (f *HeaderBasedTestingFilter) TypedName() plugins.TypedName {
	return f.typedName
}

// Filter selects pods that match the IP addresses specified in the request header.
func (f *HeaderBasedTestingFilter) Filter(_ context.Context, _ *types.CycleState, request *types.LLMRequest, pods []types.Pod) []types.Pod {
	headerValue, ok := request.Headers[headerTestEppEndPointSelectionKey]
	if !ok || headerValue == "" {
		return []types.Pod{}
	}

	podAddressMap := make(map[string]types.Pod, len(pods))
	for _, pod := range pods {
		podAddressMap[pod.GetPod().Address] = pod
	}

	endpoints := strings.Split(headerValue, ",")
	filteredPods := make([]types.Pod, 0, len(endpoints))
	for _, endpoint := range endpoints {
		trimmedEndpoint := strings.TrimSpace(endpoint)
		if pod, found := podAddressMap[trimmedEndpoint]; found {
			filteredPods = append(filteredPods, pod)
		}
	}
	return filteredPods
}
