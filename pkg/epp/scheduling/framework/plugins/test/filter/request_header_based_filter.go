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
	"net"
	"strings"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/test"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	// HeaderBasedTestingFilterType is the filter type that is used in plugins registry.
	HeaderBasedTestingFilterType = "header-based-testing-filter"
)

// compile-time type assertion
var _ framework.Filter = &HeaderBasedTestingFilter{}

// HeaderBasedTestingFilterFactory defines the factory function for HeaderBasedTestingFilter.
func HeaderBasedTestingFilterFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewHeaderBasedTestingFilter().WithName(name), nil
}

// NewHeaderBasedTestingFilter initializes a new HeaderBasedTestingFilter.
// This should only be used for testing purposes.
func NewHeaderBasedTestingFilter() *HeaderBasedTestingFilter {
	return &HeaderBasedTestingFilter{
		typedName: plugins.TypedName{Type: HeaderBasedTestingFilterType, Name: HeaderBasedTestingFilterType},
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

// WithName sets the name of the filter.
func (f *HeaderBasedTestingFilter) WithName(name string) *HeaderBasedTestingFilter {
	f.typedName.Name = name
	return f
}

// Filter selects pods whose IP or IP:port matches any value in the
// "test-epp-endpoint-selection" header. Values may be "IP" or "IP:port".
// If a port is provided, only an exact IP:port match is accepted.
func (f *HeaderBasedTestingFilter) Filter(_ context.Context, _ *types.CycleState, request *types.LLMRequest, pods []types.Pod) []types.Pod {
	hv, ok := request.Headers[test.HeaderTestEppEndPointSelectionKey]
	if !ok || strings.TrimSpace(hv) == "" {
		return []types.Pod{}
	}

	normalizeIP := func(s string) string { return strings.Trim(s, "[]") }

	// Build lookup maps:
	//   ip -> pod
	//   ip:port -> pod (only when pod GetPort() is non-empty)
	ipToPod := make(map[string]types.Pod, len(pods))
	hpToPod := make(map[string]types.Pod, len(pods))
	for _, p := range pods {
		if p == nil || p.GetPod() == nil {
			continue
		}
		ip := normalizeIP(strings.TrimSpace(p.GetPod().GetIPAddress()))
		if ip == "" {
			continue
		}
		ipToPod[ip] = p
		if port := strings.TrimSpace(p.GetPod().GetPort()); port != "" {
			hpToPod[ip+":"+port] = p
		}
	}

	headerVals := strings.Split(hv, ",")
	filteredPods := make([]types.Pod, 0, len(headerVals))
	seen := make(map[string]struct{}, len(headerVals)) // de-dupe by pod IP

	for _, raw := range headerVals {
		item := strings.TrimSpace(raw)
		if item == "" {
			continue
		}

		host := item
		port := ""
		if h, pt, err := net.SplitHostPort(item); err == nil {
			host, port = h, pt
		} else {
			host = normalizeIP(host) // bare IP, possibly bracketed IPv6
		}
		host = normalizeIP(host)

		var pod types.Pod
		if port != "" {
			// Require an exact ip:port match
			if p, ok := hpToPod[host+":"+port]; ok {
				pod = p
			}
		} else {
			// IP-only selection
			if p, ok := ipToPod[host]; ok {
				pod = p
			}
		}

		if pod != nil {
			ip := normalizeIP(pod.GetPod().GetIPAddress())
			if _, dup := seen[ip]; !dup {
				seen[ip] = struct{}{}
				filteredPods = append(filteredPods, pod)
			}
		}
	}
	return filteredPods
}
