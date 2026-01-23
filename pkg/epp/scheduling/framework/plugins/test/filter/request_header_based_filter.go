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

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/test"
)

const (
	// HeaderBasedTestingFilterType is the filter type that is used in plugins registry.
	HeaderBasedTestingFilterType = "header-based-testing-filter"
)

// compile-time type assertion
var _ framework.Filter = &HeaderBasedTestingFilter{}

// HeaderBasedTestingFilterFactory defines the factory function for HeaderBasedTestingFilter.
func HeaderBasedTestingFilterFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewHeaderBasedTestingFilter().WithName(name), nil
}

// NewHeaderBasedTestingFilter initializes a new HeaderBasedTestingFilter.
// This should only be used for testing purposes.
func NewHeaderBasedTestingFilter() *HeaderBasedTestingFilter {
	return &HeaderBasedTestingFilter{
		typedName: fwkplugin.TypedName{Type: HeaderBasedTestingFilterType, Name: HeaderBasedTestingFilterType},
	}
}

// HeaderBasedTestingFilter filters Endpoints based on an address specified in the "test-epp-endpoint-selection" request header.
type HeaderBasedTestingFilter struct {
	typedName fwkplugin.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (f *HeaderBasedTestingFilter) TypedName() fwkplugin.TypedName {
	return f.typedName
}

// WithName sets the name of the filter.
func (f *HeaderBasedTestingFilter) WithName(name string) *HeaderBasedTestingFilter {
	f.typedName.Name = name
	return f
}

// Filter selects endpoints whose IP or IP:port matches any value in the
// "test-epp-endpoint-selection" header. Values may be "IP" or "IP:port".
// If a port is provided, only an exact IP:port match is accepted.
func (f *HeaderBasedTestingFilter) Filter(_ context.Context, _ *framework.CycleState, request *framework.LLMRequest, endpoints []framework.Endpoint) []framework.Endpoint {
	hv, ok := request.Headers[test.HeaderTestEppEndPointSelectionKey]
	if !ok || strings.TrimSpace(hv) == "" {
		return []framework.Endpoint{}
	}

	normalizeIP := func(s string) string { return strings.Trim(s, "[]") }

	// Build lookup maps:
	//   ip -> endpoint
	//   ip:port -> endpoint (only when endpoint GetPort() is non-empty)
	ipToEndpoint := make(map[string]framework.Endpoint, len(endpoints))
	hpToPod := make(map[string]framework.Endpoint, len(endpoints))
	for _, e := range endpoints {
		if e == nil || e.GetMetadata() == nil {
			continue
		}
		ip := normalizeIP(strings.TrimSpace(e.GetMetadata().GetIPAddress()))
		if ip == "" {
			continue
		}
		ipToEndpoint[ip] = e
		if port := strings.TrimSpace(e.GetMetadata().GetPort()); port != "" {
			hpToPod[ip+":"+port] = e
		}
	}

	headerVals := strings.Split(hv, ",")
	filteredEndpoints := make([]framework.Endpoint, 0, len(headerVals))
	seen := make(map[string]struct{}, len(headerVals)) // de-dupe by endpoint IP

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

		var endpoint framework.Endpoint
		if port != "" {
			// Require an exact ip:port match
			if p, ok := hpToPod[host+":"+port]; ok {
				endpoint = p
			}
		} else {
			// IP-only selection
			if p, ok := ipToEndpoint[host]; ok {
				endpoint = p
			}
		}

		if endpoint != nil {
			ip := normalizeIP(endpoint.GetMetadata().GetIPAddress())
			if _, dup := seen[ip]; !dup {
				seen[ip] = struct{}{}
				filteredEndpoints = append(filteredEndpoints, endpoint)
			}
		}
	}
	return filteredEndpoints
}
