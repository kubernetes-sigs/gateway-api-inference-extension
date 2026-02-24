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

// Package requestcontrol provides request control plugins for GIE.
package requestcontrol

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"strings"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

const (
	// PrefillInjectionPluginType is the type of the PrefillInjectionPlugin
	PrefillInjectionPluginType = "prefill-injection-plugin"

	defaultPrefillProfile = "prefill"
)

type prefillInjectionPluginParameters struct {
	PrefillProfile string `json:"prefillProfile"`
	HeaderName     string `json:"headerNameOverride"` // Optional override
}

// compile-time type assertion
var _ requestcontrol.PreRequest = &PrefillInjectionPlugin{}

// PrefillInjectionPluginFactory defines the factory function for the PrefillInjectionPlugin
func PrefillInjectionPluginFactory(name string, rawParameters json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	parameters := prefillInjectionPluginParameters{
		PrefillProfile: defaultPrefillProfile,
	}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' pre-request plugin - %w", PrefillInjectionPluginType, err)
		}
	}
	return NewPrefillInjectionPlugin(parameters.PrefillProfile, parameters.HeaderName).WithName(name), nil
}

// NewPrefillInjectionPlugin initializes a new PrefillInjectionPlugin and returns its pointer.
func NewPrefillInjectionPlugin(prefillProfile, headerName string) *PrefillInjectionPlugin {
	if headerName == "" {
		headerName = metadata.PrefillEndpointsHeader
	}
	return &PrefillInjectionPlugin{
		typedName:      plugin.TypedName{Type: PrefillInjectionPluginType},
		prefillProfile: prefillProfile,
		headerName:     headerName,
	}
}

// PrefillInjectionPlugin PreRequest plugin
type PrefillInjectionPlugin struct {
	typedName      plugin.TypedName
	prefillProfile string
	headerName     string
}

// TypedName returns the typed name of the plugin.
func (p *PrefillInjectionPlugin) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin.
func (p *PrefillInjectionPlugin) WithName(name string) *PrefillInjectionPlugin {
	p.typedName.Name = name
	return p
}

// PreRequest wires prefill SchedulerProfile result into a header to indicate prefill worker
func (p *PrefillInjectionPlugin) PreRequest(_ context.Context, request *scheduling.LLMRequest, schedulingResult *scheduling.SchedulingResult) {
	if _, found := request.Headers[p.headerName]; found {
		request.Headers[p.headerName] = "" // clear header, if already set
	}

	prefillProfileRunResult, _ := schedulingResult.ProfileResults.PrefillResult(p.prefillProfile)
	if prefillProfileRunResult == nil {
		return // prefill profile failed to run or we chose not to run it, no-op in this case
	}

	if len(prefillProfileRunResult.TargetEndpoints) == 0 {
		return
	}

	var endpoints []string
	for _, target := range prefillProfileRunResult.TargetEndpoints {
		targetPod := target.GetMetadata()
		endpoints = append(endpoints, net.JoinHostPort(targetPod.Address, targetPod.Port))
	}

	// Join with comma (e.g. "10.1.2.3:8000,10.1.2.4:8000")
	request.Headers[p.headerName] = strings.Join(endpoints, ",")
}
