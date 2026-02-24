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

package requestcontrol

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestPrefillInjectionPluginFactory(t *testing.T) {
	tests := []struct {
		name       string
		pluginName string
		params     string
		expectErr  bool
	}{
		{
			name:       "valid defaults",
			pluginName: "default-plugin",
			params:     `{}`,
			expectErr:  false,
		},
		{
			name:       "valid custom",
			pluginName: "custom-plugin",
			params:     `{"prefillProfile": "my-prefill", "headerNameOverride": "x-custom"}`,
			expectErr:  false,
		},
		{
			name:       "invalid json",
			pluginName: "invalid-json",
			params:     `{invalid}`,
			expectErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := PrefillInjectionPluginFactory(tt.pluginName, json.RawMessage(tt.params), nil)
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, p)

				if tt.name == "valid custom" {
					plugin := p.(*PrefillInjectionPlugin)
					assert.Equal(t, "my-prefill", plugin.prefillProfile)
					assert.Equal(t, "x-custom", plugin.headerName)
				}
			}
		})
	}
}

// createEndpoint creates a mock Endpoint with customizable IP and port.
func createEndpoint(nsn k8stypes.NamespacedName, ipaddr, port string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: nsn,
			Address:        ipaddr,
			Port:           port,
		},
		nil,
		fwkdl.NewAttributes(),
	)
}

func TestPrefillInjectionPlugin_PreRequest(t *testing.T) {
	defaultProfile := "prefill"
	defaultHeader := "x-gateway-prefill-endpoints"

	tests := []struct {
		name             string
		prefillProfile   string
		headerName       string
		schedulingResult *scheduling.SchedulingResult
		existingHeader   string
		expectedHeader   string
	}{
		{
			name:           "no prefill result -> no header",
			prefillProfile: defaultProfile,
			headerName:     defaultHeader,
			schedulingResult: &scheduling.SchedulingResult{
				ProfileResults: map[string]*scheduling.ProfileRunResult{},
			},
			existingHeader: "",
			expectedHeader: "",
		},
		{
			name:           "prefill result exists -> header set",
			prefillProfile: defaultProfile,
			headerName:     defaultHeader,
			schedulingResult: &scheduling.SchedulingResult{
				ProfileResults: map[string]*scheduling.ProfileRunResult{
					defaultProfile: {
						TargetEndpoints: []scheduling.Endpoint{
							createEndpoint(k8stypes.NamespacedName{Name: "pod1"}, "10.0.0.1", "8000"),
						},
					},
				},
			},
			existingHeader: "",
			expectedHeader: "10.0.0.1:8000",
		},
		{
			name:           "prefill result exists with custom header -> header set",
			prefillProfile: defaultProfile,
			headerName:     "x-custom-prefill",
			schedulingResult: &scheduling.SchedulingResult{
				ProfileResults: map[string]*scheduling.ProfileRunResult{
					defaultProfile: {
						TargetEndpoints: []scheduling.Endpoint{
							createEndpoint(k8stypes.NamespacedName{Name: "pod1"}, "1.2.3.4", "9090"),
						},
					},
				},
			},
			existingHeader: "",
			expectedHeader: "1.2.3.4:9090",
		},
		{
			name:           "existing header cleared if no prefill",
			prefillProfile: defaultProfile,
			headerName:     defaultHeader,
			schedulingResult: &scheduling.SchedulingResult{
				ProfileResults: map[string]*scheduling.ProfileRunResult{},
			},
			existingHeader: "old-value",
			expectedHeader: "",
		},
		{
			name:           "existing header overwritten",
			prefillProfile: defaultProfile,
			headerName:     defaultHeader,
			schedulingResult: &scheduling.SchedulingResult{
				ProfileResults: map[string]*scheduling.ProfileRunResult{
					defaultProfile: {
						TargetEndpoints: []scheduling.Endpoint{
							createEndpoint(k8stypes.NamespacedName{Name: "pod1"}, "10.0.0.1", "8000"),
						},
					},
				},
			},
			existingHeader: "old-value",
			expectedHeader: "10.0.0.1:8000",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewPrefillInjectionPlugin(tt.prefillProfile, tt.headerName)
			req := &scheduling.LLMRequest{
				Headers: map[string]string{},
			}
			if tt.existingHeader != "" {
				req.Headers[tt.headerName] = tt.existingHeader
			}

			p.PreRequest(context.Background(), req, tt.schedulingResult)

			assert.Equal(t, tt.expectedHeader, req.Headers[tt.headerName])
		})
	}
}

// Check interfaces
var _ plugin.Plugin = &PrefillInjectionPlugin{}
