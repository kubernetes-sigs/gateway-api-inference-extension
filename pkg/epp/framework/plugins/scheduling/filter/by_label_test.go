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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestByLabelFactory(t *testing.T) {
	// Define test constants locally since pd_role.go is removed
	const (
		RoleLabel   = "llm-d.ai/role"
		RolePrefill = "prefill"
		RoleDecode  = "decode"
		RoleBoth    = "both"
	)

	tests := []struct {
		name       string
		pluginName string
		jsonParams string
		expectErr  bool
	}{
		{
			name:       "valid configuration with non-empty validValues",
			pluginName: "valid-filter",
			jsonParams: fmt.Sprintf(`{
				"label": %q,
				"allowsNoLabel": false,
				"validValues": [%q]
			}`, RoleLabel, RolePrefill),
			expectErr: false,
		},
		{
			name:       "allowsNoLabel true with empty validValues",
			pluginName: "allow-no-label",
			jsonParams: fmt.Sprintf(`{
				"label": %q,
				"allowsNoLabel": true,
				"validValues": []
			}`, RoleLabel),
			expectErr: false,
		},
		{
			name:       "allowsNoLabel true with multiple valid roles",
			pluginName: "mixed-mode",
			jsonParams: fmt.Sprintf(`{
				"label": %q,
				"allowsNoLabel": true,
				"validValues": [%q, %q]
			}`, RoleLabel, RoleDecode, RoleBoth),
			expectErr: false,
		},
		{
			name:       "empty label name should error",
			pluginName: "empty-label",
			jsonParams: fmt.Sprintf(`{
				"label": "",
				"allowsNoLabel": false,
				"validValues": [%q]
			}`, RolePrefill),
			expectErr: true,
		},
		{
			name:       "missing label field should error",
			pluginName: "missing-label",
			jsonParams: fmt.Sprintf(`{
				"allowsNoLabel": false,
				"validValues": [%q]
			}`, RolePrefill),
			expectErr: true,
		},
		{
			name:       "contradictory config: empty validValues and allowsNoLabel=false",
			pluginName: "invalid-contradiction",
			jsonParams: fmt.Sprintf(`{
				"label": %q,
				"allowsNoLabel": false,
				"validValues": []
			}`, RoleLabel),
			expectErr: true,
		},
		{
			name:       "contradictory config: no validValues field and allowsNoLabel=false",
			pluginName: "no-valid-values",
			jsonParams: fmt.Sprintf(`{
				"label": %q,
				"allowsNoLabel": false
			}`, RoleLabel),
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rawParams := json.RawMessage(tt.jsonParams)
			plugin, err := ByLabelFactory(tt.pluginName, rawParams, nil)

			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, plugin)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, plugin)
			}
		})
	}
}

func TestByLabelFactoryInvalidJSON(t *testing.T) {
	invalidTests := []struct {
		name       string
		jsonParams string
	}{
		{
			name:       "malformed JSON",
			jsonParams: `{"label": "app", "validValues": ["a"`, // missing closing ]
		},
		{
			name:       "validValues as string instead of array",
			jsonParams: `{"label": "app", "validValues": "true"}`,
		},
		{
			name:       "allowsNoLabel as string",
			jsonParams: `{"label": "app", "allowsNoLabel": "yes", "validValues": ["true"]}`,
		},
	}

	for _, tt := range invalidTests {
		t.Run(tt.name, func(t *testing.T) {
			rawParams := json.RawMessage(tt.jsonParams)
			plugin, err := ByLabelFactory("test", rawParams, nil)

			assert.Error(t, err)
			assert.Nil(t, plugin)
		})
	}
}

// Helper functions
func createEndpoint(nsn k8stypes.NamespacedName, ipaddr string, labels map[string]string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: nsn,
			Address:        ipaddr,
			Labels:         labels,
		},
		&fwkdl.Metrics{},
		nil,
	)
}

func TestByLabelFiltering(t *testing.T) {
	endpoints := []scheduling.Endpoint{
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "nginx-1"},
			"10.0.0.1",
			map[string]string{
				"app":     "nginx",
				"version": "v1.0",
				"tier":    "frontend",
			}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "nginx-2"},
			"10.0.0.2",
			map[string]string{
				"app":     "nginx",
				"version": "v1.1",
				"tier":    "frontend",
			}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "kube-system", Name: "coredns-1"},
			"10.0.0.3",
			map[string]string{
				"app":  "coredns",
				"tier": "system",
			}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "redis-1"},
			"10.0.0.4",
			map[string]string{
				"app":        "redis",
				"tier":       "backend",
				"deprecated": "true",
			}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "web-1"},
			"10.0.0.5",
			map[string]string{
				"app":         "web",
				"tier":        "frontend",
				"environment": "production",
			}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "no-tier-pod"},
			"10.0.0.6",
			map[string]string{
				"app": "unknown",
			}),
	}

	tests := []struct {
		testName      string
		label         string
		validValues   []string
		allowsNoLabel bool
		expectedPods  []string // pod names that should match
	}{
		{
			testName:      "match app nginx",
			label:         "app",
			validValues:   []string{"nginx"},
			expectedPods:  []string{"nginx-1", "nginx-2"},
			allowsNoLabel: false,
		},
		{
			testName:      "match exact version v1.0",
			label:         "version",
			validValues:   []string{"v1.0"},
			expectedPods:  []string{"nginx-1"},
			allowsNoLabel: false,
		},
		{
			testName:      "allow pods without 'tier' label",
			label:         "tier",
			allowsNoLabel: true,
			expectedPods:  []string{"no-tier-pod"}, // only "no-tier-pod" doesn't have a "tier" label
		},
		{
			testName:      "allow all known tier values",
			label:         "tier",
			validValues:   []string{"frontend", "backend", "system"},
			allowsNoLabel: false,
			expectedPods:  []string{"nginx-1", "nginx-2", "coredns-1", "redis-1", "web-1"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.testName, func(t *testing.T) {
			rawParams, err := json.Marshal(byLabelParameters{
				Label:         tt.label,
				ValidValues:   tt.validValues,
				AllowsNoLabel: tt.allowsNoLabel,
			})
			require.NoError(t, err)

			plugin, err := ByLabelFactory("test-label", rawParams, nil)
			require.NoError(t, err)
			require.NotNil(t, plugin)

			blf, ok := plugin.(*ByLabel)
			require.True(t, ok, "plugin should be of type *ByLabel")

			// Use context.Background() as standard context
			filteredEndpoints := blf.Filter(context.Background(), nil, nil, endpoints)

			// We need to compare endpoints. Since the original test used namespaced names comparison
			// which is simpler, let's keep it but also use cmp if we want more robust comparison.
			// Reusing the original test's simple name comparison logic.
			actualEndpointNames := make([]string, len(filteredEndpoints))
			for idx, endpoint := range filteredEndpoints {
				actualEndpointNames[idx] = endpoint.GetMetadata().NamespacedName.Name
			}

			assert.ElementsMatch(t, tt.expectedPods, actualEndpointNames,
				"filtered endpoints should match expected endpoints")
			assert.Len(t, filteredEndpoints, len(tt.expectedPods),
				"filtered endpoints count should match expected count")

			// Optional: use cmp to double check if we can
			// Not strictly necessary if ElementsMatch works effectively
		})
	}
}
