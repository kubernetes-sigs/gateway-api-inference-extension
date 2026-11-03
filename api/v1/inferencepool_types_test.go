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

package v1

import "testing"

func TestPortNumberForRole(t *testing.T) {
	tests := []struct {
		name  string
		ports []Port
		role  PortRole
		want  PortNumber
	}{
		{
			name:  "unspecified role defaults to serving and is returned for Serving",
			ports: []Port{{Number: 8000}},
			role:  PortRoleServing,
			want:  8000,
		},
		{
			name:  "falls back to serving port when requested role absent",
			ports: []Port{{Number: 8000, Role: PortRoleServing}},
			role:  PortRoleMetrics,
			want:  8000,
		},
		{
			name:  "falls back to unspecified-role port treated as serving",
			ports: []Port{{Number: 8000}},
			role:  PortRoleHealth,
			want:  8000,
		},
		{
			name: "dedicated port for role wins over serving fallback",
			ports: []Port{
				{Number: 8000, Role: PortRoleServing},
				{Number: 9090, Role: PortRoleMetrics},
			},
			role: PortRoleMetrics,
			want: 9090,
		},
		{
			name: "falls back to the first serving port, not the last",
			ports: []Port{
				{Number: 8000, Role: PortRoleServing},
				{Number: 8001, Role: PortRoleServing},
			},
			role: PortRoleMetrics,
			want: 8000,
		},
		{
			name:  "returns 0 when the spec has no serving port",
			ports: []Port{{Number: 9090, Role: PortRoleMetrics}},
			role:  PortRoleHealth,
			want:  0,
		},
		{
			name: "dedicated health port is independent of metrics port",
			ports: []Port{
				{Number: 8000, Role: PortRoleServing},
				{Number: 9090, Role: PortRoleMetrics},
				{Number: 8080, Role: PortRoleHealth},
			},
			role: PortRoleHealth,
			want: 8080,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			spec := InferencePoolSpec{TargetPorts: tc.ports}
			if got := spec.PortNumberForRole(tc.role); got != tc.want {
				t.Errorf("PortNumberForRole(%v) = %v, want %v", tc.role, got, tc.want)
			}
		})
	}
}
