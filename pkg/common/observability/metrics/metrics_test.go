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
	"testing"

	compbasemetrics "k8s.io/component-base/metrics"
)

func TestHelpMsgWithStability(t *testing.T) {
	tests := []struct {
		name      string
		msg       string
		stability compbasemetrics.StabilityLevel
		expected  string
	}{
		{
			name:      "ALPHA stability",
			msg:       "tracks request count",
			stability: compbasemetrics.ALPHA,
			expected:  "[ALPHA] tracks request count",
		},
		{
			name:      "BETA stability",
			msg:       "tracks request count",
			stability: compbasemetrics.BETA,
			expected:  "[BETA] tracks request count",
		},
		{
			name:      "STABLE stability",
			msg:       "tracks request count",
			stability: compbasemetrics.STABLE,
			expected:  "[STABLE] tracks request count",
		},
		{
			name:      "INTERNAL stability",
			msg:       "internal use only",
			stability: compbasemetrics.INTERNAL,
			expected:  "[INTERNAL] internal use only",
		},
		{
			name:      "empty message",
			msg:       "",
			stability: compbasemetrics.ALPHA,
			expected:  "[ALPHA] ",
		},
		{
			name:      "message with special characters",
			msg:       "p99 latency (ms) [experimental]",
			stability: compbasemetrics.BETA,
			expected:  "[BETA] p99 latency (ms) [experimental]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := HelpMsgWithStability(tt.msg, tt.stability)
			if result != tt.expected {
				t.Errorf("HelpMsgWithStability(%q, %q) = %q, want %q", tt.msg, tt.stability, result, tt.expected)
			}
		})
	}
}
