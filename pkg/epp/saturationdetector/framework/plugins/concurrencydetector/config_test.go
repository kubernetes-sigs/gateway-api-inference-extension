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

package concurrencydetector

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

func TestNewConfigFromAPI(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    *configapi.SaturationDetector
		expected *Config
	}{
		{
			name: "Valid Configuration",
			input: &configapi.SaturationDetector{
				Concurrency: &configapi.ConcurrencySaturationDetectorConfig{
					MaxConcurrency: 200,
					Headroom:       0.3,
				},
			},
			expected: &Config{
				MaxConcurrency: 200,
				Headroom:       0.3,
			},
		},
		{
			name:  "Nil Input (Defaults)",
			input: nil,
			expected: &Config{
				MaxConcurrency: DefaultMaxConcurrency,
				Headroom:       DefaultHeadroom,
			},
		},
		{
			name: "Nil Concurrency Config (Defaults)",
			input: &configapi.SaturationDetector{
				Concurrency: nil,
			},
			expected: &Config{
				MaxConcurrency: DefaultMaxConcurrency,
				Headroom:       DefaultHeadroom,
			},
		},
		{
			name: "Invalid Values (Fallback to Defaults)",
			input: &configapi.SaturationDetector{
				Concurrency: &configapi.ConcurrencySaturationDetectorConfig{
					MaxConcurrency: -10,
					Headroom:       1.5,
				},
			},
			expected: &Config{
				MaxConcurrency: DefaultMaxConcurrency,
				Headroom:       DefaultHeadroom,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := NewConfigFromAPI(tc.input)
			if diff := cmp.Diff(tc.expected, got); diff != "" {
				t.Errorf("NewConfigFromAPI mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
