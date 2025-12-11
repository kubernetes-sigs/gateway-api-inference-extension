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

package plugins

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// noOpFactory is a helper for testing registration.
func noOpFactory(name string, _ json.RawMessage, _ Handle) (Plugin, error) {
	return nil, nil
}

func TestRegister(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		pluginType  string
		lifecycle   PluginLifecycle
		shouldPanic bool
	}{
		{
			name:       "Success - Registers Singleton",
			pluginType: "test-singleton-1",
			lifecycle:  LifecycleSingleton,
		},
		{
			name:       "Success - Registers Transient",
			pluginType: "test-transient-1",
			lifecycle:  LifecycleTransient,
		},
		{
			name:        "Panics - Duplicate Registration",
			pluginType:  "test-duplicate-1",
			lifecycle:   LifecycleSingleton,
			shouldPanic: true, // We will register it once in setup, then try again.
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			reg := PluginRegistration{
				Factory:   noOpFactory,
				Lifecycle: tc.lifecycle,
			}

			if tc.shouldPanic {
				// Pre-register the duplicate.
				RegisterWithMetadata(tc.pluginType, reg)

				assert.PanicsWithValue(t, "plugin type \"test-duplicate-1\" is already registered", func() {
					RegisterWithMetadata(tc.pluginType, reg)
				}, "expected RegisterWithMetadata to panic on duplicate registration")
			} else {
				assert.NotPanics(t, func() {
					RegisterWithMetadata(tc.pluginType, reg)
				}, "expected fresh registration to succeed")

				err := ValidatePluginRef(tc.pluginType, tc.lifecycle)
				assert.NoError(t, err, "expected plugin to be registered and valid")
			}
		})
	}
}

func TestValidatePluginRef(t *testing.T) {
	// We must register types globally to test validation.
	// We use a unique prefix to ensure this setup doesn't conflict with other tests.
	const (
		validSingleton = "val-singleton"
		validTransient = "val-transient"
	)

	// Setup Global State (Idempotent for this test file)
	// Since we can't unregister, we just ensure we don't panic if it ran before.
	defer func() { _ = recover() }()
	Register(validSingleton, noOpFactory)
	RegisterWithMetadata(validTransient, PluginRegistration{
		Factory:   noOpFactory,
		Lifecycle: LifecycleTransient,
	})

	t.Parallel()

	tests := []struct {
		name              string
		requestedType     string
		expectedLifecycle PluginLifecycle
		expectError       bool
		errorContains     string
	}{
		{
			name:              "Valid - Singleton Reference",
			requestedType:     validSingleton,
			expectedLifecycle: LifecycleSingleton,
			expectError:       false,
		},
		{
			name:              "Valid - Transient Reference",
			requestedType:     validTransient,
			expectedLifecycle: LifecycleTransient,
			expectError:       false,
		},
		{
			name:              "Error - Mismatched Lifecycle Singleton Expected Transient",
			requestedType:     validSingleton,
			expectedLifecycle: LifecycleTransient,
			expectError:       true,
			errorContains:     "has lifecycle 0, but expected 1",
		},
		{
			name:              "Error - Mismatched Lifecycle Transient Expected Singleton",
			requestedType:     validTransient,
			expectedLifecycle: LifecycleSingleton,
			expectError:       true,
			errorContains:     "has lifecycle 1, but expected 0",
		},
		{
			name:              "Error - Unknown Plugin Type",
			requestedType:     "non-existent-plugin",
			expectedLifecycle: LifecycleSingleton,
			expectError:       true,
			errorContains:     "not found in registry",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := ValidatePluginRef(tc.requestedType, tc.expectedLifecycle)

			if tc.expectError {
				require.Error(t, err, "expected ValidatePluginRef to fail")
				assert.Contains(t, err.Error(), tc.errorContains, "error message should match expected cause")
			} else {
				assert.NoError(t, err, "expected ValidatePluginRef to succeed")
			}
		})
	}
}
