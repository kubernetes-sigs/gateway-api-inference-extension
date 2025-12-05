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

package loader

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

// Define constants for test plugins.
// Constants must match those used in testdata_test.go.
const (
	testPluginType     = "test-plugin"
	testPickerType     = "test-picker"
	testScorerType     = "test-scorer"
	testProfileHandler = "test-profile-handler"
)

// --- Test: Phase 1 (Raw Loading & Static Defaults) ---

func TestLoadRawConfiguration(t *testing.T) {
	t.Parallel()

	// Register known feature gates for validation.
	RegisterFeatureGate(datalayer.FeatureGate)

	tests := []struct {
		name       string
		configText string
		want       *configapi.EndpointPickerConfig
		wantErr    bool
	}{
		{
			name:       "Success - Full Configuration",
			configText: successConfigText,
			want: &configapi.EndpointPickerConfig{
				TypeMeta: metav1.TypeMeta{
					Kind:       "EndpointPickerConfig",
					APIVersion: "inference.networking.x-k8s.io/v1alpha1",
				},
				Plugins: []configapi.PluginSpec{
					{Name: "test1", Type: testPluginType, Parameters: json.RawMessage(`{"threshold":10}`)},
					{Name: "profileHandler", Type: testProfileHandler},
					{Name: testScorerType, Type: testScorerType, Parameters: json.RawMessage(`{"blockSize":32}`)},
					{Name: "testPicker", Type: testPickerType},
				},
				SchedulingProfiles: []configapi.SchedulingProfile{
					{
						Name: "default",
						Plugins: []configapi.SchedulingPlugin{
							{PluginRef: "test1"},
							{PluginRef: testScorerType, Weight: ptr.To(50)},
							{PluginRef: "testPicker"},
						},
					},
				},
				FeatureGates: configapi.FeatureGates{datalayer.FeatureGate},
				SaturationDetector: &configapi.SaturationDetector{
					QueueDepthThreshold:       10,
					KVCacheUtilThreshold:      0.8,
					MetricsStalenessThreshold: metav1.Duration{Duration: 100 * time.Millisecond},
				},
			},
			wantErr: false,
		},
		{
			name:       "Success - No Profiles",
			configText: successNoProfilesText,
			want: &configapi.EndpointPickerConfig{
				TypeMeta: metav1.TypeMeta{
					Kind:       "EndpointPickerConfig",
					APIVersion: "inference.networking.x-k8s.io/v1alpha1",
				},
				Plugins: []configapi.PluginSpec{
					{Name: "test1", Type: testPluginType, Parameters: json.RawMessage(`{"threshold":10}`)},
				},
				FeatureGates: configapi.FeatureGates{},
			},
			wantErr: false,
		},
		{
			name:       "Error - Invalid YAML",
			configText: errorBadYamlText,
			wantErr:    true,
		},
		{
			name:       "Error - Unknown Feature Gate",
			configText: errorUnknownFeatureGateText,
			wantErr:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			logger := logging.NewTestLogger()

			got, _, err := LoadRawConfig([]byte(tc.configText), logger)

			if tc.wantErr {
				require.Error(t, err, "Expected LoadRawConfig to fail")
				return
			}
			require.NoError(t, err, "Expected LoadRawConfig to succeed")
			diff := cmp.Diff(tc.want, got, cmpopts.IgnoreFields(configapi.EndpointPickerConfig{}, "TypeMeta"))
			require.Empty(t, diff, "Config mismatch (-want +got):\n%s", diff)
		})
	}
}

// --- Test: Phase 2 (Instantiation, Architecture Injection, Deep Validation) ---

func TestInstantiateAndConfigure(t *testing.T) {
	// Not parallel because it modifies global plugin registry.
	registerTestPlugins(t)

	RegisterFeatureGate(datalayer.FeatureGate)

	tests := []struct {
		name       string
		configText string
		wantErr    bool
		validate   func(t *testing.T, handle plugins.Handle, cfg *configapi.EndpointPickerConfig)
	}{
		// --- Success Scenarios ---
		{
			name:       "Success - Complex Scheduler",
			configText: successSchedulerConfigText,
			wantErr:    false,
			validate: func(t *testing.T, handle plugins.Handle, cfg *configapi.EndpointPickerConfig) {
				// 1. Verify all explicit plugins exist in the registry
				require.NotNil(t, handle.Plugin("testScorer"), "Explicit scorer should be instantiated")
				require.NotNil(t, handle.Plugin("maxScorePicker"), "Explicit picker should be instantiated")
				require.NotNil(t, handle.Plugin("profileHandler"), "Explicit profile handler should be instantiated")

				// 2. Verify Profile Integrity
				// We explicitly defined a picker, so the defaulter should NOT have added a second one.
				require.Len(t, cfg.SchedulingProfiles, 1)
				require.Len(t, cfg.SchedulingProfiles[0].Plugins, 2,
					"Profile should have exactly 2 plugins (Scorer + Explicit Picker)")

				// 3. Verify Weight Propagation
				// The YAML specified weight: 50. Ensure it wasn't overwritten by defaults.
				scorerRef := cfg.SchedulingProfiles[0].Plugins[0]
				require.Equal(t, "testScorer", scorerRef.PluginRef)
				require.NotNil(t, scorerRef.Weight)
				require.Equal(t, 50, *scorerRef.Weight, "Explicit weight of 50 should be preserved")
			},
		},
		{
			name:       "Success - Default Scorer Weight",
			configText: successWithNoWeightText,
			wantErr:    false,
			validate: func(t *testing.T, _ plugins.Handle, cfg *configapi.EndpointPickerConfig) {
				require.Len(t, cfg.SchedulingProfiles, 1, "Unexpected profile structure")
				require.Len(t, cfg.SchedulingProfiles[0].Plugins, 2, "Expected Scorer + Default Picker")
				w := cfg.SchedulingProfiles[0].Plugins[0].Weight
				require.NotNil(t, w, "Weight should not be nil")
				require.Equal(t, 1, *w, "Expected default scorer weight of 1")
			},
		},
		{
			name:       "Success - Default Profile Handler Injection",
			configText: successWithNoProfileHandlersText,
			wantErr:    false,
			validate: func(t *testing.T, handle plugins.Handle, cfg *configapi.EndpointPickerConfig) {
				require.True(t, hasPluginType(handle, profile.SingleProfileHandlerType),
					"Defaults: SingleProfileHandler was not injected")
			},
		},

		// --- Instantiation Errors ---
		{
			name:       "Error (Instantiation) - Missing Type Field",
			configText: errorBadPluginReferenceText,
			wantErr:    true,
		},
		{
			name:       "Error (Instantiation) - Unknown Plugin Type",
			configText: errorBadPluginReferencePluginText,
			wantErr:    true,
		},
		{
			name:       "Error (Instantiation) - Invalid JSON Parameters",
			configText: errorBadPluginJsonText,
			wantErr:    true,
		},
		{
			name:       "Error (Instantiation) - Duplicate Plugin Name",
			configText: errorDuplicatePluginText,
			wantErr:    true,
		},

		// --- Deep Validation Errors ---
		{
			name:       "Error (Deep Validation) - Missing Profile Name",
			configText: errorNoProfileNameText,
			wantErr:    true,
		},
		{
			name:       "Error (Deep Validation) - Missing PluginRef in Profile",
			configText: errorBadProfilePluginText,
			wantErr:    true,
		},
		{
			name:       "Error (Deep Validation) - Profile References Undefined Plugin",
			configText: errorBadProfilePluginRefText,
			wantErr:    true,
		},
		{
			name:       "Error (Deep Validation) - Duplicate Profile Name",
			configText: errorDuplicateProfileText,
			wantErr:    true,
		},

		// --- Architectural Errors ---
		{
			name:       "Error (Architectural) - Two Pickers in One Profile",
			configText: errorTwoPickersText,
			wantErr:    true,
		},
		{
			name:       "Error (Deep Architectural) - Multiple Profile Handlers",
			configText: errorTwoProfileHandlersText,
			wantErr:    true,
		},
		{
			name:       "Error (Deep Architectural) - Missing Profile Handler",
			configText: errorNoProfileHandlersText,
			wantErr:    true,
		},
		{
			name:       "Error (Deep Architectural) - Multi-Profile with Single Handler",
			configText: errorMultiProfilesUseSingleProfileHandlerText,
			wantErr:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger := logging.NewTestLogger()

			// 1. Load Raw (Assuming valid yaml/structure for Phase 2 tests)
			rawConfig, _, err := LoadRawConfig([]byte(tc.configText), logger)
			if err != nil {
				// If we expected failure (and it failed early in Phase 1), success.
				if tc.wantErr {
					return
				}
				require.NoError(t, err, "Setup: LoadRawConfig failed")
			}

			// 2. Instantiate & Configure
			handle := utils.NewTestHandle(context.Background())
			_, err = InstantiateAndConfigure(rawConfig, handle, logger)

			if tc.wantErr {
				require.Error(t, err, "Expected InstantiateAndConfigure to fail")
				return
			}
			require.NoError(t, err, "Expected InstantiateAndConfigure to succeed")

			if tc.validate != nil {
				tc.validate(t, handle, rawConfig)
			}
		})
	}
}

// Add this new test function to verify the builder logic specifically
func TestBuildSaturationConfig(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    *configapi.SaturationDetector
		expected *saturationdetector.Config
	}{
		{
			name: "Valid Configuration",
			input: &configapi.SaturationDetector{
				QueueDepthThreshold:       20,
				KVCacheUtilThreshold:      0.9,
				MetricsStalenessThreshold: metav1.Duration{Duration: 500 * time.Millisecond},
			},
			expected: &saturationdetector.Config{
				QueueDepthThreshold:       20,
				KVCacheUtilThreshold:      0.9,
				MetricsStalenessThreshold: 500 * time.Millisecond,
			},
		},
		{
			name:  "Nil Input (Defaults)",
			input: nil,
			expected: &saturationdetector.Config{
				QueueDepthThreshold:       saturationdetector.DefaultQueueDepthThreshold,
				KVCacheUtilThreshold:      saturationdetector.DefaultKVCacheUtilThreshold,
				MetricsStalenessThreshold: saturationdetector.DefaultMetricsStalenessThreshold,
			},
		},
		{
			name: "Invalid Values (Fallback to Defaults)",
			input: &configapi.SaturationDetector{
				QueueDepthThreshold:       -5,
				KVCacheUtilThreshold:      1.5,
				MetricsStalenessThreshold: metav1.Duration{Duration: -10 * time.Second},
			},
			expected: &saturationdetector.Config{
				QueueDepthThreshold:       saturationdetector.DefaultQueueDepthThreshold,
				KVCacheUtilThreshold:      saturationdetector.DefaultKVCacheUtilThreshold,
				MetricsStalenessThreshold: saturationdetector.DefaultMetricsStalenessThreshold,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := buildSaturationConfig(tc.input)
			if diff := cmp.Diff(tc.expected, got); diff != "" {
				t.Errorf("buildSaturationConfig mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// --- Helpers & Mocks ---

func hasPluginType(handle plugins.Handle, typeName string) bool {
	for _, p := range handle.GetAllPlugins() {
		if p.TypedName().Type == typeName {
			return true
		}
	}
	return false
}

type mockPlugin struct {
	t plugins.TypedName
}

func (m *mockPlugin) TypedName() plugins.TypedName { return m.t }

// Mock Scorer
type mockScorer struct{ mockPlugin }

func (m *mockScorer) Score(context.Context, *types.CycleState, *types.LLMRequest, []types.Pod) map[types.Pod]float64 {
	return nil
}

// Mock Picker
type mockPicker struct{ mockPlugin }

func (m *mockPicker) Pick(context.Context, *types.CycleState, []*types.ScoredPod) *types.ProfileRunResult {
	return nil
}

// Mock Handler
type mockHandler struct{ mockPlugin }

func (m *mockHandler) Pick(
	context.Context,
	*types.CycleState,
	*types.LLMRequest,
	map[string]*framework.SchedulerProfile,
	map[string]*types.ProfileRunResult,
) map[string]*framework.SchedulerProfile {
	return nil
}
func (m *mockHandler) ProcessResults(
	context.Context,
	*types.CycleState,
	*types.LLMRequest,
	map[string]*types.ProfileRunResult,
) (*types.SchedulingResult, error) {
	return nil, nil
}

func registerTestPlugins(t *testing.T) {
	t.Helper()

	// Helper to generate simple factories.
	register := func(name string, factory plugins.FactoryFunc) {
		plugins.Register(name, factory)
	}

	mockFactory := func(tType string) plugins.FactoryFunc {
		return func(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
			return &mockPlugin{t: plugins.TypedName{Name: name, Type: tType}}, nil
		}
	}

	// Register standard test mocks.
	register(testPluginType, mockFactory(testPluginType))

	plugins.Register(testScorerType, func(name string, params json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
		// Attempt to unmarshal to trigger errors for invalid JSON in tests.
		if len(params) > 0 {
			var p struct {
				BlockSize int `json:"blockSize"`
			}
			if err := json.Unmarshal(params, &p); err != nil {
				return nil, err
			}
		}
		return &mockScorer{mockPlugin{t: plugins.TypedName{Name: name, Type: testScorerType}}}, nil
	})

	plugins.Register(testPickerType, func(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
		return &mockPicker{mockPlugin{t: plugins.TypedName{Name: name, Type: testPickerType}}}, nil
	})

	plugins.Register(testProfileHandler, func(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
		return &mockHandler{mockPlugin{t: plugins.TypedName{Name: name, Type: testProfileHandler}}}, nil
	})

	// Ensure system defaults are registered too.
	plugins.Register(picker.MaxScorePickerType, picker.MaxScorePickerFactory)
	plugins.Register(profile.SingleProfileHandlerType, profile.SingleProfileHandlerFactory)
}
