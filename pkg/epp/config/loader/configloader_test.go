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
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

const (
	testProfileHandlerType = "test-profile-handler"
	test1Type              = "test-one"
	test2Type              = "test-two"
	testPickerType         = "test-picker"
)

type testStruct struct {
	name       string
	configText string
	configFile string
	want       *configapi.EndpointPickerConfig
	wantErr    bool
}

func TestLoadRawConfiguration(t *testing.T) {
	registerTestPlugins()

	goodConfig := &configapi.EndpointPickerConfig{
		TypeMeta: metav1.TypeMeta{
			Kind:       "EndpointPickerConfig",
			APIVersion: "inference.networking.x-k8s.io/v1alpha1",
		},
		Plugins: []configapi.PluginSpec{
			{
				Name:       "test1",
				Type:       test1Type,
				Parameters: json.RawMessage("{\"threshold\":10}"),
			},
			{
				Name: "profileHandler",
				Type: "test-profile-handler",
			},
			{
				Type:       test2Type,
				Parameters: json.RawMessage("{\"hashBlockSize\":32}"),
			},
			{
				Name: "testPicker",
				Type: testPickerType,
			},
		},
		SchedulingProfiles: []configapi.SchedulingProfile{
			{
				Name: "default",
				Plugins: []configapi.SchedulingPlugin{
					{
						PluginRef: "test1",
					},
					{
						PluginRef: "test-two",
						Weight:    ptr.To(50),
					},
					{
						PluginRef: "testPicker",
					},
				},
			},
		},
	}

	goodConfigNoProfiles := &configapi.EndpointPickerConfig{
		TypeMeta: metav1.TypeMeta{
			Kind:       "EndpointPickerConfig",
			APIVersion: "inference.networking.x-k8s.io/v1alpha1",
		},
		Plugins: []configapi.PluginSpec{
			{
				Name:       "test1",
				Type:       test1Type,
				Parameters: json.RawMessage("{\"threshold\":10}"),
			},
		},
	}

	tests := []testStruct{
		{
			name:       "success",
			configText: successConfigText,
			configFile: "",
			want:       goodConfig,
			wantErr:    false,
		},
		{
			name:       "successNoProfiles",
			configText: successNoProfilesText,
			configFile: "",
			want:       goodConfigNoProfiles,
			wantErr:    false,
		},
		{
			name:       "errorBadYaml",
			configText: errorBadYamlText,
			configFile: "",
			wantErr:    true,
		},
		{
			name:       "successFromFile",
			configText: "",
			configFile: "../../../../test/testdata/configloader_1_test.yaml",
			want:       goodConfig,
			wantErr:    false,
		},
	}

	for _, test := range tests {
		configBytes := []byte(test.configText)
		if test.configFile != "" {
			configBytes, _ = os.ReadFile(test.configFile)
		}

		got, err := loadRawConfig(configBytes)
		checker(t, "loadRawConfig", test, got, err)
	}
}

func TestLoadRawConfigurationWithDefaults(t *testing.T) {
	registerTestPlugins()

	goodConfig := &configapi.EndpointPickerConfig{
		TypeMeta: metav1.TypeMeta{
			Kind:       "EndpointPickerConfig",
			APIVersion: "inference.networking.x-k8s.io/v1alpha1",
		},
		Plugins: []configapi.PluginSpec{
			{
				Name:       "test1",
				Type:       test1Type,
				Parameters: json.RawMessage("{\"threshold\":10}"),
			},
			{
				Name: "profileHandler",
				Type: "test-profile-handler",
			},
			{
				Name:       test2Type,
				Type:       test2Type,
				Parameters: json.RawMessage("{\"hashBlockSize\":32}"),
			},
			{
				Name: "testPicker",
				Type: testPickerType,
			},
		},
		SchedulingProfiles: []configapi.SchedulingProfile{
			{
				Name: "default",
				Plugins: []configapi.SchedulingPlugin{
					{
						PluginRef: "test1",
					},
					{
						PluginRef: "test-two",
						Weight:    ptr.To(50),
					},
					{
						PluginRef: "testPicker",
					},
				},
			},
		},
	}

	goodConfigNoProfiles := &configapi.EndpointPickerConfig{
		TypeMeta: metav1.TypeMeta{
			Kind:       "EndpointPickerConfig",
			APIVersion: "inference.networking.x-k8s.io/v1alpha1",
		},
		Plugins: []configapi.PluginSpec{
			{
				Name:       "test1",
				Type:       test1Type,
				Parameters: json.RawMessage("{\"threshold\":10}"),
			},
			{
				Name: profile.SingleProfileHandlerType,
				Type: profile.SingleProfileHandlerType,
			},
			{
				Name: picker.MaxScorePickerType,
				Type: picker.MaxScorePickerType,
			},
		},
		SchedulingProfiles: []configapi.SchedulingProfile{
			{
				Name: "default",
				Plugins: []configapi.SchedulingPlugin{
					{
						PluginRef: "test1",
					},
					{
						PluginRef: "max-score-picker",
					},
				},
			},
		},
	}

	tests := []testStruct{
		{
			name:       "success",
			configText: successConfigText,
			configFile: "",
			want:       goodConfig,
			wantErr:    false,
		},
		{
			name:       "successNoProfiles",
			configText: successNoProfilesText,
			configFile: "",
			want:       goodConfigNoProfiles,
			wantErr:    false,
		},
		{
			name:       "errorBadPluginReferenceText",
			configText: errorBadPluginReferenceText,
			configFile: "",
			wantErr:    true,
		},
		{
			name:       "errorBadPluginReferencePluginText",
			configText: errorBadPluginReferencePluginText,
			configFile: "",
			wantErr:    true,
		},
		{
			name:       "errorNoProfileName",
			configText: errorNoProfileNameText,
			configFile: "",
			wantErr:    true,
		},
		{
			name:       "errorBadProfilePlugin",
			configText: errorBadProfilePluginText,
			configFile: "",
			wantErr:    true,
		},
		{
			name:       "errorBadProfilePluginRef",
			configText: errorBadProfilePluginRefText,
			configFile: "",
			wantErr:    true,
		},
		{
			name:       "errorDuplicatePlugin",
			configText: errorDuplicatePluginText,
			configFile: "",
			wantErr:    true,
		},
		{
			name:       "errorDuplicateProfile",
			configText: errorDuplicateProfileText,
			configFile: "",
			wantErr:    true,
		},
		{
			name:       "successFromFile",
			configText: "",
			configFile: "../../../../test/testdata/configloader_1_test.yaml",
			want:       goodConfig,
			wantErr:    false,
		},
	}

	for _, test := range tests {
		handle := utils.NewTestHandle(context.Background())

		configBytes := []byte(test.configText)
		if test.configFile != "" {
			configBytes, _ = os.ReadFile(test.configFile)
		}

		got, err := loadRawConfig(configBytes)
		if err == nil {
			setDefaultsPhaseOne(got)

			err = instantiatePlugins(got.Plugins, handle)
			if err == nil {
				setDefaultsPhaseTwo(got, handle)

				err = validateSchedulingProfiles(got)
			}
		}
		checker(t, "tested function", test, got, err)
	}
}

func checker(t *testing.T, function string, test testStruct, got *configapi.EndpointPickerConfig, err error) {
	checkError(t, function, test, err)
	if err == nil && !test.wantErr {
		if diff := cmp.Diff(test.want, got); diff != "" {
			t.Errorf("In test %s %s returned unexpected response, diff(-want, +got): %v", test.name, function, diff)
		}
	}
}

func checkError(t *testing.T, function string, test testStruct, err error) {
	if err != nil {
		if !test.wantErr {
			t.Fatalf("In test '%s' %s returned unexpected error: %v, want %v", test.name, function, err, test.wantErr)
		}
		t.Logf("error was %s", err)
	} else if test.wantErr {
		t.Fatalf("In test %s %s did not return an expected error", test.name, function)
	}
}

func TestInstantiatePlugins(t *testing.T) {
	handle := utils.NewTestHandle(context.Background())
	_, err := LoadConfig([]byte(successConfigText), handle, logging.NewTestLogger())
	if err != nil {
		t.Fatalf("LoadConfig returned unexpected error - %v", err)
	}
	if len(handle.GetAllPlugins()) == 0 {
		t.Fatalf("unexpected empty set of loaded plugins")
	}
	if t1 := handle.Plugin("test1"); t1 == nil {
		t.Fatalf("loaded plugins did not contain test1")
	} else if _, ok := t1.(*test1); !ok {
		t.Fatalf("loaded plugins returned test1 has the wrong type %#v", t1)
	}

	handle = utils.NewTestHandle(context.Background())
	_, err = LoadConfig([]byte(errorBadPluginReferenceParametersText), handle, logging.NewTestLogger())
	if err == nil {
		t.Fatalf("LoadConfig did not return error as expected ")
	}
}

func TestLoadConfig(t *testing.T) {

	tests := []struct {
		name       string
		configText string
		want       *config.Config
		wantErr    bool
	}{
		{
			name:       "schedulerSuccess",
			configText: successSchedulerConfigText,
			wantErr:    false,
		},
		{
			name:       "successWithNoWeight",
			configText: successWithNoWeightText,
			wantErr:    false,
		},
		{
			name:       "successWithNoProfileHandlers",
			configText: successWithNoProfileHandlersText,
			wantErr:    false,
		},
		{
			name:       "errorBadYaml",
			configText: errorBadYamlText,
			wantErr:    true,
		},
		{
			name:       "errorBadPluginJson",
			configText: errorBadPluginJsonText,
			wantErr:    true,
		},
		{
			name:       "errorNoProfileName",
			configText: errorNoProfileNameText,
			wantErr:    true,
		},
		{
			name:       "errorTwoPickers",
			configText: errorTwoPickersText,
			wantErr:    true,
		},
		{
			name:       "errorTwoProfileHandlers",
			configText: errorTwoProfileHandlersText,
			wantErr:    true,
		},
		{
			name:       "errorNoProfileHandlers",
			configText: errorNoProfileHandlersText,
			wantErr:    true,
		},
	}

	registerNeededPlgugins()

	logger := logging.NewTestLogger()
	for _, test := range tests {
		handle := utils.NewTestHandle(context.Background())
		_, err := LoadConfig([]byte(test.configText), handle, logger)
		if err != nil {
			if !test.wantErr {
				t.Errorf("LoadConfig returned an unexpected error. error %v", err)
			}
			t.Logf("error was %s", err)
		} else if test.wantErr {
			t.Errorf("LoadConfig did not return an expected error (%s)", test.name)
		}
	}
}

func registerNeededPlgugins() {
	plugins.Register(prefix.PrefixCachePluginType, prefix.PrefixCachePluginFactory)
	plugins.Register(picker.MaxScorePickerType, picker.MaxScorePickerFactory)
	plugins.Register(picker.RandomPickerType, picker.RandomPickerFactory)
	plugins.Register(profile.SingleProfileHandlerType, profile.SingleProfileHandlerFactory)
}

// The following multi-line string constants, cause false positive lint errors (dupword)

// valid configuration
//
//nolint:dupword
const successConfigText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-one
  parameters:
    threshold: 10
- name: profileHandler
  type: test-profile-handler
- type: test-two
  parameters:
    hashBlockSize: 32
- name: testPicker
  type: test-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
  - pluginRef: test-two
    weight: 50
  - pluginRef: testPicker
`

// success with missing scheduling profiles
//
//nolint:dupword
const successNoProfilesText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-one
  parameters:
    threshold: 10
`

// YAML does not follow expected structure of config
//
//nolint:dupword
const errorBadYamlText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- testing 1 2 3
`

// missing required Plugin type
//
//nolint:dupword
const errorBadPluginReferenceText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- parameters:
    a: 1234
`

// plugin type does not exist
//
//nolint:dupword
const errorBadPluginReferencePluginText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: testx
  type: test-x
- name: profileHandler
  type: test-profile-handler
`

// missing required scheduling profile name
//
//nolint:dupword
const errorNoProfileNameText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-one
  parameters:
    threshold: 10
- name: profileHandler
  type: test-profile-handler
schedulingProfiles:
- plugins:
  - pluginRef: test1
`

// missing required plugin reference name, only weight is provided
//
//nolint:dupword
const errorBadProfilePluginText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - weight: 10
`

// reference a non-existent plugin
//
//nolint:dupword
const errorBadProfilePluginRefText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: plover
`

// invalid parameters (string provided where int is expected)
//
//nolint:dupword
const errorBadPluginReferenceParametersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-one
  parameters:
    threshold: asdf
- name: profileHandler
  type: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
`

// duplicate names in plugin list
//
//nolint:dupword
const errorDuplicatePluginText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-one
  parameters:
    threshold: 10
- name: test1
  type: test-one
  parameters:
    threshold: 20
- name: profileHandler
  type: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
`

// duplicate scheduling profile name
//
//nolint:dupword
const errorDuplicateProfileText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-one
  parameters:
    threshold: 10
- name: test2
  type: test-one
  parameters:
    threshold: 20
- name: profileHandler
  type: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
- name: default
  plugins:
  - pluginRef: test2
`

// compile-time type validation
var _ framework.Filter = &test1{}

type test1 struct {
	typedName plugins.TypedName
	Threshold int `json:"threshold"`
}

func newTest1() *test1 {
	return &test1{
		typedName: plugins.TypedName{Type: test1Type, Name: "test-1"},
	}
}

func (f *test1) TypedName() plugins.TypedName {
	return f.typedName
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *test1) Filter(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, pods []types.Pod) []types.Pod {
	return pods
}

// compile-time type validation
var _ framework.Scorer = &test2{}
var _ framework.PostCycle = &test2{}

type test2 struct {
	typedName plugins.TypedName
}

func newTest2() *test2 {
	return &test2{
		typedName: plugins.TypedName{Type: test2Type, Name: "test-2"},
	}
}

func (m *test2) TypedName() plugins.TypedName {
	return m.typedName
}

func (m *test2) Score(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, _ []types.Pod) map[types.Pod]float64 {
	return map[types.Pod]float64{}
}

func (m *test2) PostCycle(_ context.Context, _ *types.CycleState, _ *types.ProfileRunResult) {}

// compile-time type validation
var _ framework.Picker = &testPicker{}

type testPicker struct {
	typedName plugins.TypedName
}

func newTestPicker() *testPicker {
	return &testPicker{
		typedName: plugins.TypedName{Type: testPickerType, Name: "test-picker"},
	}
}

func (p *testPicker) TypedName() plugins.TypedName {
	return p.typedName
}

func (p *testPicker) Pick(_ context.Context, _ *types.CycleState, _ []*types.ScoredPod) *types.ProfileRunResult {
	return nil
}

// compile-time type validation
var _ framework.ProfileHandler = &testProfileHandler{}

type testProfileHandler struct {
	typedName plugins.TypedName
}

func newTestProfileHandler() *testProfileHandler {
	return &testProfileHandler{
		typedName: plugins.TypedName{Type: testProfileHandlerType, Name: "test-profile-handler"},
	}
}

func (p *testProfileHandler) TypedName() plugins.TypedName {
	return p.typedName
}

func (p *testProfileHandler) Pick(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, _ map[string]*framework.SchedulerProfile, _ map[string]*types.ProfileRunResult) map[string]*framework.SchedulerProfile {
	return nil
}

func (p *testProfileHandler) ProcessResults(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, _ map[string]*types.ProfileRunResult) (*types.SchedulingResult, error) {
	return nil, nil
}

func registerTestPlugins() {
	plugins.Register(test1Type,
		func(_ string, parameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
			result := newTest1()
			err := json.Unmarshal(parameters, result)
			return result, err
		},
	)

	plugins.Register(test2Type,
		func(_ string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
			return newTest2(), nil
		},
	)

	plugins.Register(testPickerType,
		func(_ string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
			return newTestPicker(), nil
		},
	)

	plugins.Register(testProfileHandlerType,
		func(_ string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
			return newTestProfileHandler(), nil
		},
	)
}

// valid configuration
//
//nolint:dupword
const successSchedulerConfigText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: prefixCacheScorer
  type: prefix-cache-scorer
  parameters:
    hashBlockSize: 32
- name: maxScorePicker
  type: max-score-picker
- name: profileHandler
  type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefixCacheScorer
    weight: 50
  - pluginRef: maxScorePicker
`

// valid configuration, with default weight for scorer
//
//nolint:dupword
const successWithNoWeightText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: single-profile-handler
- name: prefixCacheScorer
  type: prefix-cache-scorer
  parameters:
    hashBlockSize: 32
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefixCacheScorer
`

// valid configuration using default profile handler
//
//nolint:dupword
const successWithNoProfileHandlersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: maxScore
  type: max-score-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
`

// invalid parameter configuration for plugin (string passed, in expected)
//
//nolint:dupword
const errorBadPluginJsonText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: single-profile-handler
- name: prefixCacheScorer
  type: prefix-cache-scorer
  parameters:
    hashBlockSize: asdf
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefixCacheScorer
    weight: 50
`

// multiple pickers in scheduling profile
//
//nolint:dupword
const errorTwoPickersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: single-profile-handler
- name: maxScore
  type: max-score-picker
- name: random
  type: random-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
  - pluginRef: random
`

// multiple profile handlers when only one is allowed
//
//nolint:dupword
const errorTwoProfileHandlersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: single-profile-handler
- name: secondProfileHandler
  type: single-profile-handler
- name: maxScore
  type: max-score-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
`

// missing required profile handler
//
//nolint:dupword
const errorNoProfileHandlersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: maxScore
  type: max-score-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
- name: prof2
  plugins:
  - pluginRef: maxScore
`
