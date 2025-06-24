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

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configapi "sigs.k8s.io/gateway-api-inference-extension/api/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

const (
	testProfileHandlerType = "test-profile-handler"
	test1Type              = "test-one"
	test2Type              = "test-two"
	testPickerType         = "test-picker"
)

func TestLoadConfiguration(t *testing.T) {
	test2Weight := 50

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
						Weight:    &test2Weight,
					},
					{
						PluginRef: "testPicker",
					},
				},
			},
		},
	}

	tests := []struct {
		name       string
		configText string
		configFile string
		want       *configapi.EndpointPickerConfig
		wantErr    bool
	}{
		{
			name:       "success",
			configText: successConfigText,
			configFile: "",
			want:       goodConfig,
			wantErr:    false,
		},
		{
			name:       "errorBadYaml",
			configText: errorBadYamlText,
			configFile: "",
			wantErr:    true,
		},
		{
			name:       "errorNoProfileHandler",
			configText: errorNoProfileHandlerText,
			configFile: "",
			wantErr:    true,
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
			name:       "errorNoProfiles",
			configText: errorNoProfilesText,
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
			name:       "errorNoProfilePlugins",
			configText: errorNoProfilePluginsText,
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
			name:       "errorBadProfilePluginName",
			configText: errorBadProfilePluginNameText,
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
			configFile: "../../../../../test/testdata/configloader_1_test.yaml",
			want:       goodConfig,
			wantErr:    false,
		},
		{
			name:       "noSuchFile",
			configText: "",
			configFile: "../../../../../test/testdata/configloader_error_test.yaml",
			wantErr:    true,
		},
	}

	for _, test := range tests {
		got, err := LoadConfig([]byte(test.configText), test.configFile)
		if err != nil {
			if !test.wantErr {
				t.Fatalf("In test %s LoadConfig returned unexpected error: %v, want %v", test.name, err, test.wantErr)
			}
			t.Logf("error was %s", err)
		} else {
			if test.wantErr {
				t.Fatalf("In test %s LoadConfig did not return an expected error", test.name)
			}
			if diff := cmp.Diff(test.want, got); diff != "" {
				t.Errorf("In test %s LoadConfig returned unexpected response, diff(-want, +got): %v", test.name, diff)
			}
		}
	}
}

func TestLoadPluginReferences(t *testing.T) {
	theConfig, err := LoadConfig([]byte(successConfigText), "")
	if err != nil {
		t.Fatalf("LoadConfig returned unexpected error: %v", err)
	}
	handle := utils.NewTestHandle()
	err = LoadPluginReferences(theConfig.Plugins, handle)
	if err != nil {
		t.Fatalf("LoadPluginReferences returned unexpected error: %v", err)
	}
	if len(handle.Plugins().GetAllPlugins()) == 0 {
		t.Fatalf("LoadPluginReferences returned an empty set of references")
	}
	if t1 := handle.Plugins().Plugin("test1"); t1 == nil {
		t.Fatalf("LoadPluginReferences returned references did not contain test1")
	} else if _, ok := t1.(*test1); !ok {
		t.Fatalf("LoadPluginReferences returned references value for test1 has the wrong type %#v", t1)
	}

	theConfig, err = LoadConfig([]byte(errorBadPluginReferenceParametersText), "")
	if err != nil {
		t.Fatalf("LoadConfig returned unexpected error: %v", err)
	}
	err = LoadPluginReferences(theConfig.Plugins, utils.NewTestHandle())
	if err == nil {
		t.Fatalf("LoadPluginReferences did not return the expected error")
	}
}

func TestInstantiatePlugin(t *testing.T) {
	plugSpec := configapi.PluginSpec{Type: "plover"}
	_, err := instantiatePlugin(plugSpec, utils.NewTestHandle())
	if err == nil {
		t.Fatalf("InstantiatePlugin did not return the expected error")
	}
}

func TestLoadSchedulerConfig(t *testing.T) {
	tests := []struct {
		name       string
		configText string
		wantErr    bool
	}{
		{
			name:       "schedulerSuccess",
			configText: successSchedulerConfigText,
			wantErr:    false,
		},
		{
			name:       "errorBadPluginJson",
			configText: errorBadPluginJsonText,
			wantErr:    true,
		},
		{
			name:       "errorBadReferenceNoWeight",
			configText: errorBadReferenceNoWeightText,
			wantErr:    true,
		},
		{
			name:       "errorPluginReferenceJson",
			configText: errorPluginReferenceJsonText,
			wantErr:    true,
		},
		{
			name:       "errorTwoPickers",
			configText: errorTwoPickersText,
			wantErr:    true,
		},
		{
			name:       "errorConfig",
			configText: errorConfigText,
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

	for _, test := range tests {
		theConfig, err := LoadConfig([]byte(test.configText), "")
		if err != nil {
			if test.wantErr {
				continue
			}
			t.Fatalf("LoadConfig returned unexpected error: %v", err)
		}
		handle := utils.NewTestHandle()
		err = LoadPluginReferences(theConfig.Plugins, handle)
		if err != nil {
			if test.wantErr {
				continue
			}
			t.Fatalf("LoadPluginReferences returned unexpected error: %v", err)
		}

		_, err = LoadSchedulerConfig(theConfig.SchedulingProfiles, handle)
		if err != nil {
			if !test.wantErr {
				t.Errorf("LoadSchedulerConfig returned an unexpected error. error %v", err)
			}
		} else if test.wantErr {
			t.Errorf("LoadSchedulerConfig did not return an expected error (%s)", test.name)
		}
	}
}

func registerNeededPlgugins() {
	plugins.Register(filter.LowQueueFilterType, filter.LowQueueFilterFactory)
	plugins.Register(prefix.PrefixCachePluginType, prefix.PrefixCachePluginFactory)
	plugins.Register(picker.MaxScorePickerType, picker.MaxScorePickerFactory)
	plugins.Register(picker.RandomPickerType, picker.RandomPickerFactory)
	plugins.Register(profile.SingleProfileHandlerType, profile.SingleProfileHandlerFactory)
}

// The following multi-line string constants, cause false positive lint errors (dupword)

//nolint:dupword
const successConfigText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  pluginName: test-one
  parameters:
    threshold: 10
- name: profileHandler
  pluginName: test-profile-handler
- pluginName: test-two
  parameters:
    hashBlockSize: 32
- name: testPicker
  pluginName: test-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
  - pluginRef: test-two
    weight: 50
  - pluginRef: testPicker
`

//nolint:dupword
const errorBadYamlText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- testing 1 2 3
`

//nolint:dupword
const errorBadPluginReferenceText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- parameters:
    a: 1234
`

//nolint:dupword
const errorBadPluginReferencePluginText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: testx
  pluginName: test-x
- name: profileHandler
  pluginName: test-profile-handler
`

//nolint:dupword
const errorNoProfileHandlerText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  pluginName: test-one
  parameters:
    threshold: 10
schedulingProfiles:
- name: default
`

//nolint:dupword
const errorNoProfilesText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  pluginName: test-one
  parameters:
    threshold: 10
- name: profileHandler
  pluginName: test-profile-handler
`

//nolint:dupword
const errorNoProfileNameText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  pluginName: test-one
  parameters:
    threshold: 10
- name: profileHandler
  pluginName: test-profile-handler
schedulingProfiles:
- plugins:
  - pluginRef: test1
`

//nolint:dupword
const errorNoProfilePluginsText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  pluginName: test-one
  parameters:
    threshold: 10
- name: profileHandler
  pluginName: test-profile-handler
schedulingProfiles:
- name: default
`

//nolint:dupword
const errorBadProfilePluginText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  pluginName: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - weight: 10
`

//nolint:dupword
const errorBadProfilePluginRefText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  pluginName: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: plover
`

//nolint:dupword
const errorBadProfilePluginNameText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- pluginName: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: plover
`

//nolint:dupword
const errorBadPluginReferenceParametersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  pluginName: test-one
  parameters:
    threshold: asdf
- name: profileHandler
  pluginName: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
`

//nolint:dupword
const errorDuplicatePluginText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  pluginName: test-one
  parameters:
    threshold: 10
- name: test1
  pluginName: test-one
  parameters:
    threshold: 20
- name: profileHandler
  pluginName: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
`

//nolint:dupword
const errorDuplicateProfileText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  pluginName: test-one
  parameters:
    threshold: 10
- name: test2
  pluginName: test-one
  parameters:
    threshold: 20
- name: profileHandler
  pluginName: test-profile-handler
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
	Threshold int `json:"threshold"`
}

func (f *test1) Type() string {
	return test1Type
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *test1) Filter(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, pods []types.Pod) []types.Pod {
	return pods
}

// compile-time type validation
var _ framework.Scorer = &test2{}
var _ framework.PostCycle = &test2{}

type test2 struct{}

func (f *test2) Type() string {
	return test2Type
}

func (m *test2) Score(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, _ []types.Pod) map[types.Pod]float64 {
	return map[types.Pod]float64{}
}

func (m *test2) PostCycle(_ context.Context, _ *types.CycleState, _ *types.ProfileRunResult) {}

// compile-time type validation
var _ framework.Picker = &testPicker{}

type testPicker struct{}

func (p *testPicker) Type() string {
	return testPickerType
}

func (p *testPicker) Pick(_ context.Context, _ *types.CycleState, _ []*types.ScoredPod) *types.ProfileRunResult {
	return nil
}

// compile-time type validation
var _ framework.ProfileHandler = &testProfileHandler{}

type testProfileHandler struct{}

func (p *testProfileHandler) Type() string {
	return testProfileHandlerType
}

func (p *testProfileHandler) Pick(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, _ map[string]*framework.SchedulerProfile, _ map[string]*types.ProfileRunResult) map[string]*framework.SchedulerProfile {
	return nil
}

func (p *testProfileHandler) ProcessResults(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, _ map[string]*types.ProfileRunResult) (*types.SchedulingResult, error) {
	return nil, nil
}

func registerTestPlugins() {
	plugins.Register(test1Type,
		func(name string, parameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
			result := test1{}
			err := json.Unmarshal(parameters, &result)
			return &result, err
		},
	)

	plugins.Register(test2Type,
		func(name string, parameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
			return &test2{}, nil
		},
	)

	plugins.Register(testPickerType,
		func(name string, parameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
			return &testPicker{}, nil
		},
	)

	plugins.Register(testProfileHandlerType,
		func(name string, parameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
			return &testProfileHandler{}, nil
		},
	)
}

//nolint:dupword
const successSchedulerConfigText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: lowQueue
  pluginName: low-queue
  parameters:
    threshold: 10
- name: prefixCache
  pluginName: prefix-cache
  parameters:
    hashBlockSize: 32
- name: maxScore
  pluginName: max-score
- name: profileHandler
  pluginName: single-profile
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: lowQueue
  - pluginRef: prefixCache
    weight: 50
  - pluginRef: maxScore
`

//nolint:dupword
const errorBadPluginJsonText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name:profileHandler
  pluginName: single-profile
- name: prefixCache
  pluginName: prefix-cache
  parameters:
    hashBlockSize: asdf
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefixCache
    weight: 50
`

//nolint:dupword
const errorBadReferenceNoWeightText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  pluginName: single-profile
- name: prefixCache
  pluginName: prefix-cache
  parameters:
    hashBlockSize: 32
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefixCache
`

//nolint:dupword
const errorPluginReferenceJsonText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: lowQueue
  pluginName: low-queue
  parameters:
    threshold: qwer
- name: profileHandler
  pluginName: single-profile
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: lowQueue
`

//nolint:dupword
const errorTwoPickersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  pluginName: single-profile
- name: maxScore
  pluginName: max-score
- name: random
  pluginName: random
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
  - pluginRef: random
`

//nolint:dupword
const errorConfigText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: lowQueue
  pluginName: low-queue
  parameters:
    threshold: 10
`

//nolint:dupword
const errorTwoProfileHandlersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  pluginName: single-profile
- name: secondProfileHandler
  pluginName: single-profile
- name: maxScore
  pluginName: max-score
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
`

//nolint:dupword
const errorNoProfileHandlersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: maxScore
  pluginName: max-score
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
`
