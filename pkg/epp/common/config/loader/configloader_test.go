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
	}

	for _, test := range tests {
		configBytes := []byte(test.configText)
		if test.configFile != "" {
			configBytes, _ = os.ReadFile(test.configFile)
		}

		got, err := LoadConfig(configBytes, utils.NewTestHandle(context.Background()))
		if err != nil {
			if !test.wantErr {
				t.Fatalf("In test '%s' LoadConfig returned unexpected error: %v, want %v", test.name, err, test.wantErr)
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

func TestInstantiatePlugins(t *testing.T) {
	handle := utils.NewTestHandle(context.Background())
	_, err := LoadConfig([]byte(successConfigText), handle)
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
	_, err = LoadConfig([]byte(errorBadPluginReferenceParametersText), handle)
	if err == nil {
		t.Fatalf("LoadConfig did not return error as expected ")
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
		handle := utils.NewTestHandle(context.Background())
		config, err := LoadConfig([]byte(test.configText), handle)
		if err != nil {
			if test.wantErr {
				continue
			}
			t.Fatalf("LoadConfig returned unexpected error: %v", err)
		}

		_, err = LoadSchedulerConfig(config.SchedulingProfiles, handle)
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

// missing required profile handler
//
//nolint:dupword
const errorNoProfileHandlerText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-one
  parameters:
    threshold: 10
schedulingProfiles:
- name: default
`

// missing scheduling profiles
//
//nolint:dupword
const errorNoProfilesText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-one
  parameters:
    threshold: 10
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

// missing plugins in scheduling profile
//
//nolint:dupword
const errorNoProfilePluginsText = `
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
- name: default
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
  pluginName: test-one
  type:
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
- name: lowQueue
  type: low-queue
  parameters:
    threshold: 10
- name: prefixCache
  type: prefix-cache
  parameters:
    hashBlockSize: 32
- name: maxScore
  type: max-score
- name: profileHandler
  type: single-profile
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: lowQueue
  - pluginRef: prefixCache
    weight: 50
  - pluginRef: maxScore
`

// invalid parameter configuration for plugin (string passed, in expected)
//
//nolint:dupword
const errorBadPluginJsonText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name:profileHandler
  type: single-profile
- name: prefixCache
  type: prefix-cache
  parameters:
    hashBlockSize: asdf
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefixCache
    weight: 50
`

// missing weight for scorer
//
//nolint:dupword
const errorBadReferenceNoWeightText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: single-profile
- name: prefixCache
  type: prefix-cache
  parameters:
    hashBlockSize: 32
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefixCache
`

// multiple pickers in scheduling profile
//
//nolint:dupword
const errorTwoPickersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: single-profile
- name: maxScore
  type: max-score
- name: random
  type: random
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
  - pluginRef: random
`

// missing required scheduling profile
//
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

// multiple profile handlers when only one is allowed
//
//nolint:dupword
const errorTwoProfileHandlersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: single-profile
- name: secondProfileHandler
  type: single-profile
- name: maxScore
  type: max-score
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
  type: max-score
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
`
