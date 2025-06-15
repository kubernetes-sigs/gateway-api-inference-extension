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

package config

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configapi "sigs.k8s.io/gateway-api-inference-extension/api/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	testProfileHandlerName = "test-profile-handler"
	test1Name              = "test-one"
	test2Name              = "test-two"
	testPickerName         = "test-picker"
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
				PluginName: test1Name,
				Parameters: json.RawMessage("{\"threshold\":10}"),
			},
			{
				Name:       "profileHandler",
				PluginName: "test-profile-handler",
			},
			{
				Name:       test2Name,
				PluginName: test2Name,
				Parameters: json.RawMessage("{\"hashBlockSize\":32}"),
			},
			{
				Name:       "testPicker",
				PluginName: testPickerName,
			},
		},
		SchedulingProfiles: []configapi.SchedulingProfile{
			{
				Name: "default",
				Plugins: []configapi.SchedulingProfilePlugin{
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
			configFile: "../../../../test/testdata/configloader_1_test.yaml",
			want:       goodConfig,
			wantErr:    false,
		},
		{
			name:       "noSuchFile",
			configText: "",
			configFile: "../../../../test/testdata/configloader_error_test.yaml",
			wantErr:    true,
		},
	}

	log := logutil.NewTestLogger()
	for _, test := range tests {
		got, err := LoadConfig([]byte(test.configText), test.configFile, log)
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
	log := logutil.NewTestLogger()

	theConfig, err := LoadConfig([]byte(successConfigText), "", log)
	if err != nil {
		t.Fatalf("LoadConfig returned unexpected error: %v", err)
	}
	references, err := LoadPluginReferences(theConfig, log)
	if err != nil {
		t.Fatalf("LoadPluginReferences returned unexpected error: %v", err)
	}
	if len(references) == 0 {
		t.Fatalf("LoadPluginReferences returned an empty set of references")
	}
	if t1, ok := references["test1"]; !ok {
		t.Fatalf("LoadPluginReferences returned references did not contain test1")
	} else if _, ok := t1.(*test1); !ok {
		t.Fatalf("LoadPluginReferences returned references value for test1 has the wrong type %#v", t1)
	}

	theConfig, err = LoadConfig([]byte(errorBadPluginReferenceParametersText), "", log)
	if err != nil {
		t.Fatalf("LoadConfig returned unexpected error: %v", err)
	}
	_, err = LoadPluginReferences(theConfig, log)
	if err == nil {
		t.Fatalf("LoadPluginReferences did not return the expected error")
	}
}

func TestInstantiatePlugin(t *testing.T) {
	log := logutil.NewTestLogger()

	plugSpec := configapi.PluginSpec{PluginName: "plover"}
	_, err := InstantiatePlugin(plugSpec, log)
	if err == nil {
		t.Fatalf("InstantiatePlugin did not return the expected error")
	}
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

func (f *test1) Name() string {
	return test1Name
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *test1) Filter(ctx context.Context, request *types.LLMRequest, cycleState *types.CycleState, pods []types.Pod) []types.Pod {
	return pods
}

// compile-time type validation
var _ framework.Scorer = &test2{}
var _ framework.PostCycle = &test2{}

type test2 struct{}

func (f *test2) Name() string {
	return test2Name
}

func (m *test2) Score(ctx context.Context, request *types.LLMRequest, cycleState *types.CycleState, pods []types.Pod) map[types.Pod]float64 {
	return map[types.Pod]float64{}
}

func (m *test2) PostCycle(ctx context.Context, cycleState *types.CycleState, res *types.ProfileRunResult) {
}

// compile-time type validation
var _ framework.Picker = &testPicker{}

type testPicker struct{}

func (p *testPicker) Name() string {
	return testPickerName
}

func (p *testPicker) Pick(ctx context.Context, cycleState *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult {
	return nil
}

// compile-time type validation
var _ framework.ProfileHandler = &testProfileHandler{}

type testProfileHandler struct{}

func (p *testProfileHandler) Name() string {
	return testProfileHandlerName
}

func (p *testProfileHandler) Pick(ctx context.Context, request *types.LLMRequest, profiles map[string]*framework.SchedulerProfile, executionResults map[string]*types.ProfileRunResult) map[string]*framework.SchedulerProfile {
	return nil
}

func (p *testProfileHandler) ProcessResults(ctx context.Context, request *types.LLMRequest, profileResults map[string]*types.ProfileRunResult) *types.SchedulingResult {
	return nil
}

func registerTestPlugins() {
	plugins.Register(test1Name,
		func(parameters json.RawMessage) (plugins.Plugin, error) {
			result := test1{}
			err := json.Unmarshal(parameters, &result)
			return &result, err
		},
	)

	plugins.Register(test2Name,
		func(parameters json.RawMessage) (plugins.Plugin, error) {
			return &test2{}, nil
		},
	)

	plugins.Register(testPickerName,
		func(parameters json.RawMessage) (plugins.Plugin, error) {
			return &testPicker{}, nil
		},
	)

	plugins.Register(testProfileHandlerName,
		func(parameters json.RawMessage) (plugins.Plugin, error) {
			return &testProfileHandler{}, nil
		},
	)
}
