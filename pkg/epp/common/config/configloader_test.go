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
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	testProfilePicker = "test-profile-picker"
	test1Name         = "test-one"
	test2Name         = "test-two"
	testPickerName    = "test-picker"
)

func TestLoadConfiguration(t *testing.T) {
	test1 := "test1"
	test2Weight := 50

	registerTestPlugins()

	goodConfig := &Config{
		PluginDefinitions: []ConfigPluginDefinition{
			{
				Name: "test1",
				BaseConfigPluginDefinition: BaseConfigPluginDefinition{
					PluginName: test1Name,
					Parameters: json.RawMessage("{\"threshold\":10}"),
				},
			},
		},
		ProfilePicker: BaseConfigProfilePlugin{
			Plugin: &BaseConfigPluginDefinition{
				PluginName: testProfilePicker,
			},
		},
		SchedulerProfiles: []ConfigSchedulerProfile{
			{
				Name: "default",
				Plugins: []ConfigProfilePlugin{
					{
						BaseConfigProfilePlugin: BaseConfigProfilePlugin{
							Reference: &test1,
						},
					},
					{
						BaseConfigProfilePlugin: BaseConfigProfilePlugin{
							Plugin: &BaseConfigPluginDefinition{
								PluginName: test2Name,
								Parameters: json.RawMessage("{\"hash-block-size\":32}"),
							},
						},
						Weight: &test2Weight,
					},
					{
						BaseConfigProfilePlugin: BaseConfigProfilePlugin{
							Plugin: &BaseConfigPluginDefinition{
								PluginName: testPickerName,
							},
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name       string
		configText string
		configFile string
		want       *Config
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
			name:       "errorNoProfilePicker",
			configText: errorNoProfilePickerText,
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

	plugReference := BaseConfigPluginDefinition{PluginName: "plover"}
	_, err := InstantiatePlugin(plugReference, log)
	if err == nil {
		t.Fatalf("InstantiatePlugin did not return the expected error")
	}
}

// The following multi-line string constants, cause false positive lint errors (dupword)

//nolint:dupword
const successConfigText = `
plugin_definitions:
- name: test1
  plugin_name: test-one
  parameters:
    threshold: 10
profile_picker:
  plugin: 
    plugin_name: test-profile-picker
scheduler_profiles:
- name: default
  plugins:
  - reference: test1
  - plugin:
      plugin_name: test-two
      parameters:
        hash-block-size: 32
    weight: 50
  - plugin:
      plugin_name: test-picker
`

//nolint:dupword
const errorBadYamlText = `
plugin_definitions:
- testing 1 2 3
`

//nolint:dupword
const errorBadPluginReferenceText = `
plugin_definitions:
- test: 1234
`

//nolint:dupword
const errorBadPluginReferencePluginText = `
plugin_definitions:
- name: testx
  plugin_name: test-x
profile_picker:
  plugin: 
    plugin_name: test-profile-picker
`

//nolint:dupword
const errorNoProfilePickerText = `
plugin_definitions:
- name: test1
  plugin_name: test-one
  parameters:
    threshold: 10
scheduler_profiles:
- name: default
`

//nolint:dupword
const errorNoProfilesText = `
plugin_definitions:
- name: test1
  plugin_name: test-one
  parameters:
    threshold: 10
profile_picker:
  plugin: 
    plugin_name: test-profile-picker
`

//nolint:dupword
const errorNoProfileNameText = `
plugin_definitions:
- name: test1
  plugin_name: test-one
  parameters:
    threshold: 10
profile_picker:
  plugin: 
    plugin_name: test-profile-picker
scheduler_profiles:
- test: x
`

//nolint:dupword
const errorNoProfilePluginsText = `
plugin_definitions:
- name: test1
  plugin_name: test-one
  parameters:
    threshold: 10
profile_picker:
  plugin: 
    plugin_name: test-profile-picker
scheduler_profiles:
- name: default
`

//nolint:dupword
const errorBadProfilePluginText = `
profile_picker:
  plugin: 
    plugin_name: test-profile-picker
scheduler_profiles:
- name: default
  plugins:
  - name: test
`

//nolint:dupword
const errorBadProfilePluginRefText = `
profile_picker:
  plugin: 
    plugin_name: test-profile-picker
scheduler_profiles:
- name: default
  plugins:
  - reference: plover
`

//nolint:dupword
const errorBadProfilePluginNameText = `
profile_picker:
  plugin: 
    plugin_name: test-profile-picker
scheduler_profiles:
- name: default
  plugins:
  - plugin:
      plugin_name: plover
`

//nolint:dupword
const errorBadPluginReferenceParametersText = `
plugin_definitions:
- name: test1
  plugin_name: test-one
  parameters:
    threshold: asdf
profile_picker:
  plugin: 
    plugin_name: test-profile-picker
scheduler_profiles:
- name: default
  plugins:
  - reference: test1
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
func (f *test1) Filter(ctx *types.SchedulingContext, pods []types.Pod) []types.Pod {
	return pods
}

// compile-time type validation
var _ framework.PreCycle = &test2{}
var _ framework.Scorer = &test2{}
var _ framework.PostCycle = &test2{}

type test2 struct{}

func (f *test2) Name() string {
	return test2Name
}

func (m *test2) PreCycle(ctx *types.SchedulingContext) {}

func (m *test2) Score(ctx *types.SchedulingContext, pods []types.Pod) map[types.Pod]float64 {
	return map[types.Pod]float64{}
}

func (m *test2) PostCycle(ctx *types.SchedulingContext, res *types.Result) {}

// compile-time type validation
var _ framework.Picker = &testPicker{}

type testPicker struct{}

func (p *testPicker) Name() string {
	return testPickerName
}

func (p *testPicker) Pick(ctx *types.SchedulingContext, scoredPods []*types.ScoredPod) *types.Result {
	return nil
}

func registerTestPlugins() {
	framework.Register(test1Name,
		func(parameters json.RawMessage) (framework.Plugin, error) {
			result := test1{}
			err := json.Unmarshal(parameters, &result)
			return &result, err
		},
	)

	framework.Register(test2Name,
		func(parameters json.RawMessage) (framework.Plugin, error) {
			return &test2{}, nil
		},
	)

	framework.Register(testPickerName,
		func(parameters json.RawMessage) (framework.Plugin, error) {
			return &testPicker{}, nil
		},
	)
}
