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

package scheduling

import (
	"fmt"
	"testing"

	commonconfig "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/common/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	profilepicker "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile-picker"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// Cause the various init() functions to registers the plugins
var _ = profilepicker.AllProfilesPicker{}
var _ = filter.LeastQueueFilter{}
var _ = prefix.Plugin{}
var _ = picker.MaxScorePicker{}

func TestLoadSchedulerConfig(t *testing.T) {
	log := logutil.NewTestLogger()

	tests := []struct {
		name       string
		configText string
		wantErr    bool
	}{
		{
			name:       "success",
			configText: successConfigText,
			wantErr:    false,
		},
		{
			name:       "errorBadPluginJson",
			configText: errorBadPluginJsonText,
			wantErr:    true,
		},
		{
			name:       "errorBadPluginNoWeight",
			configText: errorBadPluginNoWeightText,
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
	}

	for _, test := range tests {
		fmt.Printf("\n\n%s\n\n", test.name)
		theConfig, err := commonconfig.LoadConfig([]byte(test.configText), "", log)
		if err != nil {
			if test.wantErr {
				continue
			}
			t.Fatalf("LoadConfig returned unexpected error: %v", err)
		}

		_, err = LoadSchedulerConfig(theConfig, log)
		if err != nil {
			if !test.wantErr {
				t.Errorf("LoadSchedulerConfig returned an unexpected error. error %v", err)
			}
		} else if test.wantErr {
			t.Errorf("LoadSchedulerConfig did not return an expected error (%s)", test.name)
		}
	}
}

// The following multi-line string constants, cause false positive lint errors (dupword)

//nolint:dupword
const successConfigText = `
plugin_definitions:
- name: lowQueue
  plugin_name: low-queue
  parameters:
    threshold: 10
profile_picker:
  plugin:
    plugin_name: all-profiles
scheduler_profiles:
- name: default
  plugins:
  - reference: lowQueue
  - plugin:
      plugin_name: prefix-cache
      parameters:
        hash-block-size: 32
    weight: 50
  - plugin:
      plugin_name: max-score
`

//nolint:dupword
const errorBadPluginJsonText = `
profile_picker:
  plugin:
    plugin_name: all-profiles
scheduler_profiles:
- name: default
  plugins:
  - plugin:
      plugin_name: prefix-cache
      parameters:
        hash-block-size: asdf
    weight: 50
`

//nolint:dupword
const errorBadPluginNoWeightText = `
profile_picker:
  plugin:
    plugin_name: all-profiles
scheduler_profiles:
- name: default
  plugins:
  - plugin:
      plugin_name: prefix-cache
      parameters:
        hash-block-size: 32
`

//nolint:dupword
const errorBadReferenceNoWeightText = `
plugin_definitions:
- name: prefix
  plugin_name: prefix-cache
  parameters:
    hash-block-size: 32
profile_picker:
  plugin:
    plugin_name: all-profiles
scheduler_profiles:
- name: default
  plugins:
  - reference: prefix
`

//nolint:dupword
const errorPluginReferenceJsonText = `
plugin_definitions:
- name: lowQueue
  plugin_name: low-queue
  parameters:
    threshold: qwer
profile_picker:
  plugin:
    plugin_name: all-profiles
scheduler_profiles:
- name: default
  plugins:
  - reference: lowQueue
`

//nolint:dupword
const errorTwoPickersText = `
profile_picker:
  plugin:
    plugin_name: all-profiles
scheduler_profiles:
- name: default
  plugins:
  - plugin:
      plugin_name: max-score
  - plugin:
      plugin_name: random
`

//nolint:dupword
const errorConfigText = `
plugin_definitions:
- name: lowQueue
  plugin_name: low-queue
  parameters:
    threshold: 10
`
