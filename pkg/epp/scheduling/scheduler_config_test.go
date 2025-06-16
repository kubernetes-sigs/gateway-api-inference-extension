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
	"testing"

	commonconfig "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/common/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

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
		theConfig, err := commonconfig.LoadConfig([]byte(test.configText), "")
		if err != nil {
			if test.wantErr {
				continue
			}
			t.Fatalf("LoadConfig returned unexpected error: %v", err)
		}
		instantiatedPlugins, err := commonconfig.LoadPluginReferences(theConfig, testHandle{})
		if err != nil {
			if test.wantErr {
				continue
			}
			t.Fatalf("LoadPluginReferences returned unexpected error: %v", err)
		}

		_, err = LoadSchedulerConfig(theConfig.SchedulingProfiles, instantiatedPlugins, log)
		if err != nil {
			if !test.wantErr {
				t.Errorf("LoadSchedulerConfig returned an unexpected error. error %v", err)
			}
		} else if test.wantErr {
			t.Errorf("LoadSchedulerConfig did not return an expected error (%s)", test.name)
		}
	}
}

type testHandle struct {
}

func registerNeededPlgugins() {
	allPlugins := map[string]plugins.Factory{
		filter.LowQueueFilterName:        filter.LowQueueFilterFactory,
		prefix.PrefixCachePluginName:     prefix.PrefixCachePluginFactory,
		picker.MaxScorePickerName:        picker.MaxScorePickerFactory,
		picker.RandomPickerName:          picker.RandomPickerFactory,
		profile.SingleProfileHandlerName: profile.SingleProfileHandlerFactory,
	}
	for name, factory := range allPlugins {
		plugins.Register(name, factory)
	}
}

// The following multi-line string constants, cause false positive lint errors (dupword)

//nolint:dupword
const successConfigText = `
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
