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

// --- Valid Configurations ---

// successConfigText represents a fully populated, valid configuration.
// It uses a mix of explicit names and type-derived names.
const successConfigText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-plugin
  parameters:
    threshold: 10
- name: profileHandler
  type: test-profile-handler
- type: test-scorer
  parameters:
    blockSize: 32
- name: testPicker
  type: test-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
  - pluginRef: test-scorer
    weight: 50
  - pluginRef: testPicker
featureGates:
- dataLayer
saturationDetector:
  queueDepthThreshold: 10
  kvCacheUtilThreshold: 0.8
  metricsStalenessThreshold: 100ms
`

// successNoProfilesText represents a valid config with plugins but no profiles.
// The loader should apply the system default profile automatically.
const successNoProfilesText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-plugin
  parameters:
    threshold: 10
`

// successSchedulerConfigText represents a complex scheduler setup.
const successSchedulerConfigText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: testScorer
  type: test-scorer
  parameters:
    blockSize: 32
- name: maxScorePicker
  type: max-score-picker
- name: profileHandler
  type: single-profile-handler
- name: testSource
  type: test-source
- name: testExtractor
  type: test-extractor
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: testScorer
    weight: 50
  - pluginRef: maxScorePicker
data:
  sources:
  - pluginRef: testSource
    extractors:
    - pluginRef: testExtractor
featureGates:
- dataLayer
`

// successWithNoWeightText tests that scorers receive the default weight if unspecified.
const successWithNoWeightText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: single-profile-handler
- name: testScorer
  type: test-scorer
  parameters:
    blockSize: 32
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: testScorer
`

// successWithNoProfileHandlersText tests that a default profile handler is injected.
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

// successPickerBeforeScorerText tests the regression case where a Picker appears before a Scorer (without weight) in
// the plugin list.
const successPickerBeforeScorerText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: single-profile-handler
- type: test-picker
- type: test-scorer
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test-picker
  - pluginRef: test-scorer
`

// --- Invalid Configurations (Syntax/Structure) ---

// errorBadYamlText contains invalid YAML syntax.
const errorBadYamlText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- testing 1 2 3
`

// errorBadPluginReferenceText is missing the required 'type' field.
const errorBadPluginReferenceText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- parameters:
    a: 1234
`

// errorBadPluginReferencePluginText references a plugin type that does not exist in the registry.
const errorBadPluginReferencePluginText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: testx
  type: unknown-plugin-type
- name: profileHandler
  type: test-profile-handler
`

// errorBadPluginJsonText has invalid JSON in parameters (string where int expected).
const errorBadPluginJsonText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: single-profile-handler
- name: testScorer
  type: test-scorer
  parameters:
    blockSize: asdf
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: testScorer
    weight: 50
`

// errorUnknownFeatureGateText includes a feature gate not defined in the code.
const errorUnknownFeatureGateText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-plugin
  parameters:
    threshold: 10
featureGates:
- unknown-gate
`

// --- Invalid Configurations (Logical/Architectural) ---

// errorNoProfileNameText is missing the required profile name.
const errorNoProfileNameText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-plugin
  parameters:
    threshold: 10
- name: profileHandler
  type: test-profile-handler
schedulingProfiles:
- plugins:
  - pluginRef: test1
`

// errorBadProfilePluginText is missing the required pluginRef.
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

// errorBadProfilePluginRefText references a plugin name that wasn't defined.
const errorBadProfilePluginRefText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: non-existent-plugin
`

// errorDuplicatePluginText defines the same plugin name twice.
const errorDuplicatePluginText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-plugin
  parameters:
    threshold: 10
- name: test1
  type: test-plugin
  parameters:
    threshold: 20
- name: profileHandler
  type: test-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
`

// errorDuplicateProfileText defines the same profile name twice.
const errorDuplicateProfileText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-plugin
  parameters:
    threshold: 10
- name: test2
  type: test-plugin
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

// errorTwoPickersText defines multiple pickers in a single profile (invalid).
const errorTwoPickersText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: single-profile-handler
- name: maxScore
  type: max-score-picker
- name: random
  type: test-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
  - pluginRef: random
`

// errorTwoProfileHandlersText defines multiple profile handlers (global singleton).
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

// errorNoProfileHandlersText fails to define any profile handler.
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

// errorMultiProfilesUseSingleProfileHandlerText uses SingleProfileHandler with multiple profiles.
const errorMultiProfilesUseSingleProfileHandlerText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: profileHandler
  type: single-profile-handler
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

// errorMissingDataConfigText has the datalayer enabled without config
const errorMissingDataConfigText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-one
  parameters:
    threshold: 10
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
featureGates:
- dataLayer
`

// errorBadSourceReferenceText has a bad DataSource plugin reference
const errorBadSourceReferenceText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-one
  parameters:
    threshold: 10
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
data:
  sources:
  - pluginRef: test-one
featureGates:
- dataLayer
`

// errorBadExtractorReferenceText has a bad Extractor plugin reference
const errorBadExtractorReferenceText = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: test1
  type: test-one
  parameters:
    threshold: 10
- type: test-source
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: test1
data:
  sources:
  - pluginRef: test-source
    extractors:
    - test-one
featureGates:
- dataLayer
`
