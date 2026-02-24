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

package profile

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
)

// Mock decider for testing
type mockDecider struct {
	shouldDisagg bool
}

func (m *mockDecider) Name() string {
	return "mock-decider"
}

func (m *mockDecider) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "mock-decider", Name: "mock-decider"}
}

func (m *mockDecider) Produces() map[string]any { return nil }
func (m *mockDecider) Consumes() map[string]any { return nil }

func (m *mockDecider) disaggregate(ctx context.Context, inputTokens int, endpoint scheduling.Endpoint) bool {
	return m.shouldDisagg
}

func TestPdProfileHandlerFactory(t *testing.T) {
	ctx := context.Background()
	// Register metrics to avoid panics if tests run in parallel or if metrics are accessed
	// But in unit tests usually we don't need real metrics unless we test them.
	// Reset metrics just in case.
	metrics.Reset()

	tests := []struct {
		name       string
		pluginName string
		params     map[string]any
		expectErr  bool
	}{
		{
			name:       "valid configuration with all defaults",
			pluginName: "default-handler",
			params:     map[string]any{},
			expectErr:  false,
		},
		{
			name:       "valid configuration with custom values",
			pluginName: "custom-handler",
			params: map[string]any{
				"decodeProfile":     "my-decode",
				"prefillProfile":    "my-prefill",
				"prefixPluginName":  "my-prefix-cache",
				"primaryPort":       8080,
				"deciderPluginName": PrefixBasedPDDeciderPluginType,
			},
			expectErr: false,
		},
		{
			name:       "zero primaryPort is allowed",
			pluginName: "zero-port",
			params: map[string]any{
				"primaryPort": 0,
			},
			expectErr: false,
		},
		{
			name:       "nonCachedTokens = 0 is allowed",
			pluginName: "zero-non-cached-tokens",
			params: map[string]any{
				"deciderPluginName": PrefixBasedPDDeciderPluginType,
			},
			expectErr: false,
		},
		{
			name:       "primaryPort below range should error",
			pluginName: "port-too-low",
			params:     map[string]any{"primaryPort": 0}, // OK
			expectErr:  false,
		},
		{
			name:       "primaryPort = 1 is valid",
			pluginName: "port-min",
			params:     map[string]any{"primaryPort": 1},
			expectErr:  false,
		},
		{
			name:       "primaryPort = 65535 is valid",
			pluginName: "port-max",
			params:     map[string]any{"primaryPort": 65535},
			expectErr:  false,
		},
		{
			name:       "empty decodeProfile is valid",
			pluginName: "empty-decode",
			params:     map[string]any{"decodeProfile": ""},
			expectErr:  false,
		},
		{
			name:       "empty prefillProfile is valid",
			pluginName: "empty-prefill",
			params:     map[string]any{"prefillProfile": ""},
			expectErr:  false,
		},
		{
			name:       "empty prefixPluginName is valid",
			pluginName: "empty-prefix-plugin",
			params:     map[string]any{"prefixPluginName": ""},
			expectErr:  false,
		},
		{
			name:       "primaryPort = 65536 should error",
			pluginName: "port-too-high",
			params:     map[string]any{"primaryPort": 65536},
			expectErr:  true,
		},
		{
			name:       "primaryPort = -10 should error",
			pluginName: "port-negative",
			params:     map[string]any{"primaryPort": -10},
			expectErr:  true,
		},
	}

	handle, err := createHandleWithDeciderPlugins(ctx)
	assert.NoError(t, err)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rawParams json.RawMessage
			if tt.params != nil {
				bytes, err := json.Marshal(tt.params)
				assert.NoError(t, err)
				rawParams = json.RawMessage(bytes)
			}
			plugin, err := PdProfileHandlerFactory(tt.pluginName, rawParams, handle)

			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, plugin)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, plugin)
			}
		})
	}
}

func TestPdProfileHandlerFactoryInvalidJSON(t *testing.T) {
	ctx := context.Background()

	invalidTests := []struct {
		name       string
		jsonParams string
	}{
		{
			name:       "malformed JSON",
			jsonParams: `{"deciderPluginName": `, // incomplete
		},
		{
			name:       "invalid decider plugin type",
			jsonParams: `{"deciderPluginName": "INVALID"}`,
		},
		{
			name:       "primaryPort as float",
			jsonParams: `{"primaryPort": 8080.5}`,
		},
	}

	handle, err := createHandleWithDeciderPlugins(ctx)
	assert.NoError(t, err)

	for _, tt := range invalidTests {
		t.Run(tt.name, func(t *testing.T) {
			rawParams := json.RawMessage(tt.jsonParams)
			plugin, err := PdProfileHandlerFactory("test", rawParams, handle)

			assert.Error(t, err)
			assert.Nil(t, plugin)
		})
	}
}

const DefaultTestPodPort = "8000"

// createEndpoint creates a mock Endpoint with customizable IP and port.
func createEndpoint(nsn k8stypes.NamespacedName, ipaddr, port string, labels map[string]string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: nsn,
			Address:        ipaddr,
			Port:           port,
			Labels:         labels,
		},
		nil,
		fwkdl.NewAttributes(),
	)
}

// newMockProfileRunResult creates a ProfileRunResult with Pods using the given port.
func newMockProfileRunResult(port string, endpointNames ...string) *scheduling.ProfileRunResult {
	endpoints := make([]scheduling.Endpoint, 0, len(endpointNames))
	for i, name := range endpointNames {
		ip := fmt.Sprintf("10.0.0.%d", i+1)
		endpoints = append(endpoints, createEndpoint(
			k8stypes.NamespacedName{Namespace: "default", Name: name},
			ip,
			port,
			map[string]string{},
		))
	}
	return &scheduling.ProfileRunResult{
		TargetEndpoints: endpoints,
	}
}

func newMockSchedulerProfile() scheduling.SchedulerProfile {
	return &mockSchedulerProfile{}
}

type mockSchedulerProfile struct{}

func (p *mockSchedulerProfile) Run(_ context.Context, _ *scheduling.LLMRequest, _ *scheduling.CycleState, _ []scheduling.Endpoint) (*scheduling.ProfileRunResult, error) {
	return &scheduling.ProfileRunResult{}, nil
}

// creates and returns llm completion request forthe given prompt
func createRequest(prompt string) *scheduling.LLMRequest {
	return &scheduling.LLMRequest{
		Body: &scheduling.LLMRequestBody{
			Completions: &scheduling.CompletionsRequest{
				Prompt: prompt,
			},
		},
	}
}

// returns array of profile names in the given profile pick result
func getProfilesFromResult(result map[string]scheduling.SchedulerProfile) []string {
	profiles := make([]string, len(result))
	index := 0

	for name := range result {
		profiles[index] = name
		index++
	}

	return profiles
}

func TestPdProfileHandler_Pick(t *testing.T) {
	ctx := context.Background()
	metrics.Reset()
	request := createRequest("hello world hello world hello world")

	profiles := map[string]scheduling.SchedulerProfile{
		"decode":  newMockSchedulerProfile(),
		"prefill": newMockSchedulerProfile(),
	}

	tests := []struct {
		name                 string
		nonCachedTokensLimit int
		cachedTokens         int
		profileResults       map[string]*scheduling.ProfileRunResult
		expectedProfiles     []string
	}{
		{
			name:                 "decode not executed yet → run decode",
			nonCachedTokensLimit: 10,
			profileResults:       map[string]*scheduling.ProfileRunResult{},
			expectedProfiles:     []string{defaultDecodeProfile},
		},
		{
			name:                 "decode failed (nil result) → run nothing",
			nonCachedTokensLimit: 10,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: nil,
			},
			expectedProfiles: []string{},
		},
		{
			name:                 "all profiles already executed → run nothing",
			nonCachedTokensLimit: 10,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile:  newMockProfileRunResult(DefaultTestPodPort, "pod1"),
				defaultPrefillProfile: newMockProfileRunResult(DefaultTestPodPort, "pod2"),
			},
			expectedProfiles: []string{},
		},
		{
			name: "has enough not-cached tokens → run prefill",
			// Need at least 4 non-cached tokens (16+ chars) to trigger disaggregated prefill
			// In this case: prompt length is 35 chars (8 tokens), cached length is 2 tokens -> disaggregated prefill should trigger
			nonCachedTokensLimit: 4,
			cachedTokens:         2,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: newMockProfileRunResult(DefaultTestPodPort, "pod1"),
			},
			expectedProfiles: []string{defaultPrefillProfile},
		},
		{
			name: "short non-cached suffix → skip prefill",
			// Need at least 4 non-cached tokens (16+ chars) to trigger disaggregated prefill
			// In this case: prompt length is 35 chars (8 tokens), cached length is 5 tokens -> skip prefill
			nonCachedTokensLimit: 4,
			cachedTokens:         5,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: newMockProfileRunResult(DefaultTestPodPort, "pod1"),
			},
			expectedProfiles: []string{},
		},
	}

	for _, tt := range tests {
		deciderPlugin, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: tt.nonCachedTokensLimit})
		assert.NoError(t, err)

		t.Run(tt.name, func(t *testing.T) {
			handler, err := NewPdProfileHandler(
				defaultPrefillProfile,
				defaultDecodeProfile,
				0,
				deciderPlugin,
			)
			assert.NoError(t, err)

			// set prefix to the given cached tokens number for pod "pod1" in decode profile results
			inputTokens := len(request.Body.Completions.Prompt) / AverageCharactersPerToken

			for profileName, profileRes := range tt.profileResults {
				if profileName == defaultDecodeProfile && profileRes != nil {
					for _, pod := range profileRes.TargetEndpoints {
						pod.Put(attrprefix.PrefixCacheMatchInfoKey,
							attrprefix.NewPrefixCacheMatchInfo(tt.cachedTokens, inputTokens, 1))
					}
				}
			}
			result := handler.Pick(ctx, nil, request, profiles, tt.profileResults)
			assert.ElementsMatch(t, tt.expectedProfiles, getProfilesFromResult(result))
		})
	}
}

func TestPdProfileHandler_PickSeries(t *testing.T) {
	ctx := context.Background()
	metrics.Reset()
	prompt := "hello world, hello world, hello world, hello world, hello world, hello world, hello world!"
	request := createRequest(prompt)
	longerRequest := createRequest(prompt + "123")
	longRequest := createRequest(prompt + prompt)

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  newMockSchedulerProfile(),
		defaultPrefillProfile: newMockSchedulerProfile(),
	}
	profileResults := map[string]*scheduling.ProfileRunResult{
		defaultDecodeProfile: newMockProfileRunResult(DefaultTestPodPort, "pod1"),
	}

	type testData struct {
		request          *scheduling.LLMRequest
		cachedTokens     int
		expectedProfiles []string
	}
	tests := []struct {
		name                 string
		nonCachedTokensLimit int
		tests                []testData
	}{
		{
			name:                 "same request twice",
			nonCachedTokensLimit: 2,
			tests: []testData{{
				request:          request,
				cachedTokens:     0,
				expectedProfiles: []string{defaultPrefillProfile},
			}, {
				request:          request,
				cachedTokens:     len(request.Body.Completions.Prompt) / AverageCharactersPerToken,
				expectedProfiles: []string{},
			}},
		}, {
			name: "short request and a little bit longer after it",
			// Need at least 2 non-cached tokens (8+ chars) to trigger disaggregated prefill
			// In this case: longer request is longer in 4 chars than the request -> no disaggregated prefill
			nonCachedTokensLimit: 2,
			tests: []testData{{
				request:          request,
				cachedTokens:     0,
				expectedProfiles: []string{defaultPrefillProfile},
			}, {
				request:          longerRequest,
				cachedTokens:     len(request.Body.Completions.Prompt) / AverageCharactersPerToken,
				expectedProfiles: []string{},
			}},
		}, {
			name: "short request and a long one after it",
			// Need at least 2 non-cached tokens (8+ chars) to trigger disaggregated prefill
			// In this case: long request is longer enough than the request -> should have disaggregated prefill
			nonCachedTokensLimit: 2,
			tests: []testData{{
				request:          request,
				cachedTokens:     0,
				expectedProfiles: []string{defaultPrefillProfile},
			}, {
				request:          longRequest,
				cachedTokens:     len(request.Body.Completions.Prompt) / AverageCharactersPerToken,
				expectedProfiles: []string{defaultPrefillProfile},
			}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			deciderPlugin, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: tt.nonCachedTokensLimit})
			assert.NoError(t, err)

			handler, err := NewPdProfileHandler(
				defaultPrefillProfile,
				defaultDecodeProfile,
				0,
				deciderPlugin,
			)
			assert.NoError(t, err)

			// run sequences of request
			for _, innerTest := range tt.tests {
				cs := &scheduling.CycleState{}

				// set prefix to the given cached tokens number for pod "pod1" in decode profile results
				inputTokens := len(innerTest.request.Body.Completions.Prompt) / AverageCharactersPerToken

				for profileName, profileRes := range profileResults {
					if profileName == defaultDecodeProfile && profileRes != nil {
						for _, endpoint := range profileRes.TargetEndpoints {
							endpoint.Put(attrprefix.PrefixCacheMatchInfoKey,
								attrprefix.NewPrefixCacheMatchInfo(innerTest.cachedTokens, inputTokens, 1))
						}
					}
				}

				result := handler.Pick(ctx, cs, innerTest.request, profiles, profileResults)
				assert.ElementsMatch(t, innerTest.expectedProfiles, getProfilesFromResult(result))
			}
		})
	}
}

func TestPdProfileHandler_AlwaysDisagg(t *testing.T) {
	ctx := context.Background()
	prompt := "hello world"
	request := createRequest(prompt)

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  newMockSchedulerProfile(),
		defaultPrefillProfile: newMockSchedulerProfile(),
	}
	profileResults := map[string]*scheduling.ProfileRunResult{
		defaultDecodeProfile: newMockProfileRunResult(DefaultTestPodPort, "pod1"),
	}

	// AlwaysDisaggPDDecider should always return true, even if prefix match is perfect.
	deciderPlugin := NewAlwaysDisaggPDDecider()

	handler, err := NewPdProfileHandler(
		defaultPrefillProfile,
		defaultDecodeProfile,
		0,
		deciderPlugin,
	)
	assert.NoError(t, err)

	// Case 1: Perfect prefix match (should skip prefill with PrefixBasedDecider, but run it with AlwaysDisagg)
	cs := &scheduling.CycleState{}
	inputTokens := len(request.Body.Completions.Prompt) / AverageCharactersPerToken

	// Simulate perfect cache hit on decode node
	for _, profileRes := range profileResults {
		for _, endpoint := range profileRes.TargetEndpoints {
			endpoint.Put(attrprefix.PrefixCacheMatchInfoKey,
				attrprefix.NewPrefixCacheMatchInfo(inputTokens, inputTokens, 1))
		}
	}

	result := handler.Pick(ctx, cs, request, profiles, profileResults)
	// Expect prefill because decider says "Always"
	assert.Contains(t, getProfilesFromResult(result), defaultPrefillProfile)

	// Case 2: No cache hit (should run prefill anyway)
	for _, profileRes := range profileResults {
		for _, endpoint := range profileRes.TargetEndpoints {
			endpoint.Put(attrprefix.PrefixCacheMatchInfoKey,
				attrprefix.NewPrefixCacheMatchInfo(0, inputTokens, 1))
		}
	}
	result = handler.Pick(ctx, cs, request, profiles, profileResults)
	assert.Contains(t, getProfilesFromResult(result), defaultPrefillProfile)
}

func TestPdProfileHandler_ProcessResults(t *testing.T) {
	tests := []struct {
		name           string
		primaryPort    int
		profileResults map[string]*scheduling.ProfileRunResult
		expectError    bool
		checkResult    func(*testing.T, *scheduling.SchedulingResult, map[string]string)
	}{
		{
			name: "decode failed → error",
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: nil,
			},
			expectError: true,
		},
		{
			name:        "decode success, no prefill, no primaryPort",
			primaryPort: 0,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: newMockProfileRunResult(DefaultTestPodPort, "pod1"),
			},
			expectError: false,
			checkResult: func(t *testing.T, res *scheduling.SchedulingResult, headers map[string]string) {
				assert.Equal(t, defaultDecodeProfile, res.PrimaryProfileName)
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultPrefillProfile)
				metadata := res.ProfileResults[defaultDecodeProfile].TargetEndpoints[0].GetMetadata()
				assert.Equal(t, DefaultTestPodPort, metadata.Port)
				// check header is empty or not set
				val, ok := headers[DataParallelPodHeader]
				assert.True(t, !ok || val == "")
			},
		},
		{
			name:        "decode success, with prefill",
			primaryPort: 0,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile:  newMockProfileRunResult(DefaultTestPodPort, "pod1"),
				defaultPrefillProfile: newMockProfileRunResult(DefaultTestPodPort, "pod2"),
			},
			expectError: false,
			checkResult: func(t *testing.T, res *scheduling.SchedulingResult, _ map[string]string) {
				assert.Equal(t, defaultDecodeProfile, res.PrimaryProfileName)
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.Contains(t, res.ProfileResults, defaultPrefillProfile)
			},
		},
		{
			name:        "with primaryPort → port updated and header set",
			primaryPort: 9000,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: newMockProfileRunResult(DefaultTestPodPort, "pod1"),
			},
			expectError: false,
			checkResult: func(t *testing.T, res *scheduling.SchedulingResult, headers map[string]string) {
				metadata := res.ProfileResults[defaultDecodeProfile].TargetEndpoints[0].GetMetadata()
				assert.Equal(t, "9000", metadata.Port)

				hostPort := headers[DataParallelPodHeader]
				assert.Equal(t, "10.0.0.1:8000", hostPort)
			},
		},
	}

	for _, tt := range tests {
		deciderPlugin, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 0})
		assert.NoError(t, err)

		t.Run(tt.name, func(t *testing.T) {
			handler, err := NewPdProfileHandler(
				defaultPrefillProfile,
				defaultDecodeProfile,
				tt.primaryPort,
				deciderPlugin,
			)
			assert.NoError(t, err)

			headers := make(map[string]string)
			req := &scheduling.LLMRequest{
				Headers: headers,
			}
			result, err := handler.ProcessResults(context.Background(), &scheduling.CycleState{}, req, tt.profileResults)

			if tt.expectError {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, result)
			tt.checkResult(t, result, headers)
		})
	}
}

func createHandleWithDeciderPlugins(ctx context.Context) (plugin.Handle, error) {
	handle := plugin.NewEppHandle(ctx, func() []k8stypes.NamespacedName { return nil })
	plugin1, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 4})
	if err != nil {
		return nil, err
	}
	handle.AddPlugin(PrefixBasedPDDeciderPluginType, plugin1)

	plugin3, err := NewAlwaysDisaggPDDecider(), nil
	if err != nil {
		return nil, err
	}
	handle.AddPlugin(AlwaysDisaggDeciderPluginType, plugin3)

	// Add mock decider
	plugin2 := &mockDecider{shouldDisagg: true}
	handle.AddPlugin("mock-decider", plugin2)

	return handle, nil
}
