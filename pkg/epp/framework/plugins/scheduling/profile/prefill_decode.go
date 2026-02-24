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

// Package profile provides profile handler plugin for the epp.
package profile

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strconv"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
)

const (
	// PdProfileHandlerType is the type of the PdProfileHandler
	PdProfileHandlerType = "pd-profile-handler"

	defaultDecodeProfile     = "decode"
	defaultPrefillProfile    = "prefill"
	defaultDeciderPluginName = AlwaysDisaggDeciderPluginType

	// AverageCharactersPerToken is an estimated average characters per token, used since the request we cached is not tokenized.
	AverageCharactersPerToken = 4

	// DataParallelPodHeader is the header name used to pass the data parallel pod address to the backend.
	DataParallelPodHeader = "x-data-parallel-host-port"
)

// pdDeciderPlugin interface for pd decider plugins
type pdDeciderPlugin interface {
	plugin.Plugin
	// disaggregate checks if disaggregated PD is required for the given request and endpoint.
	disaggregate(ctx context.Context, inputTokens int, endpoint scheduling.Endpoint) bool
}

type pdProfileHandlerParameters struct {
	DecodeProfile     string `json:"decodeProfile"`
	PrefillProfile    string `json:"prefillProfile"`
	PrimaryPort       int    `json:"primaryPort"`
	DeciderPluginName string `json:"deciderPluginName"`
}

// compile-time type assertion
var _ scheduling.ProfileHandler = &PdProfileHandler{}

// PdProfileHandlerFactory defines the factory function for the PdProfileHandler
func PdProfileHandlerFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := pdProfileHandlerParameters{
		DecodeProfile:     defaultDecodeProfile,
		PrefillProfile:    defaultPrefillProfile,
		PrimaryPort:       0,
		DeciderPluginName: defaultDeciderPluginName,
	}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' profile handler - %w", PdProfileHandlerType, err)
		}
	}

	if parameters.PrimaryPort != 0 {
		if parameters.PrimaryPort < 1 || parameters.PrimaryPort > 65535 {
			return nil, fmt.Errorf("invalid primaryPort: must be between 1 and 65535, got %d", parameters.PrimaryPort)
		}
	}

	if parameters.DeciderPluginName == "" {
		return nil, errors.New("decider plugin name is not defined")
	}

	pluginInstance := handle.Plugin(parameters.DeciderPluginName)
	if pluginInstance == nil {
		return nil, fmt.Errorf("invalid decider plugin type: %s", parameters.DeciderPluginName)
	}

	deciderPlugin, ok := pluginInstance.(pdDeciderPlugin)
	if !ok {
		return nil, fmt.Errorf("decider plugin of type: %s does not implement pdDeciderPlugin", parameters.DeciderPluginName)
	}

	handler, err := NewPdProfileHandler(parameters.PrefillProfile, parameters.DecodeProfile, parameters.PrimaryPort, deciderPlugin)

	if err != nil {
		return nil, err
	}

	return handler.WithName(name), nil

}

// NewPdProfileHandler initializes a new PdProfileHandler and returns its pointer.
func NewPdProfileHandler(prefillProfile, decodeProfile string, primaryPort int, deciderPlugin pdDeciderPlugin) (*PdProfileHandler, error) {
	result := &PdProfileHandler{
		typedName:      plugin.TypedName{Type: PdProfileHandlerType},
		decodeProfile:  decodeProfile,
		prefillProfile: prefillProfile,
		decider:        deciderPlugin,
	}
	if primaryPort != 0 {
		result.primaryPort = strconv.Itoa(primaryPort)
	}

	return result, nil
}

// PdProfileHandler handles scheduler profiles for PD.
type PdProfileHandler struct {
	typedName plugin.TypedName

	decodeProfile  string
	prefillProfile string
	primaryPort    string
	decider        pdDeciderPlugin
}

// TypedName returns the typed name of the plugin.
func (h *PdProfileHandler) TypedName() plugin.TypedName {
	return h.typedName
}

// WithName sets the name of the plugin.
func (h *PdProfileHandler) WithName(name string) *PdProfileHandler {
	h.typedName.Name = name
	return h
}

// Pick selects the SchedulingProfiles to run from the list of candidate profiles, while taking into consideration the request properties and the
// previously executed cycles along with their results.
func (h *PdProfileHandler) Pick(ctx context.Context, _ *scheduling.CycleState, request *scheduling.LLMRequest, profiles map[string]scheduling.SchedulerProfile,
	profileResults scheduling.ProfileResults) map[string]scheduling.SchedulerProfile {
	if _, executed := profileResults.DecodeResult(h.decodeProfile); !executed {
		// if decode profile was not executed yet, first let the scheduler run the decode profile
		return map[string]scheduling.SchedulerProfile{
			h.decodeProfile: profiles[h.decodeProfile],
		}
	}
	// otherwise, decode was already executed.

	// when a profile run fails its result value is nil. we need to check decode result before continuing to prefill
	// check if all configured profiles have been executed, or if decode failed, no need to run more profiles.
	decodeResult, _ := profileResults.DecodeResult(h.decodeProfile)
	if len(profiles) == len(profileResults) || decodeResult == nil {
		return map[string]scheduling.SchedulerProfile{}
	}

	inputTokens, err := getUserInputLenInTokens(request)
	if err != nil {
		log.FromContext(ctx).V(logutil.DEBUG).Error(err, "Failed to get user input")
		return nil
	}

	// Ensure we have at least one target endpoint
	if len(decodeResult.TargetEndpoints) == 0 {
		return map[string]scheduling.SchedulerProfile{}
	}

	decodeResult, _ = profileResults.DecodeResult(h.decodeProfile)
	if h.decider != nil && h.decider.disaggregate(ctx, inputTokens, decodeResult.TargetEndpoints[0]) {
		metrics.RecordPDDecision(request.TargetModel, metrics.DecisionTypePrefillDecode)
		// run the prefill profile
		return map[string]scheduling.SchedulerProfile{
			h.prefillProfile: profiles[h.prefillProfile],
		}
	}

	metrics.RecordPDDecision(request.TargetModel, metrics.DecisionTypeDecodeOnly)
	return map[string]scheduling.SchedulerProfile{} // do not run prefill
}

// ProcessResults handles the outcome of the profile runs after the selected profiles ran.
// In case of an error in any of the profiles, the matching entry in the profileResults will contain nil, to indicate there was
// an error while running the profile.
func (h *PdProfileHandler) ProcessResults(_ context.Context, _ *scheduling.CycleState, request *scheduling.LLMRequest,
	profileResults scheduling.ProfileResults) (*scheduling.SchedulingResult, error) {
	decodeRunResults, _ := profileResults.DecodeResult(h.decodeProfile)
	if decodeRunResults == nil { // if decode profile failed to run, we should fail
		return nil, errors.New("failed to find available decode workers")
	}
	// otherwise, decode ran successfully

	updatedResults := scheduling.ProfileResults{}

	// Add decode profile to result
	if h.primaryPort != "" {
		// Data Parallel is active

		if len(decodeRunResults.TargetEndpoints) > 0 {
			targetEndpoint := decodeRunResults.TargetEndpoints[0].GetMetadata()
			request.Headers[DataParallelPodHeader] = net.JoinHostPort(targetEndpoint.Address, targetEndpoint.Port)
		}

		updatedResult := scheduling.ProfileRunResult{
			TargetEndpoints: []scheduling.Endpoint{},
		}

		for _, target := range decodeRunResults.TargetEndpoints {
			updatedEndpointInfo := target.GetMetadata().Clone()
			// Port is string, primaryPort is string (from int in factory)
			updatedEndpointInfo.Port = h.primaryPort
			targetEndpoint := scheduling.NewEndpoint(updatedEndpointInfo, target.GetMetrics().Clone(), nil)
			updatedResult.TargetEndpoints = append(updatedResult.TargetEndpoints, targetEndpoint)
		}
		updatedResults[h.decodeProfile] = &updatedResult
	} else {
		updatedResults[h.decodeProfile] = decodeRunResults
	}

	// if both prefill and decode ran successfully
	if prefillResult, _ := profileResults.PrefillResult(h.prefillProfile); prefillResult != nil {
		// Add the prefill profile to the results
		updatedResults[h.prefillProfile] = prefillResult
	}

	return &scheduling.SchedulingResult{
		PrimaryProfileName: h.decodeProfile,
		ProfileResults:     updatedResults,
	}, nil
}

// returns length of user input in tokens
func getUserInputLenInTokens(request *scheduling.LLMRequest) (int, error) {
	if request.Body.Completions != nil { // assumed to be valid if not nil
		return len([]byte(request.Body.Completions.Prompt)) / AverageCharactersPerToken, nil
	}

	// must be chat-completions request at this point, return bytes of entire messages
	prompt, err := json.Marshal(request.Body.ChatCompletions.Messages)

	if err != nil {
		return 0, err
	}

	return len(prompt) / AverageCharactersPerToken, nil
}
