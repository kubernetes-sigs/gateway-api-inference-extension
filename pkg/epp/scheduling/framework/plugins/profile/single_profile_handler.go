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
	"errors"
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	SingleProfileHandlerType = "single-profile-handler"
)

// compile-time type assertion
var _ framework.ProfileHandler = &SingleProfileHandler{}

// SingleProfileHandlerFactory defines the factory function for SingleProfileHandler.
func SingleProfileHandlerFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewSingleProfileHandler().WithName(name), nil
}

// NewSingleProfileHandler initializes a new SingleProfileHandler and returns its pointer.
func NewSingleProfileHandler() *SingleProfileHandler {
	return &SingleProfileHandler{
		typedName: plugins.TypedName{Type: SingleProfileHandlerType, Name: SingleProfileHandlerType},
	}
}

// SingleProfileHandler handles a single profile which is always the primary profile.
type SingleProfileHandler struct {
	typedName plugins.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (h *SingleProfileHandler) TypedName() plugins.TypedName {
	return h.typedName
}

// WithName sets the name of the profile handler.
func (h *SingleProfileHandler) WithName(name string) *SingleProfileHandler {
	h.typedName.Name = name
	return h
}

// Pick selects the SchedulingProfiles to run from the list of candidate profiles, while taking into consideration the request properties and the
// previously executed cycles along with their results.
func (h *SingleProfileHandler) Pick(_ context.Context, _ *types.CycleState, request *types.LLMRequest, profiles map[string]*framework.SchedulerProfile,
	profileResults map[string]*types.ProfileRunResult) map[string]*framework.SchedulerProfile {
	if len(profiles) == len(profileResults) { // all profiles have been executed already in previous call
		return map[string]*framework.SchedulerProfile{}
	}
	// return all profiles
	return profiles
}

// ProcessResults handles the outcome of the profile runs after all profiles ran.
// It may aggregate results, log test profile outputs, or apply custom logic. It specifies in the SchedulingResult the
// key of the primary profile that should be used to get the request selected destination.
// When a profile run fails, its result in the profileResults map is nil.
func (h *SingleProfileHandler) ProcessResults(_ context.Context, _ *types.CycleState, _ *types.LLMRequest,
	profileResults map[string]*types.ProfileRunResult) (*types.SchedulingResult, error) {
	if len(profileResults) != 1 {
		return nil, errors.New("single profile handler is intended to be used with a single profile, failed to process multiple profiles")
	}

	var singleProfileName string
	for profileName := range profileResults {
		singleProfileName = profileName
		break
	}

	if profileResults[singleProfileName] == nil { // there was an error while running the profile
		return nil, fmt.Errorf("failed to run scheduler profile '%s'", singleProfileName)
	}

	return &types.SchedulingResult{
		ProfileResults:     profileResults,
		PrimaryProfileName: singleProfileName,
	}, nil
}
