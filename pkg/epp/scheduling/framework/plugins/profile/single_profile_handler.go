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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const SingleProfileHandlerType = "single-profile"

// compile-time type assertion
var _ framework.ProfileHandler = &SingleProfileHandler{}

func SingleProfileHandlerFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewSingleProfileHandler(), nil
}

// NewSingleProfileHandler initializes a new SingleProfileHandler and returns its pointer.
func NewSingleProfileHandler() *SingleProfileHandler {
	return &SingleProfileHandler{}
}

// SingleProfileHandler handles a single profile which is always the primary profile.
type SingleProfileHandler struct{}

// Type returns the type of the Profiles Picker.
func (h *SingleProfileHandler) Type() string {
	return SingleProfileHandlerType
}

// Pick selects the SchedulingProfiles to run from the list of candidate profiles, while taking into consideration the request properties and the
// previously executed cycles along with their results.
func (h *SingleProfileHandler) Pick(_ context.Context, request *types.LLMRequest, profiles map[string]*framework.SchedulerProfile,
	profileResults map[string]*types.ProfileRunResult) map[string]*framework.SchedulerProfile {
	if len(profiles) == len(profileResults) { // all profiles have been executed already in previous call
		return map[string]*framework.SchedulerProfile{}
	}
	// return all profiles
	return profiles
}

func (h *SingleProfileHandler) ProcessResults(_ context.Context, _ *types.LLMRequest, profileResults map[string]*types.ProfileRunResult) *types.SchedulingResult {
	var firstKey string
	for key := range profileResults {
		firstKey = key
		break
	}

	return &types.SchedulingResult{
		ProfileResults:     profileResults,
		PrimaryProfileName: firstKey,
	}
}
