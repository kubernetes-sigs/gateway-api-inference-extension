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

package profilepicker

import (
	"context"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// compile-time type assertion
var _ framework.ProfilePicker = &AllProfilesPicker{}

// NewAllProfilesPicker initializes a new AllProfilesPicker and returns its pointer.
func NewAllProfilesPicker() *AllProfilesPicker {
	return &AllProfilesPicker{}
}

// AllProfilesPicker picks all profiles always.
type AllProfilesPicker struct{}

// Name returns the name of the Profiles Picker.
func (p *AllProfilesPicker) Name() string {
	return "all-profiles"
}

// Pick selects the SchedulingProfiles to run from the list of candidate profiles, while taking into consideration the request properties and the
// previously executed cycles along with their results.
func (p *AllProfilesPicker) Pick(ctx context.Context, request *types.LLMRequest, profiles map[string]*framework.SchedulerProfile,
	executionResults map[string]*types.Result) map[string]*framework.SchedulerProfile {
	if len(profiles) == len(executionResults) { // all profiles have been executed already in previous call
		return map[string]*framework.SchedulerProfile{}
	}
	// return all profiles
	return profiles
}
