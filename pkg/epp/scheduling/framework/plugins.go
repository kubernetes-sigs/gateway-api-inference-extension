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

package framework

import (
	"context"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	ProfilePickerExtensionPoint          = "ProfilePicker"
	FilterExtensionPoint                 = "Filter"
	ScorerExtensionPoint                 = "Scorer"
	PickerExtensionPoint                 = "Picker"
	ProcessProfilesResultsExtensionPoint = "ProcessProfilesResults"
)

// ProfileHandler defines the extension points for handling multi SchedulerProfile instances.
// More specifically, this interface defines the 'Pick' and 'ProcessResults' extension points.
type ProfileHandler interface {
	plugins.Plugin
	// Pick selects the SchedulingProfiles to run from a list of candidate profiles, while taking into consideration the request properties
	// and the previously executed SchedluderProfile cycles along with their results.
	Pick(ctx context.Context, cycleState *types.CycleState, request *types.LLMRequest, profiles map[string]*SchedulerProfile,
		profileResults map[string]*types.ProfileRunResult) map[string]*SchedulerProfile

	// ProcessResults handles the outcome of the profile runs after all profiles ran.
	// It may aggregate results, log test profile outputs, or apply custom logic. It specifies in the SchedulingResult the
	// key of the primary profile that should be used to get the request selected destination.
	// When a profile run fails, its result in the profileResults map is nil.
	ProcessResults(ctx context.Context, cycleState *types.CycleState, request *types.LLMRequest,
		profileResults map[string]*types.ProfileRunResult) (*types.SchedulingResult, error)
}

// Filter defines the interface for filtering a list of pods based on context.
type Filter interface {
	plugins.Plugin
	Filter(ctx context.Context, cycleState *types.CycleState, request *types.LLMRequest, pods []types.Pod) []types.Pod
}

// Scorer defines the interface for scoring a list of pods based on context.
// Scorers must score pods with a value within the range of [0,1] where 1 is the highest score.
// If a scorer returns value greater than 1, it will be treated as score 1.
// If a scorer returns value lower than 0, it will be treated as score 0.
type Scorer interface {
	plugins.Plugin
	Score(ctx context.Context, cycleState *types.CycleState, request *types.LLMRequest, pods []types.Pod) map[types.Pod]float64
}

// Picker picks the final pod(s) to send the request to.
type Picker interface {
	plugins.Plugin
	Pick(ctx context.Context, cycleState *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult
}
