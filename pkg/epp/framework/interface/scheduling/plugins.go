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
	"context"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// ScorerCategory marks the preference a scorer applies when scoring candidate endpoints.
type ScorerCategory string

const (
	// Affinity indicates a scorer that prefers endpoints with existing locality, such as kv-cache or session-related state,
	// and therefore tends to give higher scores to the same endpoints.
	Affinity ScorerCategory = "Affinity"

	// Distribution indicates a scorer that prefers spreading requests evenly across all candidate endpoints to avoid
	// hotspots and improve overall utilization.
	Distribution ScorerCategory = "Distribution"

	// Balance indicates a scorer that its preference is balanced between Affinity and Distribution.
	Balance ScorerCategory = "Balance"
)

// ProfileHandler defines the extension points for handling multi SchedulerProfile instances.
// More specifically, this interface defines the 'Pick' and 'ProcessResults' extension points.
type ProfileHandler interface {
	plugin.Plugin
	// Pick selects the SchedulingProfiles to run from a list of candidate profiles, while taking into consideration the request properties
	// and the previously executed SchedluderProfile cycles along with their results.
	Pick(ctx context.Context, cycleState *CycleState, request *LLMRequest, profiles map[string]SchedulerProfile,
		profileResults map[string]*ProfileRunResult) map[string]SchedulerProfile

	// ProcessResults handles the outcome of the profile runs after all profiles ran.
	// It may aggregate results, log test profile outputs, or apply custom logic. It specifies in the SchedulingResult the
	// key of the primary profile that should be used to get the request selected destination.
	// When a profile run fails, its result in the profileResults map is nil.
	ProcessResults(ctx context.Context, cycleState *CycleState, request *LLMRequest,
		profileResults map[string]*ProfileRunResult) (*SchedulingResult, error)
}

// Filter defines the interface for filtering a list of pods based on context.
type Filter interface {
	plugin.Plugin
	Filter(ctx context.Context, cycleState *CycleState, request *LLMRequest, pods []Endpoint) []Endpoint
}

// Scorer defines the interface for scoring a list of pods based on context.
// Scorers must score pods with a value within the range of [0,1] where 1 is the highest score.
// If a scorer returns value greater than 1, it will be treated as score 1.
// If a scorer returns value lower than 0, it will be treated as score 0.
type Scorer interface {
	plugin.Plugin
	Category() ScorerCategory
	Score(ctx context.Context, cycleState *CycleState, request *LLMRequest, pods []Endpoint) map[Endpoint]float64
}

// Picker picks the final pod(s) to send the request to.
type Picker interface {
	plugin.Plugin
	Pick(ctx context.Context, cycleState *CycleState, scoredPods []*ScoredEndpoint) *ProfileRunResult
}
