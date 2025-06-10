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
	ProfilePickerType   = "ProfilePicker"
	FilterPluginType    = "Filter"
	ScorerPluginType    = "Scorer"
	PickerPluginType    = "Picker"
	PostCyclePluginType = "PostCycle"
)

// ProfilePicker selects the SchedulingProfiles to run from a list of candidate profiles, while taking into consideration the request properties
// and the previously executed SchedluderProfile cycles along with their results.
type ProfilePicker interface {
	plugins.Plugin
	Pick(ctx context.Context, request *types.LLMRequest, profiles map[string]*SchedulerProfile, executionResults map[string]*types.Result) map[string]*SchedulerProfile
}

// Filter defines the interface for filtering a list of pods based on context.
type Filter interface {
	plugins.Plugin
	Filter(ctx context.Context, request *types.LLMRequest, cycleState *types.CycleState, pods []types.Pod) []types.Pod
}

// Scorer defines the interface for scoring a list of pods based on context.
// Scorers must score pods with a value within the range of [0,1] where 1 is the highest score.
type Scorer interface {
	plugins.Plugin
	Score(ctx context.Context, request *types.LLMRequest, cycleState *types.CycleState, pods []types.Pod) map[types.Pod]float64
}

// Picker picks the final pod(s) to send the request to.
type Picker interface {
	plugins.Plugin
	Pick(ctx context.Context, cycleState *types.CycleState, scoredPods []*types.ScoredPod) *types.Result
}

// PostCycle is called by the scheduler after it selects a targetPod for the request in the SchedulerProfile cycle.
type PostCycle interface {
	plugins.Plugin
	PostCycle(ctx context.Context, cycleState *types.CycleState, res *types.Result)
}
