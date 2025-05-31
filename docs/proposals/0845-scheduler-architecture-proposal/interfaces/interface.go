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

	scheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// READER NOTE: Currently CycleState is assumed to have appropriate request data rather that making a new object.

// Plugin is the parent type for all the scheduling framework plugins.
type Plugin interface {
	Name() string
}

type Endpoint struct {
	State EndpointState
	Score float64
}

type EndpointState struct {
	// storage is per Scheduling Cycle, and so has no thread-safe concerns.
	storage map[string]any
}

type SchedulingResult struct {
	results map[string][]Endpoint
}

// Scheduler is the implementation of a... scheduler.
// The scheduler object is created at startup using the provided configuration.
type Scheduler interface {
	// PreSchedule selects scheduling profiles through the implemented
	// logic, and returns:
	// - profiles - A subset of the registered scheduling profiles to be ran
	PreSchedule(request map[string]any, data scheduling.CycleState, results map[string][]Endpoint) map[string]SchedulingProfile

	// PostSchedule recieves the output of the result(s) of the scheduling cycle(s)
	// and makes sense of the data to be consumed by the calling system.
	// For example: suppose you have 2 profiles ShadowBoxing Profile & Production Profile.
	// PostSchedule would know to simply log the result of ShadowBoxing
	// profile, and do nothing else with it.
	PostSchedule(profileResults map[string][]Endpoint) SchedulingResult
}

// SchedulingProfile is used to describe a profile that will
// run for a given scheduling cycle.
type SchedulingProfile struct {
	// Name of the profile.
	Name string
	// Filters lists all Filter plugins associated with this Profile. Filters
	// are optional.
	Filters []Filter
	// Scorers lists all Score plugins associated with this Profile. Scorers
	// are optional.
	Scorers map[Scorer]int
	// Picker returns the function that picks the endpoint(s). Picker is required.
	Picker Picker
}

// Filter runs before any scoring, and remove endpoints that are not fit for
// selection. The framework will return an error to the client if the endpoints
// are filtered to zero.
type Filter interface {
	Plugin
	Filter(ctx context.Context, state scheduling.CycleState, endpoints []Endpoint) []Endpoint
}

// Scorer applies a score to each remaining endpoint provided. Scorers SHOULD
// keep their score values in a normalized range: [0-1]. Any weighting should
// be added at the SchedulingProfile configuration level.
type Scorer interface {
	Plugin
	Score(ctx context.Context, state scheduling.CycleState, endpoints []Endpoint) []Endpoint
}

// Picker selects the endpoint(s) from the provided list of scored endpoints.
// Picker MUST return, one endpoint at minimum.
type Picker interface {
	Plugin
	Pick(ctx context.Context, state scheduling.CycleState, endpoints []Endpoint) []Endpoint
}

type PostResponse interface {
	Plugin
	PostResponse(ctx context.Context, request map[string]any, response map[string]any)
}
