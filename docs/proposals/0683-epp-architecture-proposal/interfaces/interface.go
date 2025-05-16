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
	"sync"

	scheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// Plugin is the parent type for all the scheduling framework plugins.
type Plugin interface {
	Name() string
}

type Endpoint interface {
	GetState() EndpointState
	GetScore() float32
	SetScore(val float32)
}

type EndpointState struct {
	// only need to use a sync.Map if we do not plan on snapshotting data.
	storage sync.Map
}

type SchedulingResult struct {
	results map[string][]Endpoint
}

type Scheduler interface {
	Plugin
	// ProfileSelection selects scheduling profiles through the implemented
	// logic, and returns a subset of the registered scheduling profiles.
	ProfileSelection() map[string]SchedulingProfile

	// SchedulingProfiles lists all of the scheduling profiles registered
	// with the scheduler.
	SchedulingProfiles() map[string]SchedulingProfile

	// SchedulingResult takes the output of the result(s) of the scheduling cycle(s)
	// and makes sense of the data to be consumed by request control.
	// For example: suppose you have 2 profiles ShadowBoxing Profile & Production Profile.
	// SchedulingResult would know to simply log the result of ShadowBoxing
	// profile, and do nothing else with it.
	SchedulingResult(map[string][]Endpoint) SchedulingResult
}

// SchedulingProfile is an interface to used to describe a profile that will
// run for a given scheduling cycle.
type SchedulingProfile interface {
	Plugin
	// PreSchedulePlugins are optional, and will be ran at the start of a
	// scheduling cycle. This should be scoped to any foundational work needed
	// that is custom to this scheduling profile.
	PreSchedulePlugins() []PreSchedule
	// Filters lists all Filter plugins associated with this Profile. Filters
	// are optional.
	Filters() []Filter
	// Scorers lists all Score plugins associated with this Profile. At
	// least 1 scorer must be registered for a profile to be valid.
	Scorers() map[Scorer]int
	// Selection returns the function that picks the endpoint(s).
	Selection() Picker
	// PostSchedulePlugins lists all Filter plugins associated with this
	// Profile. PostSchedulePlugins are ran after every scheduling cycle,
	// and are optional.
	PostSchedulePlugins() []PostSchedule
}

// Preschedule will be ran at the start of a scheduling cycle. This should be
// scoped to any foundational work needed that is custom to this scheduling
// profile.
type PreSchedule interface {
	Plugin
	PreSchedule(ctx context.Context, state scheduling.CycleState, endpoints []Endpoint)
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
	Selection(ctx context.Context, state scheduling.CycleState, endpoints []Endpoint) []Endpoint
}

// PostSchedule runs per-scheduling cycle, and is part of a scheduling profile.
// PostSchedule performs any remaining work needed for the scheduling cycle.
// PostSchedule is not expected to change any values of the parameters.
type PostSchedule interface {
	Plugin
	PostSchedule(ctx context.Context, state scheduling.CycleState, selectedEndpoints []Endpoint)
}
