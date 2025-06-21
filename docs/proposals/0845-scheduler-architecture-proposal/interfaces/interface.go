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

type Endpoint struct {
	State EndpointState
}

type EndpointState struct {
	// storage is per Scheduling Cycle, and so has no thread-safe concerns.
	storage map[string]any //nolint:unused
}

// Request is a structured representation of the fields we parse out of the Request body.
type Request struct {
	// RequestId is the Envoy generated Id for the request being processed
	RequestId string
	// TargetModel is the final target model after traffic split.
	TargetModel string
	// Prompt is the prompt that was sent in the request body.
	Prompt string
	// Headers is a map of the request headers.
	Headers map[string]string
}

// ScoredEndpoint encapsulates Endpoint with its Score.
// The lifecycle of an endpoint is typically different than a lifecycle of a request.
// This is intended to be used only internally by Scheduler logic and/or scheduler plugins within the lifecycle of the request.
// When returning the selected Endpoint(s) out of the Scheduler, an Endpoint is returned without the score.
type ScoredEndpoint struct {
	Endpoint
	Score float64
}

type Scheduler struct {
	SchedulerConfig
}

// SchedulerConfig is the struct that maps to the configuration file that should be further discussed.
// the configuration file should include the ProfileHandler plugin as well as the profiles with their plugins.
type SchedulerConfig struct {
	// exactly one ProfileHandler instance is required.
	profileHandler ProfileHandler //nolint:unused
	// map from profile name to its set of plugins.
	profiles map[string]*SchedulerProfile //nolint:unused
}

// SchedulerProfile is used to describe a profile that will
// run for a given scheduling cycle.
type SchedulerProfile struct {
	// Filters lists all Filter plugins associated with this Profile.
	// Filters are optional.
	filters []Filter //nolint:unused
	// Scorers lists all Score plugins associated with this Profile.
	// Scorers are optional.
	scorers []*WeightedScorer //nolint:unused
	// Picker returns the function that picks the endpoint(s). Picker is required.
	picker Picker //nolint:unused
}

type SchedulingResult struct {
	ProfileResults     map[string][]*Endpoint // a map from profile name to its scheduling result
	PrimaryProfileName string                 // key of the primary profile, its selected endpoints will be used by default as the destination
}

// Plugin is the parent type for all the scheduling framework plugins.
type Plugin interface {
	Type() string
}

// ProfileHandler defines the interface for handling multi SchedulerProfile instances.
// More specifically, this interfaction defines two extension points, 'PickProfiles'
// which runs iteratively, and 'ProcessProfilesResults' which runs after all profiles runs complete
// and process the results of all profiles.
type ProfileHandler interface {
	Plugin
	// Pick picks the SchedulingProfile objects to run from a list of candidate profiles,
	// while taking into consideration the request properties
	// and the previously executed SchedluderProfile runs along with their results.
	// returns:
	// - profiles - A subset of the registered scheduling profiles to be ran in next iteration
	Pick(request *Request, profiles map[string]*SchedulerProfile, executionResults map[string][]*ScoredEndpoint) map[string]*SchedulerProfile

	// ProcessResults handles the outcome of each profile run.
	// It may aggregate results, log test profile outputs, or apply custom logic. It specifies in the SchedulingResult the
	// key of the primary profile that should be used to get the request selected destination.
	// Example: suppose you have 2 profiles ShadowBoxing Profile & Production Profile.
	// ProcessProfileResults would know to simply log the result of ShadowBoxing
	// profile, and do nothing else with it.
	ProcessResults(request *Request, profileResults map[string][]*ScoredEndpoint) *SchedulingResult
}

// Filter runs before any scoring, and remove endpoints that are not fit for selection.
// The framework will return an error to the client if the endpoints are filtered to zero.
type Filter interface {
	Plugin
	Filter(ctx context.Context, request *Request, state *scheduling.CycleState, endpoints []*Endpoint) []*Endpoint
}

// Scorer applies a score to each remaining endpoint provided.
// Scorers SHOULD keep their score values in a normalized range: [0-1].
// Any weighting should be added at the SchedulerProfile configuration level.
type Scorer interface {
	Plugin
	Score(ctx context.Context, request *Request, state *scheduling.CycleState, endpoints []*Endpoint) []*ScoredEndpoint
}

// WeightedScorer is a struct that encapsulates a scorer with its weight.
// We need this struct in order to be able to keep scorers in profile as a slice instead of a map.
// This is very useful for having a generic AddPlugin function that registers a plugin to all its extension points.
// Using a map is much less convenient for this purpose.
type WeightedScorer struct {
	Scorer
	weight int //nolint:unused
}

// Picker selects the endpoint(s) from the provided list of scored endpoints.
// Picker MUST return, one endpoint at minimum.
type Picker interface {
	Plugin
	Pick(ctx context.Context, state *scheduling.CycleState, endpoints []*ScoredEndpoint) []*ScoredEndpoint
}
