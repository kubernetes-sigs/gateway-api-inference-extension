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

package slo_aware_router

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
	SLOAwareProfileHandlerType  = "predicted-latency-profile-handler"
	NoLatencyRoutingProfileName = "predicted-latency-no-routing"
	PrefixProfileName           = "predicted-latency-prefix"
	LatencyRoutingProfileName   = "predicted-latency-routing"

	// Boolean header string for whether to use predictor based scheduling
	PreictionBasedSchedulingHeaderKey = "x-prediction-based-scheduling-off"
)

// compile-time type assertion
var _ framework.ProfileHandler = &SLOAwareProfileHandler{}

// SLOAwareProfileHandlerFactory defines the factory function for SLOAwareProfileHandler.
func SLOAwareProfileHandlerFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewSLOAwareProfileHandler().WithName(name), nil
}

// NewSLOAwareProfileHandler initializes a new SLOAwareProfileHandler and returns its pointer.
func NewSLOAwareProfileHandler() *SLOAwareProfileHandler {
	return &SLOAwareProfileHandler{
		typedName: plugins.TypedName{Type: SLOAwareProfileHandlerType, Name: SLOAwareProfileHandlerType},
	}
}

// SLOAwareProfileHandler handles two profiles: the default profile and the SLO profile.
// When the request has PredictorBasedScheduling=true, it uses the SLO profile result to select
// the destination pod. Otherwise, it uses the default profile result.
type SLOAwareProfileHandler struct {
	typedName plugins.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (h *SLOAwareProfileHandler) TypedName() plugins.TypedName {
	return h.typedName
}

// WithName sets the name of the profile handler.
func (h *SLOAwareProfileHandler) WithName(name string) *SLOAwareProfileHandler {
	h.typedName.Name = name
	return h
}

// Pick selects the SchedulingProfiles to run from the list of candidate profiles, while taking into consideration the request properties and the
// previously executed cycles along with their results.
func (h *SLOAwareProfileHandler) Pick(ctx context.Context, _ *types.CycleState, request *types.LLMRequest, profiles map[string]*framework.SchedulerProfile,
	profileResults map[string]*types.ProfileRunResult) map[string]*framework.SchedulerProfile {

	predictorBasedScheduling := !isHeaderPresent(*request, PreictionBasedSchedulingHeaderKey)

	_, prefixExecuted := profileResults[PrefixProfileName]
	// if prefix profile was not executed yet, first let the scheduler run it
	if !prefixExecuted {
		return map[string]*framework.SchedulerProfile{
			PrefixProfileName: profiles[PrefixProfileName],
		}
	}

	if predictorBasedScheduling {
		_, routingExecuted := profileResults[LatencyRoutingProfileName]
		// routing profile has not been executed yet
		if !routingExecuted {
			return map[string]*framework.SchedulerProfile{
				LatencyRoutingProfileName: profiles[LatencyRoutingProfileName],
			}
		}
	} else {
		_, defaultExecuted := profileResults[NoLatencyRoutingProfileName]
		// predictorBasedScheduling is off, and NoLatencyRoutingProfileName profile has not been executed yet
		if !defaultExecuted {
			return map[string]*framework.SchedulerProfile{
				NoLatencyRoutingProfileName: profiles[NoLatencyRoutingProfileName],
			}
		}
	}

	// all previous profiles have been executed, nothing more to run
	return map[string]*framework.SchedulerProfile{}
}

// ProcessResults handles the outcome of the profile runs after all profiles ran.
// It may aggregate results, log test profile outputs, or apply custom logic. It specifies in the SchedulingResult the
// key of the primary profile that should be used to get the request selected destination.
// When a profile run fails, its result in the profileResults map is nil.
func (h *SLOAwareProfileHandler) ProcessResults(ctx context.Context, _ *types.CycleState, request *types.LLMRequest, profileResults map[string]*types.ProfileRunResult) (*types.SchedulingResult, error) {

	predictorBasedScheduling := !isHeaderPresent(*request, PreictionBasedSchedulingHeaderKey)

	if predictorBasedScheduling { // TODO grab header directly from request.Headers instead of request field
		if len(profileResults) < 2 {
			return nil, errors.New("SLOAwareProfileHandler requires at least two profiles to operate when predictorBasedScheduling is true")
		}
		if profileResults[LatencyRoutingProfileName] == nil { // there was an error while running the SLO profile
			return nil, fmt.Errorf("failed to run scheduler profile '%s'", LatencyRoutingProfileName)
		}
		return &types.SchedulingResult{
			ProfileResults:     profileResults,
			PrimaryProfileName: LatencyRoutingProfileName,
		}, nil
	}
	if len(profileResults) < 1 {
		return nil, errors.New("SLOAwareProfileHandler requires at least one profiles to operate when predictorBasedScheduling is false")
	}

	if profileResults[NoLatencyRoutingProfileName] == nil { // there was an error while running the default profile
		return nil, fmt.Errorf("failed to run scheduler profile '%s'", NoLatencyRoutingProfileName)
	}

	return &types.SchedulingResult{
		ProfileResults:     profileResults,
		PrimaryProfileName: NoLatencyRoutingProfileName,
	}, nil
}

// isHeaderPresent checks if a header key exists in the request headers map.
func isHeaderPresent(request types.LLMRequest, headerName string) bool {
	// 1. Get header value from the map
	_, ok := request.Headers[headerName]
	return ok
}
