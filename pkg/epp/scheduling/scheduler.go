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

// Package scheduling implements request scheduling algorithms.
package scheduling

import (
	"context"
	"fmt"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

type Datastore interface {
	PodGetAll() []backendmetrics.PodMetrics
}

// NewScheduler returns a new scheduler with default scheduler plugins configuration.
func NewScheduler() *Scheduler {
	// When the scheduler is initialized with NewScheduler function, thw below config will be used as default.
	// it's possible to call NewSchedulerWithConfig to pass a different scheduler config.
	// For build time plugins changes, it's recommended to call in main.go to NewSchedulerWithConfig.
	loraAffinityFilter := filter.NewLoraAffinityFilter(config.Conf.LoraAffinityThreshold)
	leastQueueFilter := filter.NewLeastQueueFilter()
	leastKvCacheFilter := filter.NewLeastKVCacheFilter()

	lowLatencyFilter := &filter.DecisionTreeFilter{
		Current: filter.NewLowQueueFilter(config.Conf.QueueingThresholdLoRA),
		NextOnSuccess: &filter.DecisionTreeFilter{
			Current: loraAffinityFilter,
			NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
				Current: leastQueueFilter,
				NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
					Current: leastKvCacheFilter,
				},
			},
		},
		NextOnFailure: &filter.DecisionTreeFilter{
			Current: leastQueueFilter,
			NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
				Current: loraAffinityFilter,
				NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
					Current: leastKvCacheFilter,
				},
			},
		},
	}

	defaultProfile := framework.NewSchedulerProfile().
		WithFilters(lowLatencyFilter).
		WithPicker(&picker.RandomPicker{})

	profileHandler := profile.NewSingleProfileHandler()

	return NewSchedulerWithConfig(NewSchedulerConfig(profileHandler, map[string]*framework.SchedulerProfile{"default": defaultProfile}))
}

// NewSchedulerWithConfig returns a new scheduler with the given scheduler plugins configuration.
func NewSchedulerWithConfig(config *SchedulerConfig) *Scheduler {
	return &Scheduler{
		profileHandler: config.profileHandler,
		profiles:       config.profiles,
	}
}

type Scheduler struct {
	profileHandler framework.ProfileHandler
	profiles       map[string]*framework.SchedulerProfile
}

// Schedule finds the target pod based on metrics and the requested lora adapter.
func (s *Scheduler) Schedule(ctx context.Context, request *types.LLMRequest, candidatePods []types.Pod) (*types.SchedulingResult, error) {
	logger := log.FromContext(ctx).WithValues("request", request)
	loggerDebug := logger.V(logutil.DEBUG)

	scheduleStart := time.Now()
	defer func() {
		metrics.RecordSchedulerE2ELatency(time.Since(scheduleStart))
	}()

	profileRunResults := map[string]*types.ProfileRunResult{}
	cycleState := types.NewCycleState()

	for { // get the next set of profiles to run iteratively based on the request and the previous execution results
		before := time.Now()
		profiles := s.profileHandler.Pick(ctx, cycleState, request, s.profiles, profileRunResults)
		metrics.RecordSchedulerPluginProcessingLatency(framework.ProfilePickerType, s.profileHandler.GetTypedName().Type, time.Since(before))
		if len(profiles) == 0 { // profile picker didn't pick any profile to run
			break
		}

		for name, profile := range profiles {
			// run the selected profiles and collect results (current code runs all profiles)
			profileRunResult, err := profile.Run(ctx, request, cycleState, candidatePods)
			if err != nil {
				loggerDebug.Info("failed to run scheduler profile", "profile", name, "error", err.Error())
			}

			profileRunResults[name] = profileRunResult // if profile failed to run, the run result is nil
		}
	}

	if len(profileRunResults) == 0 {
		return nil, fmt.Errorf("failed to run any SchedulingProfile for the request - %s", request)
	}

	before := time.Now()
	result, err := s.profileHandler.ProcessResults(ctx, cycleState, request, profileRunResults)
	metrics.RecordSchedulerPluginProcessingLatency(framework.ProcessProfilesResultsType, s.profileHandler.GetTypedName().Type, time.Since(before))

	return result, err
}
