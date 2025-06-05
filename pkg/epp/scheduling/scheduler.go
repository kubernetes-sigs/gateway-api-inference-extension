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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	profilepicker "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile-picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// NewScheduler returns a new scheduler with default scheduler plugins configuration.
func NewScheduler(datastore Datastore) *Scheduler {
	// When the scheduler is initialized with NewScheduler function, thw below config will be used as default.
	// it's possible to call NewSchedulerWithConfig to pass a different scheduler config.
	// For build time plugins changes, it's recommended to call in main.go to NewSchedulerWithConfig.
	loraAffinityFilter := filter.NewLoraAffinityFilter()
	leastQueueFilter := filter.NewLeastQueueFilter()
	leastKvCacheFilter := filter.NewLeastKVCacheFilter()

	lowLatencyFilter := &filter.DecisionTreeFilter{
		Current: filter.NewLowQueueFilter(),
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

	profilePicker := profilepicker.NewAllProfilesPicker()

	return NewSchedulerWithConfig(datastore, NewSchedulerConfig(profilePicker, map[string]*framework.SchedulerProfile{"default": defaultProfile}))
}

// NewSchedulerWithConfig returns a new scheduler with the given scheduler plugins configuration.
func NewSchedulerWithConfig(datastore Datastore, config *SchedulerConfig) *Scheduler {
	return &Scheduler{
		datastore:     datastore,
		profilePicker: config.profilePicker,
		profiles:      config.profiles,
	}
}

type Scheduler struct {
	datastore     Datastore
	profilePicker framework.ProfilePicker
	profiles      map[string]*framework.SchedulerProfile
}

type Datastore interface {
	PodGetAll() []backendmetrics.PodMetrics
}

// Schedule finds the target pod based on metrics and the requested lora adapter.
func (s *Scheduler) Schedule(ctx context.Context, request *types.LLMRequest) (map[string]*types.Result, error) {
	logger := log.FromContext(ctx).WithValues("request", request)
	loggerDebug := logger.V(logutil.DEBUG)

	scheduleStart := time.Now()
	defer func() {
		metrics.RecordSchedulerE2ELatency(time.Since(scheduleStart))
	}()

	// Snapshot pod metrics from the datastore to:
	// 1. Reduce concurrent access to the datastore.
	// 2. Ensure consistent data during the scheduling operation of a request between all scheduling cycles.
	podsSnapshot := types.ToSchedulerPodMetrics(s.datastore.PodGetAll())
	loggerDebug.Info(fmt.Sprintf("Scheduling a request, Metrics: %+v", podsSnapshot))

	profileExecutionResults := map[string]*types.Result{}

	for { // get the next set of profiles to run iteratively based on the request and the previous execution results
		before := time.Now()
		profiles := s.profilePicker.Pick(ctx, request, s.profiles, profileExecutionResults)
		metrics.RecordSchedulerPluginProcessingLatency(framework.ProfilePickerType, s.profilePicker.Name(), time.Since(before))
		if len(profiles) == 0 { // profile picker didn't pick any profile to run
			break
		}

		for name, profile := range profiles {
			// run the selected profiles and collect results (current code runs all profiles)
			profileExecutionResult, err := profile.Run(ctx, request, types.NewCycleState(), podsSnapshot)
			if err != nil {
				return nil, fmt.Errorf("failed to run all required scheduling profiles - %w", err)
			}

			profileExecutionResults[name] = profileExecutionResult
		}
	}

	if len(profileExecutionResults) == 0 {
		return nil, fmt.Errorf("failed to run any SchedulingProfile for the request - %s", request)
	}

	return profileExecutionResults, nil
}
