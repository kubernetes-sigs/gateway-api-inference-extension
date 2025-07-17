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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

type Datastore interface {
	PodGetAll() []backendmetrics.PodMetrics
}

// NewSchedulerWithConfig returns a new scheduler with the given scheduler plugins configuration.
func NewSchedulerWithConfig(config *SchedulerConfig) *Scheduler {
	scheduler := &Scheduler{
		profileHandler: config.profileHandler,
		profiles:       config.profiles,
	}

	// Log scheduler configuration at startup
	scheduler.LogConfiguration()

	return scheduler
}

type Scheduler struct {
	profileHandler framework.ProfileHandler
	profiles       map[string]*framework.SchedulerProfile
}

// LogConfiguration logs the overall scheduler configuration at startup.
func (s *Scheduler) LogConfiguration() {
	logger := log.Log.WithName("scheduler").V(logutil.DEFAULT)

	logger.Info("Scheduler initialized",
		"profileHandler", s.profileHandler.TypedName().Type,
		"numProfiles", len(s.profiles))

	// Log available profiles
	var profileNames []string
	for name := range s.profiles {
		profileNames = append(profileNames, name)
	}
	logger.Info("Available scheduler profiles", "profiles", profileNames)
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
		loggerDebug.Info("Running profile handler, Pick profiles", "plugin", s.profileHandler.TypedName().Type)
		before := time.Now()
		profiles := s.profileHandler.Pick(ctx, cycleState, request, s.profiles, profileRunResults)
		metrics.RecordSchedulerPluginProcessingLatency(framework.ProfilePickerType, s.profileHandler.TypedName().Type, time.Since(before))

		// Log which profiles were selected
		var selectedProfileNames []string
		for name := range profiles {
			selectedProfileNames = append(selectedProfileNames, name)
		}
		loggerDebug.Info("Profile handler selected profiles",
			"plugin", s.profileHandler.TypedName().Type,
			"selectedProfiles", selectedProfileNames,
			"totalAvailableProfiles", len(s.profiles))

		if len(profiles) == 0 { // profile picker didn't pick any profile to run
			loggerDebug.Info("No profiles selected, ending scheduling cycle")
			break
		}

		for name, profile := range profiles {
			loggerDebug.Info("Running scheduler profile", "name", name, "candidatePods", len(candidatePods))
			// run the selected profiles and collect results (current code runs all profiles)
			profileRunResult, err := profile.Run(ctx, request, cycleState, candidatePods)
			if err != nil {
				loggerDebug.Info("Failed to run scheduler profile", "profile", name, "error", err.Error())
			} else {
				// Log successful profile execution with results
				var selectedPods []string
				if profileRunResult != nil && profileRunResult.TargetPods != nil {
					for _, pod := range profileRunResult.TargetPods {
						selectedPods = append(selectedPods, pod.GetPod().NamespacedName.String())
					}
				}
				loggerDebug.Info("Successfully ran scheduler profile",
					"profile", name,
					"selectedPods", selectedPods,
					"numSelectedPods", len(selectedPods))
			}

			profileRunResults[name] = profileRunResult // if profile failed to run, the run result is nil
		}
	}

	if len(profileRunResults) == 0 {
		return nil, fmt.Errorf("failed to run any scheduler profile for the request - %s", request)
	}

	loggerDebug.Info("Running profile handler, ProcessResults", "plugin", s.profileHandler.TypedName().Type)
	before := time.Now()
	result, err := s.profileHandler.ProcessResults(ctx, cycleState, request, profileRunResults)
	metrics.RecordSchedulerPluginProcessingLatency(framework.ProcessProfilesResultsType, s.profileHandler.TypedName().Type, time.Since(before))
	loggerDebug.Info("After running profile handler ProcessResults", "plugin", s.profileHandler.TypedName().Type)

	return result, err
}
