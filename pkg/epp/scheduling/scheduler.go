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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/picker"
	profilepicker "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/profile-picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
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
		WithFilters(filter.NewSheddableCapacityFilter(), lowLatencyFilter).
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
func (s *Scheduler) Schedule(ctx context.Context, req *types.LLMRequest) ([]*types.Result, error) {
	logger := log.FromContext(ctx).WithValues("request", req)
	loggerDebug := logger.V(logutil.DEBUG)

	scheduleStart := time.Now()
	defer func() {
		metrics.RecordSchedulerE2ELatency(time.Since(scheduleStart))
	}()

	profiles := s.profilePicker.Pick(req, s.profiles)
	// Snapshot pod metrics from the datastore to:
	// 1. Reduce concurrent access to the datastore.
	// 2. Ensure consistent data during the scheduling operation of a request between all scheduling cycles.
	sCtx := types.NewSchedulingContext(ctx, req, nil, types.ToSchedulerPodMetrics(s.datastore.PodGetAll()))
	loggerDebug.Info(fmt.Sprintf("Scheduling a request, Metrics: %+v", sCtx.PodsSnapshot))

	result := []*types.Result{}
	for name, profile := range profiles {
		// run the selected profiles and collect results (current code runs all profiles)
		cycleResult, err := s.runSchedulerProfileCycle(sCtx, profile)
		if err != nil {
			return result, fmt.Errorf("failed to run all required scheduling profiles - %w", err)
		}
		cycleResult.ProfileName = name
		result = append(result, cycleResult)
	}

	return result, nil
}

func (s *Scheduler) runSchedulerProfileCycle(ctx *types.SchedulingContext, profile *framework.SchedulerProfile) (*types.Result, error) {
	s.runPreCyclePlugins(ctx, profile.PreCyclePlugins())

	pods := s.runFilterPlugins(ctx, profile.Filters())
	if len(pods) == 0 {
		return nil, errutil.Error{Code: errutil.Internal, Msg: "no pods available for the given request"}
	}
	// if we got here, there is at least one pod to score
	weightedScorePerPod := s.runScorerPlugins(ctx, pods, profile.Scorers())

	result := s.runPickerPlugin(ctx, weightedScorePerPod, profile.Picker())

	s.runPostCyclePlugins(ctx, result, profile.PostCyclePlugins())

	return result, nil
}

func (s *Scheduler) runPreCyclePlugins(ctx *types.SchedulingContext, plugins []framework.PreCycle) {
	for _, plugin := range plugins {
		ctx.Logger.V(logutil.DEBUG).Info("Running pre-cycle plugin", "plugin", plugin.Name())
		before := time.Now()
		plugin.PreCycle(ctx)
		metrics.RecordSchedulerPluginProcessingLatency(framework.PreCyclePluginType, plugin.Name(), time.Since(before))
	}
}

func (s *Scheduler) runFilterPlugins(ctx *types.SchedulingContext, filters []framework.Filter) []types.Pod {
	loggerDebug := ctx.Logger.V(logutil.DEBUG)
	filteredPods := ctx.PodsSnapshot
	loggerDebug.Info("Before running filter plugins", "pods", filteredPods)

	for _, filter := range filters {
		loggerDebug.Info("Running filter plugin", "plugin", filter.Name())
		before := time.Now()
		filteredPods = filter.Filter(ctx, filteredPods)
		metrics.RecordSchedulerPluginProcessingLatency(framework.FilterPluginType, filter.Name(), time.Since(before))
		loggerDebug.Info("Filter plugin result", "plugin", filter.Name(), "pods", filteredPods)
		if len(filteredPods) == 0 {
			break
		}
	}
	loggerDebug.Info("After running filter plugins")

	return filteredPods
}

func (s *Scheduler) runScorerPlugins(ctx *types.SchedulingContext, pods []types.Pod, scorers []*framework.WeightedScorer) map[types.Pod]float64 {
	loggerDebug := ctx.Logger.V(logutil.DEBUG)
	loggerDebug.Info("Before running scorer plugins", "pods", pods)

	weightedScorePerPod := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		weightedScorePerPod[pod] = float64(0) // initialize weighted score per pod with 0 value
	}
	// Iterate through each scorer in the chain and accumulate the weighted scores.
	for _, scorer := range scorers {
		loggerDebug.Info("Running scorer", "scorer", scorer.Name())
		before := time.Now()
		scores := scorer.Score(ctx, pods)
		metrics.RecordSchedulerPluginProcessingLatency(framework.ScorerPluginType, scorer.Name(), time.Since(before))
		for pod, score := range scores { // weight is relative to the sum of weights
			weightedScorePerPod[pod] += score * float64(scorer.Weight())
		}
		loggerDebug.Info("After running scorer", "scorer", scorer.Name())
	}
	loggerDebug.Info("After running scorer plugins")

	return weightedScorePerPod
}

func (s *Scheduler) runPickerPlugin(ctx *types.SchedulingContext, weightedScorePerPod map[types.Pod]float64, picker framework.Picker) *types.Result {
	loggerDebug := ctx.Logger.V(logutil.DEBUG)
	scoredPods := make([]*types.ScoredPod, len(weightedScorePerPod))
	i := 0
	for pod, score := range weightedScorePerPod {
		scoredPods[i] = &types.ScoredPod{Pod: pod, Score: score}
		i++
	}

	loggerDebug.Info("Before running picker plugin", "pods weighted score", fmt.Sprint(weightedScorePerPod))
	before := time.Now()
	result := picker.Pick(ctx, scoredPods)
	metrics.RecordSchedulerPluginProcessingLatency(framework.PickerPluginType, picker.Name(), time.Since(before))
	loggerDebug.Info("After running picker plugin", "result", result)

	return result
}

func (s *Scheduler) runPostCyclePlugins(ctx *types.SchedulingContext, res *types.Result, plugins []framework.PostCycle) {
	for _, plugin := range plugins {
		ctx.Logger.V(logutil.DEBUG).Info("Running post-cycle plugin", "plugin", plugin.Name())
		before := time.Now()
		plugin.PostCycle(ctx, res)
		metrics.RecordSchedulerPluginProcessingLatency(framework.PostCyclePluginType, plugin.Name(), time.Since(before))
	}
}

// OnResponse is invoked during the processing of a response from an inference pod. It will invoke
// any defined plugins that process the response.
func (s *Scheduler) OnResponse(ctx context.Context, resp *types.LLMResponse, targetPodName string) {
	// Snapshot pod metrics from the datastore to:
	// 1. Reduce concurrent access to the datastore.
	// 2. Ensure consistent data during the scheduling operation of a request.
	pods := types.ToSchedulerPodMetrics(s.datastore.PodGetAll())
	var targetPod types.Pod
	for _, pod := range pods {
		if pod.GetPod().NamespacedName.String() == targetPodName {
			targetPod = pod
			break
		}
	}

	sCtx := types.NewSchedulingContext(ctx, nil, resp, pods)

	// WORKAROUND until PostResponse is out of Scheduler
	profiles := s.profilePicker.Pick(nil, s.profiles) // all profiles
	for _, profile := range profiles {
		s.runPostResponsePlugins(sCtx, targetPod, profile)
	}
}

func (s *Scheduler) runPostResponsePlugins(ctx *types.SchedulingContext, targetPod types.Pod, profile *framework.SchedulerProfile) {
	for _, plugin := range profile.PostResponsePlugins {
		ctx.Logger.V(logutil.DEBUG).Info("Running post-response plugin", "plugin", plugin.Name())
		before := time.Now()
		plugin.PostResponse(ctx, targetPod)
		metrics.RecordSchedulerPluginProcessingLatency(framework.PostResponsePluginType, plugin.Name(), time.Since(before))
	}
}
