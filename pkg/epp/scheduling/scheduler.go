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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/picker"
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

	defaultConfig := &SchedulerConfig{
		preSchedulePlugins:  []plugins.PreSchedule{},
		filters:             []plugins.Filter{filter.NewSheddableCapacityFilter(), lowLatencyFilter},
		scorers:             map[plugins.Scorer]int{},
		picker:              &picker.RandomPicker{},
		postSchedulePlugins: []plugins.PostSchedule{},
	}

	return NewSchedulerWithConfig(datastore, defaultConfig)
}

// NewSchedulerWithConfig returns a new scheduler with the given scheduler plugins configuration.
func NewSchedulerWithConfig(datastore Datastore, config *SchedulerConfig) *Scheduler {
	return &Scheduler{
		datastore:           datastore,
		preSchedulePlugins:  config.preSchedulePlugins,
		filters:             config.filters,
		scorers:             config.scorers,
		picker:              config.picker,
		postSchedulePlugins: config.postSchedulePlugins,
		postResponsePlugins: config.postResponsePlugins,
	}
}

type Scheduler struct {
	datastore           Datastore
	preSchedulePlugins  []plugins.PreSchedule
	filters             []plugins.Filter
	scorers             map[plugins.Scorer]int // map from scorer to its weight
	picker              plugins.Picker
	postSchedulePlugins []plugins.PostSchedule
	postResponsePlugins []plugins.PostResponse
}

type Datastore interface {
	PodGetAll() []backendmetrics.PodMetrics
}

// Schedule finds the target pod based on metrics and the requested lora adapter.
func (s *Scheduler) Schedule(ctx context.Context, req *types.LLMRequest) (*types.Result, error) {
	logger := log.FromContext(ctx).WithValues("request", req)
	loggerDebug := logger.V(logutil.DEBUG)

	scheduleStart := time.Now()
	defer func() {
		metrics.RecordSchedulerE2ELatency(time.Since(scheduleStart))
	}()

	// Snapshot pod metrics from the datastore to:
	// 1. Reduce concurrent access to the datastore.
	// 2. Ensure consistent data during the scheduling operation of a request.
	sCtx := types.NewSchedulingContext(ctx, req, nil, types.ToSchedulerPodMetrics(s.datastore.PodGetAll()))
	loggerDebug.Info(fmt.Sprintf("Scheduling a request, Metrics: %+v", sCtx.PodsSnapshot))

	s.runPreSchedulePlugins(sCtx)

	pods := s.runFilterPlugins(sCtx)
	if len(pods) == 0 {
		return nil, errutil.Error{Code: errutil.Internal, Msg: "no pods available for the given request"}
	}
	// if we got here, there is at least one pod to score
	weightedScorePerPod := s.runScorerPlugins(sCtx, pods)

	result := s.runPickerPlugin(sCtx, weightedScorePerPod)

	s.runPostSchedulePlugins(sCtx, result)

	return result, nil
}

func (s *Scheduler) runPreSchedulePlugins(ctx *types.SchedulingContext) {
	for _, plugin := range s.preSchedulePlugins {
		ctx.Logger.V(logutil.DEBUG).Info("Running pre-schedule plugin", "plugin", plugin.Name())
		before := time.Now()
		plugin.PreSchedule(ctx)
		metrics.RecordSchedulerPluginProcessingLatency(plugins.PreSchedulerPluginType, plugin.Name(), time.Since(before))
	}
}

func (s *Scheduler) runFilterPlugins(ctx *types.SchedulingContext) []types.Pod {
	loggerDebug := ctx.Logger.V(logutil.DEBUG)
	filteredPods := ctx.PodsSnapshot
	loggerDebug.Info("Before running filter plugins", "pods", filteredPods)

	for _, filter := range s.filters {
		loggerDebug.Info("Running filter plugin", "plugin", filter.Name())
		before := time.Now()
		filteredPods = filter.Filter(ctx, filteredPods)
		metrics.RecordSchedulerPluginProcessingLatency(plugins.FilterPluginType, filter.Name(), time.Since(before))
		loggerDebug.Info("Filter plugin result", "plugin", filter.Name(), "pods", filteredPods)
		if len(filteredPods) == 0 {
			break
		}
	}
	loggerDebug.Info("After running filter plugins")

	return filteredPods
}

func (s *Scheduler) runScorerPlugins(ctx *types.SchedulingContext, pods []types.Pod) map[types.Pod]float64 {
	loggerDebug := ctx.Logger.V(logutil.DEBUG)
	loggerDebug.Info("Before running scorer plugins", "pods", pods)

	weightedScorePerPod := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		weightedScorePerPod[pod] = float64(0) // initialize weighted score per pod with 0 value
	}
	// Iterate through each scorer in the chain and accumulate the weighted scores.
	for scorer, weight := range s.scorers {
		loggerDebug.Info("Running scorer", "scorer", scorer.Name())
		before := time.Now()
		scores := scorer.Score(ctx, pods)
		metrics.RecordSchedulerPluginProcessingLatency(plugins.ScorerPluginType, scorer.Name(), time.Since(before))
		for pod, score := range scores { // weight is relative to the sum of weights
			weightedScorePerPod[pod] += score * float64(weight) // TODO normalize score before multiply with weight
		}
		loggerDebug.Info("After running scorer", "scorer", scorer.Name())
	}
	loggerDebug.Info("After running scorer plugins")

	return weightedScorePerPod
}

func (s *Scheduler) runPickerPlugin(ctx *types.SchedulingContext, weightedScorePerPod map[types.Pod]float64) *types.Result {
	loggerDebug := ctx.Logger.V(logutil.DEBUG)
	scoredPods := make([]*types.ScoredPod, len(weightedScorePerPod))
	i := 0
	for pod, score := range weightedScorePerPod {
		scoredPods[i] = &types.ScoredPod{Pod: pod, Score: score}
		i++
	}

	loggerDebug.Info("Before running picker plugin", "pods weighted score", fmt.Sprint(weightedScorePerPod))
	before := time.Now()
	result := s.picker.Pick(ctx, scoredPods)
	metrics.RecordSchedulerPluginProcessingLatency(plugins.PickerPluginType, s.picker.Name(), time.Since(before))
	loggerDebug.Info("After running picker plugin", "result", result)

	return result
}

func (s *Scheduler) runPostSchedulePlugins(ctx *types.SchedulingContext, res *types.Result) {
	for _, plugin := range s.postSchedulePlugins {
		ctx.Logger.V(logutil.DEBUG).Info("Running post-schedule plugin", "plugin", plugin.Name())
		before := time.Now()
		plugin.PostSchedule(ctx, res)
		metrics.RecordSchedulerPluginProcessingLatency(plugins.PostSchedulePluginType, plugin.Name(), time.Since(before))
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

	s.runPostResponsePlugins(sCtx, targetPod)
}

func (s *Scheduler) runPostResponsePlugins(ctx *types.SchedulingContext, targetPod types.Pod) {
	for _, plugin := range s.postResponsePlugins {
		ctx.Logger.V(logutil.DEBUG).Info("Running post-response plugin", "plugin", plugin.Name())
		before := time.Now()
		plugin.PostResponse(ctx, targetPod)
		metrics.RecordSchedulerPluginProcessingLatency(plugins.PostResponsePluginType, plugin.Name(), time.Since(before))
	}
}
