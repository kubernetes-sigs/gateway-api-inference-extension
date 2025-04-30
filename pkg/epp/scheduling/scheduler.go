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

var (
	lowLatencyFilter = &filter.DecisionTreeFilter{
		Current: filter.LowQueueFilter,
		NextOnSuccess: &filter.DecisionTreeFilter{
			Current: filter.LoRAAffinityFilter,
			NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
				Current: filter.LeastQueueFilter,
				NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
					Current: filter.LeastKVCacheFilter,
				},
			},
		},
		NextOnFailure: &filter.DecisionTreeFilter{
			Current: filter.LeastQueueFilter,
			NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
				Current: filter.LoRAAffinityFilter,
				NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
					Current: filter.LeastKVCacheFilter,
				},
			},
		},
	}

	sheddableRequestFilter = &filter.DecisionTreeFilter{
		// When there is at least one model server that's not queuing requests, and still has KV
		// cache below a certain threshold, we consider this model server has capacity to handle
		// a sheddable request without impacting critical requests.
		Current:       filter.HasCapacityFilter,
		NextOnSuccess: lowLatencyFilter,
		// If all pods are queuing or running above the KVCache threshold, we drop the sheddable
		// request to make room for critical requests. for this, we don't define nextOnFailure.
	}
)

func NewScheduler(datastore Datastore) *Scheduler {
	return NewSchedulerWithConfig(datastore, defaultConfig)
}

func NewSchedulerWithConfig(datastore Datastore, config *SchedulerConfig) *Scheduler {
	return &Scheduler{
		datastore:           datastore,
		preSchedulePlugins:  config.preSchedulePlugins,
		filters:             config.filters,
		scorers:             config.scorers,
		picker:              config.picker,
		postSchedulePlugins: config.postSchedulePlugins,
	}
}

type Scheduler struct {
	datastore           Datastore
	preSchedulePlugins  []plugins.PreSchedule
	filters             []plugins.Filter
	scorers             map[plugins.Scorer]int // map from scorer to its weight
	picker              plugins.Picker
	postSchedulePlugins []plugins.PostSchedule
}

type Datastore interface {
	PodGetAll() []backendmetrics.PodMetrics
}

// Schedule finds the target pod based on metrics and the requested lora adapter.
func (s *Scheduler) Schedule(ctx context.Context, req *types.LLMRequest) (*types.Result, error) {
	logger := log.FromContext(ctx).WithValues("request", req)
	loggerDebug := logger.V(logutil.DEBUG)

	// Snapshot pod metrics from the datastore to:
	// 1. Reduce concurrent access to the datastore.
	// 2. Ensure consistent data during the scheduling operation of a request.
	sCtx := types.NewSchedulingContext(ctx, req, types.ToSchedulerPodMetrics(s.datastore.PodGetAll()))
	loggerDebug.Info(fmt.Sprintf("Scheduling a request, Metrics: %+v", sCtx.PodsSnapshot))

	s.runPreSchedulePlugins(sCtx)

	pods := s.runFilterPlugins(sCtx)
	if len(pods) == 0 {
		return nil, errutil.Error{Code: errutil.InferencePoolResourceExhausted, Msg: "failed to find a target pod"}
	}
	// if we got here, there is at least one pod to score
	weightedScorePerPod := s.runScorerPlugins(sCtx, pods)

	result := s.runPickerPlugin(sCtx, weightedScorePerPod)

	s.runPostSchedulePlugins(sCtx, result)

	result.MutatedHeaders = sCtx.MutatedHeaders
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

	loggerDebug.Info("Before running picker plugin", "pods", weightedScorePerPod)
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

type defaultPlugin struct {
	picker.RandomPicker
}

func (p *defaultPlugin) Name() string {
	return "DefaultPlugin"
}

func (p *defaultPlugin) Filter(ctx *types.SchedulingContext, pods []types.Pod) []types.Pod {
	if ctx.Req.Critical {
		return lowLatencyFilter.Filter(ctx, pods)
	}

	return sheddableRequestFilter.Filter(ctx, pods)
}
