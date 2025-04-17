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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

var (
	lowLatencyFilter = &plugins.DecisionTreeFilter{
		Current: plugins.LowQueueFilter,
		NextOnSuccess: &plugins.DecisionTreeFilter{
			Current: plugins.LoRAAffinityFilter,
			NextOnSuccessOrFailure: &plugins.DecisionTreeFilter{
				Current: plugins.LeastQueueFilter,
				NextOnSuccessOrFailure: &plugins.DecisionTreeFilter{
					Current: plugins.LeastKVCacheFilter,
				},
			},
		},
		NextOnFailure: &plugins.DecisionTreeFilter{
			Current: plugins.LeastQueueFilter,
			NextOnSuccessOrFailure: &plugins.DecisionTreeFilter{
				Current: plugins.LoRAAffinityFilter,
				NextOnSuccessOrFailure: &plugins.DecisionTreeFilter{
					Current: plugins.LeastKVCacheFilter,
				},
			},
		},
	}

	sheddableRequestFilter = &plugins.DecisionTreeFilter{
		// When there is at least one model server that's not queuing requests, and still has KV
		// cache below a certain threshold, we consider this model server has capacity to handle
		// a sheddable request without impacting critical requests.
		Current:       plugins.HasCapacityFilter,
		NextOnSuccess: lowLatencyFilter,
		// If all pods are queuing or running above the KVCache threshold, we drop the sheddable
		// request to make room for critical requests.
		NextOnFailure: plugins.DropRequestFilter,
	}
)

func NewScheduler(datastore Datastore) *Scheduler {
	defaultPlugin := &defaultPlugin{}

	return &Scheduler{
		datastore:           datastore,
		preSchedulePlugins:  []types.PreSchedule{},
		postSchedulePlugins: []types.PostSchedule{},
		scorers:             []types.Scorer{},
		filters:             []types.Filter{defaultPlugin},
		picker:              defaultPlugin,
	}
}

type Scheduler struct {
	datastore           Datastore
	preSchedulePlugins  []types.PreSchedule
	postSchedulePlugins []types.PostSchedule
	filters             []types.Filter
	scorers             []types.Scorer
	picker              types.Picker
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
	sCtx := types.NewContext(ctx, req, types.ToSchedulerPodMetrics(s.datastore.PodGetAll()))
	loggerDebug.Info(fmt.Sprintf("Scheduling a request. Metrics: %+v", sCtx.PodsSnapshot))

	s.runPreSchedulePlugins(sCtx)

	pods, err := s.runFilterPlugins(sCtx)
	if err != nil {
		return nil, err
	}

	if err := s.runScorerPlugins(sCtx, pods); err != nil {
		return nil, err
	}

	before := time.Now()
	res, err := s.picker.Pick(sCtx, pods)
	metrics.RecordSchedulerPluginProcessingLatency(types.PickerPluginType, s.picker.Name(), time.Since(before))
	if err != nil {
		return nil, err
	}
	loggerDebug.Info("After running picker plugins", "result", res)

	s.runPostSchedulePlugins(sCtx, res)

	return res, nil
}

func (s *Scheduler) runPreSchedulePlugins(ctx *types.Context) {
	for _, plugin := range s.preSchedulePlugins {
		ctx.Logger.V(logutil.DEBUG).Info("Running pre-schedule plugin", "plugin", plugin.Name())
		before := time.Now()
		plugin.PreSchedule(ctx)
		metrics.RecordSchedulerPluginProcessingLatency(types.PreSchedulerPluginType, plugin.Name(), time.Since(before))
	}
}

func (s *Scheduler) runPostSchedulePlugins(ctx *types.Context, res *types.Result) {
	for _, plugin := range s.postSchedulePlugins {
		ctx.Logger.V(logutil.DEBUG).Info("Running post-schedule plugin", "plugin", plugin.Name())
		before := time.Now()
		plugin.PostSchedule(ctx, res)
		metrics.RecordSchedulerPluginProcessingLatency(types.PostSchedulePluginType, plugin.Name(), time.Since(before))
	}
}

func (s *Scheduler) runFilterPlugins(ctx *types.Context) ([]types.Pod, error) {
	loggerDebug := ctx.Logger.V(logutil.DEBUG)
	pods := ctx.PodsSnapshot
	loggerDebug.Info("Before running filter plugins", "pods", pods)
	for _, filter := range s.filters {
		loggerDebug.Info("Running filter plugin", "plugin", filter.Name())
		before := time.Now()
		filteredPods, err := filter.Filter(ctx, pods)
		metrics.RecordSchedulerPluginProcessingLatency(types.FilterPluginType, filter.Name(), time.Since(before))
		if err != nil || len(filteredPods) == 0 {
			return nil, fmt.Errorf("failed to apply filter, resulted %v pods, this should never happen: %w", len(filteredPods), err)
		}
		pods = filteredPods
		loggerDebug.Info("Filter plugin result", "plugin", filter.Name(), "pods", pods)
	}
	loggerDebug.Info("After running filter plugins", "pods", pods)
	return pods, nil
}

func (s *Scheduler) runScorerPlugins(ctx *types.Context, pods []types.Pod) error {
	loggerDebug := ctx.Logger.V(logutil.DEBUG)
	loggerDebug.Info("Before running score plugins", "pods", pods)
	for _, pod := range pods {
		score, err := runScorersForPod(ctx, s.scorers, pod)
		if err != nil {
			return err
		}
		pod.SetScore(score)
	}
	loggerDebug.Info("After running score plugins", "pods", pods)
	return nil
}

// Iterate through each scorer in the chain and accumulate the scores.
func runScorersForPod(ctx *types.Context, scorers []types.Scorer, pod types.Pod) (float64, error) {
	logger := ctx.Logger.WithValues("pod", pod.GetPod().NamespacedName).V(logutil.DEBUG)
	score := float64(0)
	for _, scorer := range scorers {
		logger.Info("Running scorer", "scorer", scorer.Name())
		before := time.Now()
		oneScore, err := scorer.Score(ctx, pod)
		metrics.RecordSchedulerPluginProcessingLatency(types.ScorerPluginType, scorer.Name(), time.Since(before))
		if err != nil {
			logger.Error(err, "Failed to calculate score for scorer", "scorer", scorer.Name())
			return 0, err
		}
		score += oneScore
		logger.Info("After scorer", "scorer", scorer.Name(), "score", oneScore, "total score", score)
	}
	return score, nil
}

type defaultPlugin struct {
	plugins.RandomPicker
}

func (p *defaultPlugin) Name() string {
	return "DefaultPlugin"
}

func (p *defaultPlugin) Filter(ctx *types.Context, pods []types.Pod) ([]types.Pod, error) {
	req := ctx.Req
	var filter types.Filter
	if req.Critical {
		filter = lowLatencyFilter
	} else {
		filter = sheddableRequestFilter
	}
	return filter.Filter(ctx, pods)
}
