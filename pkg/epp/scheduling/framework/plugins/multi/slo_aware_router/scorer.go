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
	"fmt"
	"math/rand"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

type SLOAwareRouter struct {
	tn                  plugins.TypedName
	latencypredictor    latencypredictor.PredictorInterface
	runningRequestLists map[types.NamespacedName]*requestPriorityQueue
	sloContextStore     sync.Map // map[string]*SLORequestContext
	headroomStrategy    headroomStrategy
}

var _ framework.Scorer = &SLOAwareRouter{}

func NewSLOAwareRouter(latencypredictor latencypredictor.PredictorInterface, strategy headroomStrategy) *SLOAwareRouter {
	return &SLOAwareRouter{
		tn:                  plugins.TypedName{Type: SLOAwareRouterPluginType, Name: SLOAwareRouterPluginType},
		latencypredictor:    latencypredictor,
		runningRequestLists: make(map[types.NamespacedName]*requestPriorityQueue),
		sloContextStore:     sync.Map{},
		headroomStrategy:    strategy,
	}
}

func (s *SLOAwareRouter) TypedName() plugins.TypedName {
	return s.tn
}

func (s *SLOAwareRouter) WithName(name string) *SLOAwareRouter {
	s.tn.Name = name
	return s
}

func (s *SLOAwareRouter) epsilonGreedyAffinityGate(
	ctx context.Context,
	candidates []podPredictionResult,
	r *rand.Rand,
	label string, // e.g. "positive" or "negative"
	prefixStickyThreshold float64,
) ([]podPredictionResult, bool) {
	logger := log.FromContext(ctx)
	if prefixStickyThreshold <= 0 {
		// Affinity gating disabled
		logger.V(logutil.DEBUG).Info("Affinity gating disabled (threshold <= 0)", "path", label)
		return candidates, false
	}
	eligible := make([]podPredictionResult, 0, len(candidates))
	for _, p := range candidates {
		if p.PrefixCacheScore >= prefixStickyThreshold {
			eligible = append(eligible, p)
		}
	}

	// No eligible sticky pods? Explore (no gating).
	if len(eligible) == 0 {
		return candidates, false
	}

	// ε-exploration branch
	if r.Float64() < EpsilonExploreSticky {
		logger.V(logutil.DEBUG).Info("ε-greedy: exploring (ignoring affinity gate)",
			"path", label, "epsilon", EpsilonExploreSticky, "eligibleCount", len(eligible))
		return candidates, false
	}

	logger.V(logutil.DEBUG).Info("ε-greedy: exploiting (apply affinity gate)",
		"path", label, "threshold", prefixStickyThreshold, "eligibleCount", len(eligible), "total", len(candidates))
	return eligible, true
}

// scoreWithoutPredictions provides fallback scoring based only on prefix cache scores
// when latency predictions are unavailable
func (s *SLOAwareRouter) scoreWithoutPredictions(
	ctx context.Context,
	state *schedulingtypes.CycleState,
	pods []schedulingtypes.Pod,
	r *rand.Rand,
) map[schedulingtypes.Pod]float64 {
	logger := log.FromContext(ctx)
	logger.V(logutil.TRACE).Info("Using composite-only scoring without predictions")

	scores := make(map[schedulingtypes.Pod]float64, len(pods))
	for _, pod := range pods {
		scores[pod] = 0
	}

	if len(pods) == 0 {
		return scores
	}

	// Build prediction results with only prefix cache scores
	podResults := make([]podPredictionResult, 0, len(pods))
	for _, pod := range pods {
		prefixScore := s.getPrefixCacheScoreForPod(ctx, state, pod)
		podResults = append(podResults, podPredictionResult{
			Pod:              pod,
			PrefixCacheScore: prefixScore,
			IsValid:          true, // All pods are valid when we don't check predictions
		})
	}

	// Select based on composite scores (prefix cache + other non-prediction metrics)
	selectedPod := s.selectFromCompositeScores(ctx, podResults, r, headroomStrategyCompositeOnly)

	if selectedPod != nil {
		scores[selectedPod] = 1
		logger.V(logutil.TRACE).Info("Selected pod using composite-only scoring", "pod", selectedPod.GetPod().String())
	}

	return scores
}

func (s *SLOAwareRouter) Score(ctx context.Context, state *schedulingtypes.CycleState, request *schedulingtypes.LLMRequest, pods []schedulingtypes.Pod) map[schedulingtypes.Pod]float64 {
	logger := log.FromContext(ctx)
	if s.latencypredictor == nil {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: no predictor configured, returning nil scores")
		return nil
	}

	sloCtx := s.getOrMakeSLORequestContext(request)

	s.parseSLOHeaders(ctx, request, sloCtx)

	for _, pod := range pods {
		prefixCacheScore := s.getPrefixCacheScoreForPod(ctx, state, pod)
		sloCtx.prefixCacheScoresForPods[pod.GetPod().String()] = prefixCacheScore
	}

	// Check if SLOs are provided
	if !sloCtx.predictorBasedScheduling {
		logger.V(logutil.DEBUG).Info("PredictorBasedScheduling turned off, skipping prediction-based filtering")
		s.setSLOContextForRequest(request, sloCtx)
		return nil
	}

	// Initialize scores map with all pods having score 0
	scores := make(map[schedulingtypes.Pod]float64, len(pods))
	for _, pod := range pods {
		scores[pod] = 0
	}

	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)
	predictions, err := s.generatePredictions(ctx, request, sloCtx, pods)
	if err != nil {
		logger.V(logutil.DEBUG).Error(err, "SLOAwareRouter: Error generating predictions, falling back to composite-only scoring")
		// Fall back to composite-only scoring using prefix cache scores
		s.setSLOContextForRequest(request, sloCtx)
		return s.scoreWithoutPredictions(ctx, state, pods, r)
	}
	s.updateRequestContextWithPredictions(sloCtx, predictions)

	allPreds := append([]podPredictionResult(nil), predictions...)
	allPreds, sticky := s.epsilonGreedyAffinityGate(ctx, allPreds, r, "overall", AffinityGateTauGlobal)

	// Check if all pods are invalid and all have running requests. If slos are == 0 then all pods are valid
	allPodsInvalid := (sloCtx.ttftSLO > 0 && sloCtx.avgTPOTSLO > 0)
	allPodsHaveRunningRequests := true

	for _, pred := range allPreds {
		if pred.IsValid {
			allPodsInvalid = false
		}

		runningRequestCount := s.getPodRunningRequestCount(pred.Pod)
		if runningRequestCount == 0 {
			allPodsHaveRunningRequests = false
		}
	}

	// Set HasValidPod to false if all pods are invalid and all have running requests
	if allPodsInvalid && allPodsHaveRunningRequests && !sticky {
		sloCtx.hasValidPod = false
		logger.V(logutil.DEBUG).Info("All pods are invalid and have running requests, setting HasValidPod to false")
	}

	// 2) Tiered selection: positive headroom pods get 99% probability, negative get 1%
	posHeadroomPods, negHeadroomPods := s.classifyPodsByHeadroom(allPreds)

	logger.V(logutil.DEBUG).Info("Pod headroom distribution",
		"positivePods", len(posHeadroomPods),
		"negativePods", len(negHeadroomPods))

	selectedPod := s.selectPodBasedOnStrategy(ctx, r, allPreds, posHeadroomPods, negHeadroomPods)

	// Set score = 1 for selected pod, 0 for all others
	if selectedPod != nil {
		scores[selectedPod] = 1
		logger.V(logutil.DEBUG).Info("Selected pod for scheduling", "pod", selectedPod.GetPod().String())
	}

	s.setSLOContextForRequest(request, sloCtx)

	return scores
}

func (t *SLOAwareRouter) getOrMakeSLORequestContext(request *schedulingtypes.LLMRequest) *sloRequestContext {
	sloCtx, err := t.getSLOContextForRequest(request)
	if err != nil {
		sloCtx = newSLORequestContext(request)
	}
	return sloCtx
}

func (s *SLOAwareRouter) getPrefixCacheScoreForPod(ctx context.Context, cycleState *schedulingtypes.CycleState, pod schedulingtypes.Pod) float64 {
	log.FromContext(ctx).V(logutil.DEBUG).Info("Running getPrefixCacheScoreForPod, getting prefix cache score for pod", "pod", pod.GetPod().String())
	plugintype := prefix.PrefixCachePluginType
	pluginname := prefix.PrefixCachePluginType
	cycleStateKey := (plugins.TypedName{Type: plugintype, Name: pluginname}).String()
	stateData, err := cycleState.Read(plugins.StateKey(cycleStateKey))

	log.FromContext(ctx).V(logutil.DEBUG).Info("Reading prefix cache state from cycle state", "stateKey", cycleStateKey)

	if err != nil {
		// The prefix cache plugin might not be enabled, which is a valid scenario.
		log.FromContext(ctx).V(logutil.DEBUG).Info("prefix cache state not found in cycle state, returning prefix cache score of 0.0", "pod", pod.GetPod().String())
		return 0.0
	}

	prefixCacheState, ok := stateData.(*prefix.SchedulingContextState)
	if !ok {
		// This should not happen if the plugin is configured correctly.
		log.FromContext(ctx).Error(fmt.Errorf("unexpected state type: %T", stateData), "failed to read prefix cache state")
		return 0.0
	}

	total := len(prefixCacheState.PrefixHashes)
	if total == 0 {
		// if the request has no prefixes, return 0.0
		log.FromContext(ctx).V(logutil.DEBUG).Info("No prefixes found in request, returning prefix cache score of 0.0")
		return 0.0
	}

	matchLen := prefixCacheState.PrefixCacheServers[prefix.ServerID(pod.GetPod().NamespacedName)]
	log.FromContext(ctx).V(logutil.DEBUG).Info("Prefix cache score for pod", "pod", pod.GetPod().String(), "matchLen", matchLen, "totalPrefixes", total)
	return float64(matchLen) / float64(total)
}
