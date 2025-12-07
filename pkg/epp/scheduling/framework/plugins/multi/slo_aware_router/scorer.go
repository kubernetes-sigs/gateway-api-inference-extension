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
	"math/rand"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

type SLOAwareRouter struct {
	typedName           plugins.TypedName
	latencypredictor    latencypredictor.PredictorInterface
	runningRequestLists map[types.NamespacedName]*requestPriorityQueue
	sloContextStore     sync.Map // map[string]*SLORequestContext
	headroomStrategy    headroomStrategy
	config              Config
}

var _ framework.Scorer = &SLOAwareRouter{}

type Config struct {
	SamplingMean              float64 `json:"samplingMean,omitempty"`
	MaxSampledTokens          int     `json:"maxSampledTokens,omitempty"`
	SLOBufferFactor           float64 `json:"sloBufferFactor,omitempty"`
	NegHeadroomTTFTWeight     float64 `json:"negHeadroomTTFTWeight,omitempty"`
	NegHeadroomTPOTWeight     float64 `json:"negHeadroomTPOTWeight,omitempty"`
	HeadroomTTFTWeight        float64 `json:"headroomTTFTWeight,omitempty"`
	HeadroomTPOTWeight        float64 `json:"headroomTPOTWeight,omitempty"`
	HeadroomSelectionStrategy string  `json:"headroomSelectionStrategy,omitempty"`
	CompositeKVWeight         float64 `json:"compositeKVWeight,omitempty"`
	CompositeQueueWeight      float64 `json:"compositeQueueWeight,omitempty"`
	CompositePrefixWeight     float64 `json:"compositePrefixWeight,omitempty"`
	EpsilonExploreSticky      float64 `json:"epsilonExploreSticky,omitempty"`
	EpsilonExploreNeg         float64 `json:"epsilonExploreNeg,omitempty"`
	AffinityGateTau           float64 `json:"affinityGateTau,omitempty"`
	AffinityGateTauGlobal     float64 `json:"affinityGateTauGlobal,omitempty"`
	SelectionMode             string  `json:"selectionMode,omitempty"`
}

var DefaultConfig = Config{
	SamplingMean:              DefaultSamplingMean,
	MaxSampledTokens:          MaxSampledTokens,
	SLOBufferFactor:           SLOBufferFactor,
	NegHeadroomTTFTWeight:     NegHeadroomTTFTWeight,
	NegHeadroomTPOTWeight:     NegHeadroomTPOTWeight,
	HeadroomTTFTWeight:        HeadroomTTFTWeight,
	HeadroomTPOTWeight:        HeadroomTPOTWeight,
	HeadroomSelectionStrategy: string(HeadroomSelectionStrategy),
	CompositeKVWeight:         CompositeKVWeight,
	CompositeQueueWeight:      CompositeQueueWeight,
	CompositePrefixWeight:     CompositePrefixWeight,
	EpsilonExploreSticky:      EpsilonExploreSticky,
	EpsilonExploreNeg:         EpsilonExploreNeg,
	AffinityGateTau:           AffinityGateTau,
	AffinityGateTauGlobal:     AffinityGateTauGlobal,
	SelectionMode:             string(SelectionMode),
}

func SLOAwareRouterFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	parameters := DefaultConfig
	if len(rawParameters) > 0 {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config for SLOAwareRouter: %w", err)
		}
	}

	if err := parameters.validate(); err != nil {
		return nil, fmt.Errorf("invalid SLOAwareRouter config: %w", err)
	}

	predictor, err := startPredictor(handle)
	if err != nil {
		return nil, fmt.Errorf("failed to start latency predictor: %w", err)
	}

	return NewSLOAwareRouter(parameters, predictor).WithName(name), nil
}

func (c *Config) validate() error {
	var errs []error

	if c.SamplingMean <= 0 {
		errs = append(errs, fmt.Errorf("samplingMean must be > 0, got %f", c.SamplingMean))
	}

	if c.MaxSampledTokens <= 0 {
		errs = append(errs, fmt.Errorf("maxSampledTokens must be > 0, got %d", c.MaxSampledTokens))
	}

	if c.SLOBufferFactor <= 0 {
		errs = append(errs, fmt.Errorf("sloBufferFactor must be > 0, got %f", c.SLOBufferFactor))
	}

	if c.NegHeadroomTTFTWeight < 0 || c.NegHeadroomTPOTWeight < 0 ||
		c.HeadroomTTFTWeight < 0 || c.HeadroomTPOTWeight < 0 {
		errs = append(errs, errors.New("all headroom weights must be >= 0"))
	}

	if c.CompositeKVWeight < 0 || c.CompositeQueueWeight < 0 || c.CompositePrefixWeight < 0 {
		errs = append(errs, errors.New("composite weights must be >= 0"))
	}

	if c.EpsilonExploreSticky < 0 || c.EpsilonExploreSticky > 1 {
		errs = append(errs, fmt.Errorf("epsilonExploreSticky must be in [0, 1], got %f", c.EpsilonExploreSticky))
	}
	if c.EpsilonExploreNeg < 0 || c.EpsilonExploreNeg > 1 {
		errs = append(errs, fmt.Errorf("epsilonExploreNeg must be in [0, 1], got %f", c.EpsilonExploreNeg))
	}

	if c.AffinityGateTau < 0 || c.AffinityGateTau > 1 {
		errs = append(errs, fmt.Errorf("affinityGateTau must be in [0, 1], got %f", c.AffinityGateTau))
	}
	if c.AffinityGateTauGlobal <= 0 || c.AffinityGateTauGlobal > 1 {
		errs = append(errs, fmt.Errorf("affinityGateTauGlobal must be in (0, 1], got %f", c.AffinityGateTauGlobal))
	}

	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}

func NewSLOAwareRouter(config Config, predictor latencypredictor.PredictorInterface) *SLOAwareRouter {
	strategy := headroomStrategy(config.HeadroomSelectionStrategy)
	if strategy == "" {
		strategy = headroomStrategyLeast
	}

	return &SLOAwareRouter{
		typedName:           plugins.TypedName{Type: SLOAwareRouterPluginType, Name: SLOAwareRouterPluginType},
		latencypredictor:    predictor,
		runningRequestLists: make(map[types.NamespacedName]*requestPriorityQueue),
		sloContextStore:     sync.Map{},
		headroomStrategy:    strategy,
		config:              config,
	}
}

func startPredictor(handle plugins.Handle) (latencypredictor.PredictorInterface, error) {
	// Initialize the latency predictor
	predictor := latencypredictor.New(latencypredictor.ConfigFromEnv(), ctrl.Log.WithName("latency-predictor"))
	if err := predictor.Start(handle.Context()); err != nil {
		return nil, fmt.Errorf("failed to start latency predictor: %w", err)
	}

	go func() {
		<-handle.Context().Done()
		predictor.Stop()
	}()
	return predictor, nil
}

func (s *SLOAwareRouter) TypedName() plugins.TypedName {
	return s.typedName
}

func (s *SLOAwareRouter) WithName(name string) *SLOAwareRouter {
	s.typedName.Name = name
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
	predictions, err := s.generatePredictions(ctx, state, request, sloCtx, pods)
	if err != nil {
		logger.V(logutil.DEBUG).Error(err, "SLOAwareRouter: Error generating predictions, falling back to composite-only scoring")
		// Fall back to composite-only scoring using prefix cache scores
		s.setSLOContextForRequest(request, sloCtx)
		return s.scoreWithoutPredictions(ctx, state, pods, r)
	}
	s.updateRequestContextWithPredictions(sloCtx, predictions)

	allPreds := append([]podPredictionResult(nil), predictions...)
	allPreds, sticky := s.epsilonGreedyAffinityGate(ctx, allPreds, r, "overall", AffinityGateTauGlobal)

	// Check if all pods are invalid and all have running requests
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
