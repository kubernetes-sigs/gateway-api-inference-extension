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

package predictedlatency

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/jellydator/ttlcache/v3"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

type PredictedLatency struct {
	typedName           plugin.TypedName
	latencypredictor    latencypredictor.PredictorInterface
	requestBuilder      PredictionRequestBuilder
	runningRequestLists sync.Map                                      // Key: types.NamespacedName, Value: *requestPriorityQueue
	sloContextStore     *ttlcache.Cache[string, *predictedLatencyCtx] // TTL cache for request contexts
	headroomStrategy    headroomStrategy
	config              Config
}

var _ framework.Scorer = &PredictedLatency{}

type Config struct {
	SamplingMean              float64       `json:"samplingMean,omitempty"`
	MaxSampledTokens          int           `json:"maxSampledTokens,omitempty"`
	SLOBufferFactor           float64       `json:"sloBufferFactor,omitempty"`
	NegHeadroomTTFTWeight     float64       `json:"negHeadroomTTFTWeight,omitempty"`
	NegHeadroomTPOTWeight     float64       `json:"negHeadroomTPOTWeight,omitempty"`
	HeadroomTTFTWeight        float64       `json:"headroomTTFTWeight,omitempty"`
	HeadroomTPOTWeight        float64       `json:"headroomTPOTWeight,omitempty"`
	HeadroomSelectionStrategy string        `json:"headroomSelectionStrategy,omitempty"`
	CompositeKVWeight         float64       `json:"compositeKVWeight,omitempty"`
	CompositeQueueWeight      float64       `json:"compositeQueueWeight,omitempty"`
	CompositePrefixWeight     float64       `json:"compositePrefixWeight,omitempty"`
	EpsilonExploreSticky      float64       `json:"epsilonExploreSticky,omitempty"`
	EpsilonExploreNeg         float64       `json:"epsilonExploreNeg,omitempty"`
	AffinityGateTau           float64       `json:"affinityGateTau,omitempty"`
	AffinityGateTauGlobal     float64       `json:"affinityGateTauGlobal,omitempty"`
	ContextTTL                time.Duration `json:"contextTTL,omitempty"`
	SelectionMode             string        `json:"selectionMode,omitempty"`
	StreamingMode             bool          `json:"streamingMode,omitempty"`
}

var DefaultConfig = Config{
	SamplingMean:              1000,
	MaxSampledTokens:          5,
	SLOBufferFactor:           1,
	NegHeadroomTTFTWeight:     0.8,
	NegHeadroomTPOTWeight:     0.2,
	HeadroomTTFTWeight:        0.8,
	HeadroomTPOTWeight:        0.2,
	HeadroomSelectionStrategy: "least",
	CompositeKVWeight:         1,
	CompositeQueueWeight:      1,
	CompositePrefixWeight:     1,
	EpsilonExploreSticky:      0.01,
	EpsilonExploreNeg:         0.01,
	AffinityGateTau:           0.80,
	AffinityGateTauGlobal:     0.99,
	ContextTTL:                5 * time.Minute,
	SelectionMode:             "linear",
	StreamingMode:             true,
}

func PredictedLatencyFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := DefaultConfig
	if len(rawParameters) > 0 {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config for PredictedLatency: %w", err)
		}
	}

	if err := parameters.validate(); err != nil {
		return nil, fmt.Errorf("invalid PredictedLatency config: %w", err)
	}

	predictor, err := StartPredictor(handle)
	if err != nil {
		return nil, fmt.Errorf("failed to start latency predictor: %w", err)
	}

	return NewPredictedLatency(parameters, predictor).WithName(name), nil
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
	if c.AffinityGateTauGlobal < 0 || c.AffinityGateTauGlobal > 1 {
		errs = append(errs, fmt.Errorf("affinityGateTauGlobal must be in (0, 1], got %f", c.AffinityGateTauGlobal))
	}

	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}

func NewPredictedLatency(config Config, predictor latencypredictor.PredictorInterface) *PredictedLatency {
	strategy := headroomStrategy(config.HeadroomSelectionStrategy)
	if strategy == "" {
		strategy = headroomStrategyLeast
	}

	predictedLatency := &PredictedLatency{
		typedName:        plugin.TypedName{Type: PredictedLatencyPluginType, Name: PredictedLatencyPluginType},
		latencypredictor: predictor,
		requestBuilder:   &DefaultPredictionRequestBuilder{}, // Default, can be customized via SetRequestBuilder
		// runningRequestLists is a sync.Map and needs no initialization
		headroomStrategy: strategy,
		config:           config,
	}

	predictedLatency.sloContextStore = ttlcache.New(
		ttlcache.WithTTL[string, *predictedLatencyCtx](config.ContextTTL),
	)

	predictedLatency.sloContextStore.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, *predictedLatencyCtx]) {
		if reason != ttlcache.EvictionReasonExpired {
			return
		}
		predictedLatency.removeRequestFromQueue(item.Key(), item.Value())
	})

	go predictedLatency.sloContextStore.Start()
	return predictedLatency
}

// StartPredictor initializes and starts the latency predictor.
// Exported to allow reuse in custom scorer implementations (e.g., llm-d-inference-scheduler).
func StartPredictor(handle plugin.Handle) (latencypredictor.PredictorInterface, error) {
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

func (s *PredictedLatency) TypedName() plugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *PredictedLatency) Category() framework.ScorerCategory {
	return framework.Balance
}

func (s *PredictedLatency) WithName(name string) *PredictedLatency {
	s.typedName.Name = name
	return s
}

// SetRequestBuilder sets a custom prediction request builder.
// This allows external packages (e.g., llm-d-inference-scheduler) to customize
// how prediction and training requests are constructed, for example to add
// pod type information for disaggregated serving scenarios.
func (s *PredictedLatency) SetRequestBuilder(builder PredictionRequestBuilder) {
	if builder != nil {
		s.requestBuilder = builder
	}
}

func (s *PredictedLatency) epsilonGreedyAffinityGate(
	ctx context.Context,
	candidates []endpointPredictionResult,
	r *rand.Rand,
	label string, // e.g. "positive" or "negative"
	prefixStickyThreshold float64,
) ([]endpointPredictionResult, bool) {
	logger := log.FromContext(ctx)
	if prefixStickyThreshold <= 0 {
		// Affinity gating disabled
		logger.V(logutil.DEBUG).Info("Affinity gating disabled (threshold <= 0)", "path", label)
		return candidates, false
	}
	eligible := make([]endpointPredictionResult, 0, len(candidates))
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
	if r.Float64() < s.config.EpsilonExploreSticky {
		logger.V(logutil.DEBUG).Info("ε-greedy: exploring (ignoring affinity gate)",
			"path", label, "epsilon", s.config.EpsilonExploreSticky, "eligibleCount", len(eligible))
		return candidates, false
	}

	logger.V(logutil.DEBUG).Info("ε-greedy: exploiting (apply affinity gate)",
		"path", label, "threshold", prefixStickyThreshold, "eligibleCount", len(eligible), "total", len(candidates))
	return eligible, true
}

// scoreWithoutPredictions provides fallback scoring based only on prefix cache scores
// when latency predictions are unavailable
func (s *PredictedLatency) scoreWithoutPredictions(
	ctx context.Context,
	sloCtx *predictedLatencyCtx,
	endpoints []framework.Endpoint,
	r *rand.Rand,
) map[framework.Endpoint]float64 {
	logger := log.FromContext(ctx)
	logger.V(logutil.TRACE).Info("Using composite-only scoring without predictions")

	scores := make(map[framework.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scores[endpoint] = 0
	}

	if len(endpoints) == 0 {
		return scores
	}

	// Build prediction results with only prefix cache scores
	endpointResults := make([]endpointPredictionResult, 0, len(endpoints))
	for _, endpoint := range endpoints {
		prefixScore := sloCtx.prefixCacheScoresForEndpoints[endpoint.GetMetadata().NamespacedName.Name]
		endpointResults = append(endpointResults, endpointPredictionResult{
			Endpoint:         endpoint,
			PrefixCacheScore: prefixScore,
			IsValid:          true, // All endpoints are valid when we don't check predictions
		})
	}

	// Select based on composite scores (prefix cache + other non-prediction metrics)
	selectedEndpoint := s.selectFromCompositeScores(ctx, endpointResults, r, headroomStrategyCompositeOnly)

	if selectedEndpoint != nil {
		scores[selectedEndpoint] = 1
		logger.V(logutil.TRACE).Info("Selected endpoint using composite-only scoring", "endpoint", selectedEndpoint.GetMetadata().String())
	}

	return scores
}

func (s *PredictedLatency) Score(ctx context.Context, state *framework.CycleState, request *framework.LLMRequest, endpoints []framework.Endpoint) map[framework.Endpoint]float64 {
	logger := log.FromContext(ctx)
	if s.latencypredictor == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency: no predictor configured, returning nil scores")
		return nil
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	sloCtx := s.getOrMakePredictedLatencyContextForRequest(request)

	predictions, err := s.generatePredictions(ctx, request, sloCtx, endpoints)
	if err != nil || len(predictions) == 0 {
		logger.V(logutil.DEBUG).Error(err, "PredictedLatency: Error generating predictions, falling back to composite-only scoring")
		s.setPredictedLatencyContextForRequest(request, sloCtx)
		return s.scoreWithoutPredictions(ctx, sloCtx, endpoints, rng)
	}
	s.updateRequestContextWithPredictions(sloCtx, predictions)

	// Initialize scores map with all pods having score 0
	scores := make(map[framework.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scores[endpoint] = 0
	}
	allPreds := append([]endpointPredictionResult(nil), predictions...)
	allPreds, sticky := s.epsilonGreedyAffinityGate(ctx, allPreds, rng, "overall", s.config.AffinityGateTauGlobal)

	// Check if all pods are invalid and all have running requests
	allEndpointsInvalid := (sloCtx.ttftSLO > 0 && (sloCtx.avgTPOTSLO > 0 || !s.config.StreamingMode))
	allEndpointsHaveRunningRequests := true

	for _, pred := range allPreds {
		if pred.IsValid {
			allEndpointsInvalid = false
		}

		runningRequestCount := s.getEndpointRunningRequestCount(pred.Endpoint)
		if runningRequestCount == 0 {
			allEndpointsHaveRunningRequests = false
		}
	}

	// Set HasValidEndpoint to false if all endpoints are invalid and all have running requests
	if allEndpointsInvalid && allEndpointsHaveRunningRequests && !sticky {
		sloCtx.hasValidEndpoint = false
		logger.V(logutil.DEBUG).Info("All endpoints are invalid and have running requests, setting HasValidEndpoint to false")
	}

	// 2) Tiered selection: positive headroom pods get 99% probability, negative get 1%
	posHeadroomEndpoints, negHeadroomEndpoints := s.classifyEndpointsByHeadroom(allPreds)

	logger.V(logutil.DEBUG).Info("Pod headroom distribution",
		"positivePods", len(posHeadroomEndpoints),
		"negativePods", len(negHeadroomEndpoints))

	selectedEndpoint := s.selectEndpointBasedOnStrategy(ctx, rng, allPreds, posHeadroomEndpoints, negHeadroomEndpoints)

	// Set score = 1 for selected endpoint, 0 for all others
	if selectedEndpoint != nil {
		scores[selectedEndpoint] = 1
		logger.V(logutil.DEBUG).Info("Selected endpoint for scheduling", "endpoint", selectedEndpoint.GetMetadata().String())
	}

	s.setPredictedLatencyContextForRequest(request, sloCtx)

	return scores
}

func (t *PredictedLatency) getOrMakePredictedLatencyContextForRequest(request *framework.LLMRequest) *predictedLatencyCtx {
	sloCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		sloCtx = newPredictedLatencyContext(request)
	}

	return sloCtx
}

func (s *PredictedLatency) getPrefixCacheScoreForPod(ctx context.Context, cycleState *framework.CycleState, endpoint framework.Endpoint) float64 {
	log.FromContext(ctx).V(logutil.DEBUG).Info("Running getPrefixCacheScoreForPod, getting prefix cache score for endpoint", "endpoint", endpoint.GetMetadata().String())
	plugintype := prefix.PrefixCachePluginType
	pluginname := prefix.PrefixCachePluginType
	cycleStateKey := (plugin.TypedName{Type: plugintype, Name: pluginname}).String()
	stateData, err := cycleState.Read(plugin.StateKey(cycleStateKey))

	log.FromContext(ctx).V(logutil.DEBUG).Info("Reading prefix cache state from cycle state", "stateKey", cycleStateKey)

	if err != nil {
		// The prefix cache plugin might not be enabled, which is a valid scenario.
		log.FromContext(ctx).V(logutil.DEBUG).Info("prefix cache state not found in cycle state, returning prefix cache score of 0.0", "pod", endpoint.GetMetadata().String())
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

	matchLen := prefixCacheState.PrefixCacheServers[prefix.ServerID(endpoint.GetMetadata().NamespacedName)]
	log.FromContext(ctx).V(logutil.DEBUG).Info("Prefix cache score for endpoint", "endpoint", endpoint.GetMetadata().String(), "matchLen", matchLen, "totalPrefixes", total)
	return float64(matchLen) / float64(total)
}
