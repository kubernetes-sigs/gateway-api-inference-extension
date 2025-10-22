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
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/latencypredictorasync"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

// HeadroomStrategy defines how positive headroom pods should be weighted
type HeadroomStrategy string

type Choice struct {
	PodName schedulingtypes.Pod
	Weight  int
}

const (
	// HeadroomStrategyLeast prioritizes pods with least positive headroom (better packing)
	HeadroomStrategyLeast HeadroomStrategy = "least"
	// HeadroomStrategyMost prioritizes pods with most positive headroom (more conservative)
	HeadroomStrategyMost HeadroomStrategy = "most"
)

const (
	SLOAwareRouterPluginType = "slo-aware-routing"
	MinScore                 = 0
	MaxScore                 = 100
)

var SLOBufferFactor = func() float64 {
	if value, exists := os.LookupEnv("SLO_BUFFER_FACTOR"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil {
			return parsedValue
		}
	}
	return 1.0 // default value
}()

var NegHeadroomTTFTWeight = func() float64 {
	if value, exists := os.LookupEnv("NEG_HEADROOM_TTFT_WEIGHT"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue >= 0 {
			return parsedValue
		}
	}
	return 0.8 // default: TTFT dominates when violating SLOs
}()

var NegHeadroomTPOTWeight = func() float64 {
	if value, exists := os.LookupEnv("NEG_HEADROOM_TPOT_WEIGHT"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue >= 0 {
			return parsedValue
		}
	}
	return 0.2 // default: TPOT less important in your tiny-output scenario
}()

var HeadroomTTFTWeight = func() float64 {
	if value, exists := os.LookupEnv("HEADROOM_TTFT_WEIGHT"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue >= 0 {
			return parsedValue
		}
	}
	return 0.8 // default
}()

var HeadroomTPOTWeight = func() float64 {
	if value, exists := os.LookupEnv("HEADROOM_TPOT_WEIGHT"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue >= 0 {
			return parsedValue
		}
	}
	return 0.2 // default
}()

var HeadroomSelectionStrategy = func() HeadroomStrategy {
	if value, exists := os.LookupEnv("HEADROOM_SELECTION_STRATEGY"); exists {
		switch strings.ToLower(value) {
		case "least":
			return HeadroomStrategyLeast
		case "most":
			return HeadroomStrategyMost
		}
	}
	return HeadroomStrategyLeast // default to least (better packing)
}()

// With probability ε, explore (ignore affinity gate); otherwise exploit.
var EpsilonExplore = func() float64 {
	// Prefer new env; fall back to old for compatibility.
	if v, ok := os.LookupEnv("STICKY_EPSILON"); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f <= 1 {
			return f
		}
	}
	return 0.01 // default 1% exploration
}()

// τ for per-path affinity gate (aka "stickiness" threshold).
var AffinityGateTau = func() float64 {
	// Prefer new env; fall back to old for compatibility.
	if v, ok := os.LookupEnv("AFFINITY_GATE_TAU"); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f <= 1 {
			return f
		}
	}
	return 0.80
}()

// Global τ for the overall candidate set (previously "overall stickiness").
var AffinityGateTauGlobal = func() float64 {
	// Prefer new env; fall back to old for compatibility.
	if v, ok := os.LookupEnv("AFFINITY_GATE_TAU_GLOBAL"); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f <= 1 {
			return f
		}
	}
	return 0.99
}()

type PodPredictionResult struct {
	Pod              schedulingtypes.Pod
	TTFT             float64
	TPOT             float64
	TTFTValid        bool
	TPOTValid        bool
	IsValid          bool
	Error            error
	Headroom         float64 // Headroom for the pod, if applicable
	TTFTHeadroom     float64 // TTFT headroom for the pod
	PrefixCacheScore float64 // Prefix cache score for the pod
}

type SLOAwareRouter struct {
	tn                  plugins.TypedName
	latencypredictor    latencypredictor.PredictorInterface
	runningRequestLists map[types.NamespacedName]*RequestPriorityQueue
	headroomStrategy    HeadroomStrategy
}

func (s *SLOAwareRouter) Dependencies() []plugins.TypedName {
	return []plugins.TypedName{
		{Type: "prefix-cache-scorer", Name: "prefix-cache-scorer"},
	}
}

var _ framework.Scorer = &SLOAwareRouter{}
var _ requestcontrol.PreRequest = &SLOAwareRouter{}
var _ requestcontrol.ResponseReceived = &SLOAwareRouter{}
var _ requestcontrol.ResponseStreaming = &SLOAwareRouter{}
var _ requestcontrol.ResponseComplete = &SLOAwareRouter{}

func NewSLOAwareRouter(latencypredictor latencypredictor.PredictorInterface, strategy HeadroomStrategy) *SLOAwareRouter {
	return &SLOAwareRouter{
		tn:               plugins.TypedName{Type: SLOAwareRouterPluginType, Name: SLOAwareRouterPluginType},
		latencypredictor: latencypredictor,
		headroomStrategy: strategy,
	}
}

func (s *SLOAwareRouter) TypedName() plugins.TypedName {
	return s.tn
}

func (s *SLOAwareRouter) WithName(name string) *SLOAwareRouter {
	s.tn.Name = name
	return s
}

// SetHeadroomStrategy allows runtime configuration of headroom selection strategy
func (s *SLOAwareRouter) SetHeadroomStrategy(strategy HeadroomStrategy) {
	s.headroomStrategy = strategy
}

// GetHeadroomStrategy returns the current headroom selection strategy
func (s *SLOAwareRouter) GetHeadroomStrategy() HeadroomStrategy {
	return s.headroomStrategy
}

func (s *SLOAwareRouter) epsilonGreedyAffinityGate(
	ctx context.Context,
	candidates []PodPredictionResult,
	r *rand.Rand,
	label string, // e.g. "positive" or "negative"
	prefixStickyThreshold float64,
) ([]PodPredictionResult, bool) {
	logger := log.FromContext(ctx)

	eligible := make([]PodPredictionResult, 0, len(candidates))
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
	if r.Float64() < EpsilonExplore {
		logger.V(logutil.DEBUG).Info("ε-greedy: exploring (ignoring affinity gate)",
			"path", label, "epsilon", EpsilonExplore, "eligibleCount", len(eligible))
		return candidates, false
	}

	logger.V(logutil.DEBUG).Info("ε-greedy: exploiting (apply affinity gate)",
		"path", label, "threshold", prefixStickyThreshold, "eligibleCount", len(eligible), "total", len(candidates))
	return eligible, true
}

func (s *SLOAwareRouter) Score(ctx context.Context, state *schedulingtypes.CycleState, request *schedulingtypes.LLMRequest, pods []schedulingtypes.Pod) map[schedulingtypes.Pod]float64 {
	logger := log.FromContext(ctx)
	if s.latencypredictor == nil {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: no predictor configured, returning nil scores")
		return nil
	}

	// Check if SLOs are provided
	if !request.PredictorBasedScheduling {
		logger.V(logutil.DEBUG).Info("PredictorBasedScheduling turned off, skipping prediction-based filtering")
		return nil
	}

	predictions := s.generatePredictions(ctx, state, request, pods)
	s.updateRequestContextWithPredictions(request, predictions)

	allPreds := append([]PodPredictionResult(nil), predictions...)

	// Initialize scores map with all pods having score 0
	scores := make(map[schedulingtypes.Pod]float64, len(pods))
	for _, pod := range pods {
		scores[pod] = 0
	}

	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)
	allPreds, sticky := s.epsilonGreedyAffinityGate(ctx, allPreds, r, "overall", AffinityGateTauGlobal)

	// Check if all pods are invalid and all have running requests
	allPodsInvalid := true
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
		request.HasValidPod = false
		logger.V(logutil.DEBUG).Info("All pods are invalid and have running requests, setting HasValidPod to false")
	}

	// 2) Tiered selection: positive headroom pods get 99% probability, negative get 1%
	var posHeadroomPods, negHeadroomPods []PodPredictionResult
	for _, p := range allPreds {
		// A pod has positive headroom only if BOTH TTFT and TPOT have positive headroom
		if p.Headroom > 0 && p.TTFTHeadroom > 0 {
			posHeadroomPods = append(posHeadroomPods, p)
		} else {
			// A pod has negative headroom if EITHER TTFT or TPOT has negative/zero headroom
			negHeadroomPods = append(negHeadroomPods, p)
		}
	}

	logger.V(logutil.DEBUG).Info("Pod headroom distribution",
		"positivePods", len(posHeadroomPods),
		"negativePods", len(negHeadroomPods))

	var selectedPod schedulingtypes.Pod

	// If both positive and negative headroom pods exist, use tiered selection
	if len(posHeadroomPods) > 0 && len(negHeadroomPods) > 0 {
		// 99% chance to select from positive headroom pods, 1% from negative
		if r.Float64() < EpsilonExplore {
			logger.V(logutil.DEBUG).Info("Selecting from negative headroom pods (1% chance)")
			selectedPod = s.selectFromNegativeHeadroomPods(ctx, negHeadroomPods, r)
		} else {
			logger.V(logutil.DEBUG).Info("Selecting from positive headroom pods (99% chance)")
			selectedPod = s.selectFromPositiveHeadroomPods(ctx, posHeadroomPods, r)
		}
	} else if len(posHeadroomPods) > 0 {
		// If only positive headroom pods exist, select from them
		logger.V(logutil.DEBUG).Info("Only positive headroom pods available")
		selectedPod = s.selectFromPositiveHeadroomPods(ctx, posHeadroomPods, r)
	} else if len(negHeadroomPods) > 0 {
		// If only negative headroom pods exist, select from them
		logger.V(logutil.DEBUG).Info("Only negative headroom pods available")
		selectedPod = s.selectFromNegativeHeadroomPods(ctx, negHeadroomPods, r)
	} else if len(allPreds) > 0 {
		// fallback - select randomly from valid pods
		logger.V(logutil.DEBUG).Info("No headroom pods available, selecting randomly from valid pods")
		selectedPod = allPreds[r.Intn(len(allPreds))].Pod
	} else {
		// No valid pods - return all zeros
		logger.V(logutil.DEBUG).Info("No valid pods available, returning all zero scores")
		return scores
	}

	// Set score = 1 for selected pod, 0 for all others
	if selectedPod != nil {
		scores[selectedPod] = 1
		logger.V(logutil.DEBUG).Info("Selected pod for scheduling", "pod", selectedPod.GetPod().String())
	}

	return scores
}

// selectFromPositiveHeadroomPods selects a pod from positive headroom pods using headroom strategy
// Updated to incorporate TTFTHeadroom with a configurable blend vs TPOT headroom.
func (s *SLOAwareRouter) selectFromPositiveHeadroomPods(ctx context.Context, posHeadroomPods []PodPredictionResult, r *rand.Rand) schedulingtypes.Pod {
	logger := log.FromContext(ctx)

	if len(posHeadroomPods) == 1 {
		return posHeadroomPods[0].Pod
	}

	// Apply perfect stickiness (with exploration)
	candidates, sticky := s.epsilonGreedyAffinityGate(ctx, posHeadroomPods, r, "positive", AffinityGateTau)

	// If perfect stickiness collapsed us to a single pod, short-circuit
	if sticky && len(candidates) == 1 {
		return candidates[0].Pod
	}
	const Wmax = 100
	const minWeight = 1
	const eps = 1e-9

	// Find min/max for TPOT (Headroom) and TTFTHeadroom across positive pods to normalize to [0,1]
	minTPOTH, maxTPOTH := math.MaxFloat64, -math.MaxFloat64
	minTTFTH, maxTTFTH := math.MaxFloat64, -math.MaxFloat64

	for _, p := range candidates {
		if p.Headroom < minTPOTH {
			minTPOTH = p.Headroom
		}
		if p.Headroom > maxTPOTH {
			maxTPOTH = p.Headroom
		}
		if p.TTFTHeadroom < minTTFTH {
			minTTFTH = p.TTFTHeadroom
		}
		if p.TTFTHeadroom > maxTTFTH {
			maxTTFTH = p.TTFTHeadroom
		}
	}

	tpotRange := maxTPOTH - minTPOTH
	ttftRange := maxTTFTH - minTTFTH

	// Precompute blend weights (renormalize if user sets both to 0)
	alpha := HeadroomTTFTWeight
	beta := HeadroomTPOTWeight
	if alpha+beta <= 0 {
		alpha = 1.0
		beta = 0.0
	}
	sum := alpha + beta
	alpha /= sum
	beta /= sum

	logger.V(logutil.DEBUG).Info("Positive headroom normalization ranges",
		"minTPOTHeadroom", minTPOTH, "maxTPOTHeadroom", maxTPOTH,
		"minTTFTHeadroom", minTTFTH, "maxTTFTHeadroom", maxTTFTH,
		"alphaTTFT", alpha, "betaTPOT", beta, "strategy", s.headroomStrategy)

	// Calculate weights for weighted random selection
	weightedChoices := make([]Choice, 0, len(candidates))
	total := 0

	for _, p := range candidates {
		// Normalize to [0,1] within the cohort
		nTPOTH := 0.5
		if tpotRange > eps {
			nTPOTH = (p.Headroom - minTPOTH) / (tpotRange + eps)
		}
		nTTFTH := 0.5
		if ttftRange > eps {
			nTTFTH = (p.TTFTHeadroom - minTTFTH) / (ttftRange + eps)
		}

		// Blend: larger combined -> "safer"; smaller -> "tighter packing"
		combined := alpha*nTTFTH + beta*nTPOTH

		// Map to integer weights
		var w int
		switch s.headroomStrategy {
		case HeadroomStrategyLeast:
			// prefer smaller combined headroom (pack closer to limits)
			w = int((1.0-combined)*float64(Wmax-minWeight)) + minWeight + 1
		case HeadroomStrategyMost:
			// prefer larger combined headroom (more conservative / spread)
			w = int(combined*float64(Wmax-minWeight)) + minWeight + 1
		default:
			// Fallback to least
			w = int((1.0-combined)*float64(Wmax-minWeight)) + minWeight + 1
		}

		weightedChoices = append(weightedChoices, Choice{PodName: p.Pod, Weight: w})
		total += w

		logger.V(logutil.TRACE).Info("Positive headroom blended weight",
			"pod", p.Pod.GetPod().String(),
			"ttftHeadroom", p.TTFTHeadroom, "normTTFTHeadroom", nTTFTH,
			"tpotHeadroom", p.Headroom, "normTPOTHeadroom", nTPOTH,
			"combined", combined, "weight", w)
	}

	// Perform weighted random selection
	idx := r.Intn(total)
	var selectedPod schedulingtypes.Pod

	for _, c := range weightedChoices {
		if idx < c.Weight {
			selectedPod = c.PodName
			break
		}
		idx -= c.Weight
	}

	// If no pod was selected (shouldn't happen), fallback to first pod
	if selectedPod == nil {
		selectedPod = candidates[0].Pod
		selectedPod = posHeadroomPods[0].Pod
	}

	return selectedPod
}

// selectFromNegativeHeadroomPods selects a pod from negative headroom pods using hierarchical TTFT/TPOT logic
// Modified to strictly prefer pods with 0 running requests
func (s *SLOAwareRouter) selectFromNegativeHeadroomPods(ctx context.Context, negHeadroomPods []PodPredictionResult, r *rand.Rand) schedulingtypes.Pod {
	logger := log.FromContext(ctx)

	if len(negHeadroomPods) == 1 {
		return negHeadroomPods[0].Pod
	}

	// First, separate pods by running request count
	var zeroRunningRequestPods, nonZeroRunningRequestPods []PodPredictionResult

	for _, p := range negHeadroomPods {
		runningRequestCount := s.getPodRunningRequestCount(p.Pod)
		if runningRequestCount == 0 {
			zeroRunningRequestPods = append(zeroRunningRequestPods, p)
		} else {
			nonZeroRunningRequestPods = append(nonZeroRunningRequestPods, p)
		}
	}

	logger.V(logutil.DEBUG).Info("Negative headroom pods by running request count",
		"zeroRunningRequests", len(zeroRunningRequestPods),
		"nonZeroRunningRequests", len(nonZeroRunningRequestPods))

	// If we have pods with 0 running requests, strictly prefer them
	if len(zeroRunningRequestPods) > 0 {
		logger.V(logutil.DEBUG).Info("Selecting from pods with zero running requests")
		return s.selectFromNegativeHeadroomPodsInternal(ctx, zeroRunningRequestPods, r)
	}

	// Otherwise, fall back to pods with running requests
	logger.V(logutil.DEBUG).Info("No pods with zero running requests, selecting from pods with running requests")
	return s.selectFromNegativeHeadroomPodsInternal(ctx, nonZeroRunningRequestPods, r)
}

// selectFromNegativeHeadroomPodsInternal handles the actual selection logic for negative headroom pods
func (s *SLOAwareRouter) selectFromNegativeHeadroomPodsInternal(ctx context.Context, negHeadroomPods []PodPredictionResult, r *rand.Rand) schedulingtypes.Pod {
	if len(negHeadroomPods) == 1 {
		return negHeadroomPods[0].Pod
	}

	// Apply perfect stickiness (with exploration)
	candidates, sticky := s.epsilonGreedyAffinityGate(ctx, negHeadroomPods, r, "negative", AffinityGateTau)

	// If perfect stickiness collapsed us to a single pod, short-circuit
	if sticky && len(candidates) == 1 {
		return candidates[0].Pod
	}

	const minWeightForNegative = 1

	// Build weighted choices for selection
	weightedChoices := make([]Choice, 0, len(candidates))
	total := 0

	s.handleNegativeHeadroomPodsHierarchical(ctx, candidates, &weightedChoices, &total, minWeightForNegative)

	// Perform weighted random selection
	idx := r.Intn(total)
	var selectedPod schedulingtypes.Pod

	for _, c := range weightedChoices {
		if idx < c.Weight {
			selectedPod = c.PodName
			break
		}
		idx -= c.Weight
	}

	// If no pod was selected (shouldn't happen), fallback to first pod
	if selectedPod == nil {
		selectedPod = candidates[0].Pod
	}

	return selectedPod
}

// weightPodsByBlendedDeficit applies blended weighting using TTFT and TPOT deficits.
// Lower blended deficit => higher weight.
func (ps *SLOAwareRouter) weightPodsByBlendedDeficit(
	ctx context.Context,
	pods []PodPredictionResult,
	choices *[]Choice,
	total *int,
	minWeight int,
	alpha, beta float64, // weights for TTFT and TPOT deficits
	category string,
) {
	logger := log.FromContext(ctx)
	if len(pods) == 0 {
		return
	}

	const Wrange = 80
	const eps = 1e-9

	// Compute raw deficits (only when headroom is negative)
	type deficits struct {
		pod     PodPredictionResult
		ttftDef float64
		tpotDef float64
	}
	defs := make([]deficits, 0, len(pods))

	minTTFT, maxTTFT := math.MaxFloat64, -math.MaxFloat64
	minTPOT, maxTPOT := math.MaxFloat64, -math.MaxFloat64

	for _, p := range pods {
		ttftDef := 0.0
		if p.TTFTHeadroom < 0 {
			ttftDef = -p.TTFTHeadroom
		}
		tpotDef := 0.0
		if p.Headroom < 0 {
			tpotDef = -p.Headroom
		}
		defs = append(defs, deficits{pod: p, ttftDef: ttftDef, tpotDef: tpotDef})

		if ttftDef < minTTFT {
			minTTFT = ttftDef
		}
		if ttftDef > maxTTFT {
			maxTTFT = ttftDef
		}
		if tpotDef < minTPOT {
			minTPOT = tpotDef
		}
		if tpotDef > maxTPOT {
			maxTPOT = tpotDef
		}
	}

	ttftRange := maxTTFT - minTTFT
	tpotRange := maxTPOT - minTPOT

	// Normalize alpha/beta
	if alpha+beta <= 0 {
		alpha, beta = 1.0, 0.0
	} else {
		sum := alpha + beta
		alpha /= sum
		beta /= sum
	}

	logger.V(logutil.DEBUG).Info("Negative headroom blended deficits",
		"category", category,
		"minTTFTDef", minTTFT, "maxTTFTDef", maxTTFT,
		"minTPOTDef", minTPOT, "maxTPOTDef", maxTPOT,
		"alphaTTFT", alpha, "betaTPOT", beta, "podCount", len(pods))

	for _, d := range defs {
		// Normalize deficits to [0,1] within this bucket (0 = best / least violation)
		nTTFT := 0.0
		if ttftRange > eps {
			nTTFT = (d.ttftDef - minTTFT) / (ttftRange + eps)
		}
		nTPOT := 0.0
		if tpotRange > eps {
			nTPOT = (d.tpotDef - minTPOT) / (tpotRange + eps)
		}

		// Blended "badness": higher = worse violation
		blended := alpha*nTTFT + beta*nTPOT

		// Convert to selection weight: lower badness -> higher weight
		// Ensure a floor so no pod is completely excluded within the bucket.
		w := int((1.0-blended)*float64(Wrange)) + minWeight + 1

		*choices = append(*choices, Choice{PodName: d.pod.Pod, Weight: w})
		*total += w

		logger.V(logutil.TRACE).Info("Negative bucket blended weighting",
			"pod", d.pod.Pod.GetPod().String(),
			"ttftDef", d.ttftDef, "tpotDef", d.tpotDef,
			"normTTFT", nTTFT, "normTPOT", nTPOT,
			"blendedBadness", blended, "weight", w)
	}
}

func (s *SLOAwareRouter) handleNegativeHeadroomPodsHierarchical(
	ctx context.Context,
	negHeadroomPods []PodPredictionResult,
	choices *[]Choice,
	total *int,
	minWeightForNegative int,
) {
	logger := log.FromContext(ctx)

	// Categorize pods by their headroom status
	var negTTFTNegTPOT, negTTFTNonNegTPOT, nonNegTTFTNegTPOT, nonNegTTFTNonNegTPOT []PodPredictionResult

	for _, p := range negHeadroomPods {
		if p.TTFTHeadroom < 0 && p.Headroom < 0 {
			negTTFTNegTPOT = append(negTTFTNegTPOT, p)
		} else if p.TTFTHeadroom < 0 && p.Headroom >= 0 {
			negTTFTNonNegTPOT = append(negTTFTNonNegTPOT, p)
		} else if p.TTFTHeadroom >= 0 && p.Headroom < 0 {
			nonNegTTFTNegTPOT = append(nonNegTTFTNegTPOT, p)
		} else {
			nonNegTTFTNonNegTPOT = append(nonNegTTFTNonNegTPOT, p)
		}
	}

	logger.V(logutil.DEBUG).Info("Hierarchical negative headroom pod distribution",
		"totalNegative", len(negHeadroomPods),
		"negTTFT_negTPOT", len(negTTFTNegTPOT),
		"negTTFT_nonNegTPOT", len(negTTFTNonNegTPOT),
		"nonNegTTFT_negTPOT", len(nonNegTTFTNegTPOT),
		"nonNegTTFT_nonNegTPOT", len(nonNegTTFTNonNegTPOT))

	// Priority 1: both TTFT and TPOT negative -> blended deficits (both active)
	if len(negTTFTNegTPOT) > 0 {
		s.weightPodsByBlendedDeficit(ctx, negTTFTNegTPOT, choices, total, minWeightForNegative,
			NegHeadroomTTFTWeight, NegHeadroomTPOTWeight, "both_negative")
	}

	// Priority 2: TTFT negative, TPOT non-negative -> blended still works (TPOT deficit=0)
	if len(negTTFTNonNegTPOT) > 0 {
		s.weightPodsByBlendedDeficit(ctx, negTTFTNonNegTPOT, choices, total, minWeightForNegative,
			NegHeadroomTTFTWeight, NegHeadroomTPOTWeight, "ttft_negative")
	}

	// Priority 3: TTFT non-negative, TPOT negative -> blended (TTFT deficit=0)
	if len(nonNegTTFTNegTPOT) > 0 {
		s.weightPodsByBlendedDeficit(ctx, nonNegTTFTNegTPOT, choices, total, minWeightForNegative,
			NegHeadroomTTFTWeight, NegHeadroomTPOTWeight, "tpot_negative")
	}

	// Priority 4: edge-case bucket -> minimal weight
	for _, p := range nonNegTTFTNonNegTPOT {
		*choices = append(*choices, Choice{PodName: p.Pod, Weight: minWeightForNegative})
		*total += minWeightForNegative
	}
}

// generatePredictions creates prediction results for all candidate pods
func (s *SLOAwareRouter) generatePredictions(ctx context.Context, state *schedulingtypes.CycleState, request *schedulingtypes.LLMRequest, candidatePods []schedulingtypes.Pod) []PodPredictionResult {
	logger := log.FromContext(ctx)
	predictions := make([]PodPredictionResult, 0, len(candidatePods))

	for _, pod := range candidatePods {
		predResult := PodPredictionResult{Pod: pod}

		logger.V(logutil.TRACE).Info("Candidate pod for scheduling", "pod", pod.GetPod().String(), "metrics", pod.GetMetrics().String())

		// Get prefix cache score for the pod
		prefixCacheScore := s.getPrefixCacheScoreForPod(ctx, state, pod)

		// TODO update the request in the datastore request tracker

		// Generate prediction
		prediction, err := requestcontrol.PredictWithMetrics(ctx, s.latencypredictor, pod.GetMetrics(), request.Body.Completions.Prompt, 1, prefixCacheScore)
		if err != nil {
			logger.V(logutil.DEBUG).Info("Skipping pod due to prediction error", "pod", pod.GetPod().String(), "error", err)
			predResult.Error = err
			predictions = append(predictions, predResult)
			continue
		}
		predResult.PrefixCacheScore = prefixCacheScore
		predResult.TTFT = prediction.TTFT
		predResult.TPOT = prediction.TPOT
		podMinTPOTSLO := 0.0
		//if pod.GetPod().RunningRequests.Peek() != nil {
		//	podMinTPOTSLO = pod.GetPod().RunningRequests.Peek().TPOT
		//}
		// Do this:
		podMinTPOTSLO = s.getPodMinTPOTSLO(pod)
		predResult.TTFTValid, predResult.TPOTValid, predResult.IsValid, predResult.Headroom, predResult.TTFTHeadroom = s.validatePrediction(prediction, request, podMinTPOTSLO)

		logger.V(logutil.DEBUG).Info("Prediction for scheduling",
			"pod", pod.GetPod().String(),
			"prefixCacheScore", prefixCacheScore,
			"TTFT", prediction.TTFT,
			"TPOT", prediction.TPOT,
			"buffer", SLOBufferFactor,
			"podMinTPOTSLO", podMinTPOTSLO,
			"ttftSLO", request.TTFTSLO,
			"requestTPOTSLO", request.AvgTPOTSLO,
			"tpotHeadroom", predResult.Headroom,
			"ttftHeadroom", predResult.TTFTHeadroom,
			"tpotValid", predResult.TPOTValid,
			"ttftValid", predResult.TTFTValid,
			"headroomStrategy", s.headroomStrategy)

		predictions = append(predictions, predResult)
	}

	return predictions
}

func (s *SLOAwareRouter) getPodMinTPOTSLO(pod schedulingtypes.Pod) float64 {
	podName := types.NamespacedName{
		Name:      pod.GetPod().NamespacedName.Name,
		Namespace: pod.GetPod().NamespacedName.Namespace,
	}
	if runningReqs, ok := s.runningRequestLists[podName]; ok && runningReqs.GetSize() > 0 {
		if topReq := runningReqs.Peek(); topReq != nil {
			return topReq.TPOT
		}
	}
	return 0 // no running requests or no TPOT SLOs
}

func (s *SLOAwareRouter) getPodRunningRequestCount(pod schedulingtypes.Pod) int {
	podName := types.NamespacedName{
		Name:      pod.GetPod().NamespacedName.Name,
		Namespace: pod.GetPod().NamespacedName.Namespace,
	}
	if runningReqs, ok := s.runningRequestLists[podName]; ok {
		return runningReqs.GetSize()
	}
	return 0 // no running requests
}

func (s *SLOAwareRouter) validatePrediction(
	pred *latencypredictor.PredictionResponse,
	req *schedulingtypes.LLMRequest,
	podMinTPOTSLO float64,
) (ttftOk, tpotOk, isValid bool, headroom float64, ttftHeadroom float64) {

	bufferedTPOT := req.AvgTPOTSLO * SLOBufferFactor
	// a podMinTPOTSLO of 0 means no either no requests, or no TPOT SLOs specified on running requests
	if podMinTPOTSLO > 0 {
		if podMinTPOTSLO < req.AvgTPOTSLO {
			//print debug message
			log.FromContext(context.Background()).V(logutil.DEBUG).Info("Pod min TPOT SLO is less than the req SLO, adjusting", "podMinTPOTSLO", podMinTPOTSLO, "bufferedTPOT", req.AvgTPOTSLO)
		}
		bufferedTPOT = min(bufferedTPOT, podMinTPOTSLO*SLOBufferFactor)
	}

	tpotOk = pred.TPOT < bufferedTPOT
	ttftOk = pred.TTFT < req.TTFTSLO

	isValid = ttftOk && tpotOk
	headroom = bufferedTPOT - pred.TPOT
	ttftHeadroom = req.TTFTSLO - pred.TTFT
	return
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
		log.FromContext(ctx).V(logutil.DEBUG).Info("Prefix cache state not found in cycle state, returning prefix cache score of 0.0", "pod", pod.GetPod().String())
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

// updateRequestContextWithPredictions updates the request context with prediction data
func (s *SLOAwareRouter) updateRequestContextWithPredictions(request *schedulingtypes.LLMRequest, predictions []PodPredictionResult) {
	for _, pred := range predictions {
		if pred.Error == nil {
			podKey := pred.Pod.GetPod().String()
			if request.PredictedTTFTForScheduling == nil {
				request.PredictedTTFTForScheduling = make(map[string]float64)
			}
			if request.PredictedTPOTForScheduling == nil {
				request.PredictedTPOTForScheduling = make(map[string]float64)
			}
			request.PredictedTTFTForScheduling[podKey] = pred.TTFT
			request.PredictedTPOTForScheduling[podKey] = pred.TPOT
		}
	}
}

// REQUEST CONTROL PLUGIN HOOKS

func (t *SLOAwareRouter) PreRequest(ctx context.Context, request *schedulingtypes.LLMRequest, schedulingResult *schedulingtypes.SchedulingResult, targetPort int) {
	logger := log.FromContext(ctx)

	if schedulingResult == nil || len(schedulingResult.ProfileResults) == 0 {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: Skipping PreRequest because no scheduling result was provided.")
		return
	}

	targetPod := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName].TargetPods[0].GetPod()

	podName := types.NamespacedName{
		Name:      targetPod.NamespacedName.Name,
		Namespace: targetPod.NamespacedName.Namespace,
	}

	logger.V(logutil.DEBUG).Info("request ID for SLO tracking", "requestID", request.Headers[requtil.RequestIdHeaderKey], "podName", podName)
	if request.Headers[requtil.RequestIdHeaderKey] == "" {
		request.Headers[requtil.RequestIdHeaderKey] = uuid.New().String()
		logger.V(logutil.DEBUG).Info("Generated new request ID for SLO tracking", "requestID", request.Headers[requtil.RequestIdHeaderKey])
		logger.V(logutil.DEBUG).Info("request headers for SLO tracking", "requestHeaders", request.Headers)
	}

	id := request.Headers[requtil.RequestIdHeaderKey]
	podRequestList, ok := t.runningRequestLists[podName]
	if !ok {
		t.runningRequestLists[podName] = NewRequestPriorityQueue()
	}

	added := podRequestList.Add(id, request.AvgTPOTSLO)
	if !added {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: Item already exists in queue", "podName", podName, "requestID", id)
	}

}

func (t *SLOAwareRouter) PostResponse(ctx context.Context, reqCtx *handlers.RequestContext) {
	logger := log.FromContext(ctx)
	targetPod := reqCtx.TargetPod
	if !t.CheckPredictor(logger, targetPod) {
		return
	}

	if err := requestcontrol.ProcessHeaderForLatencyPrediction(ctx, t.latencypredictor, reqCtx); err != nil {
		logger.V(logutil.DEBUG).Error(err, "ProcessHeader in latencypredictor failed")
	}

}

func (t *SLOAwareRouter) PostResponseChunk(ctx context.Context, reqCtx *handlers.RequestContext) {
	logger := log.FromContext(ctx)
	targetPod := reqCtx.TargetPod
	if !t.CheckPredictor(logger, targetPod) {
		return
	}

	now := time.Now()

	if reqCtx.TTFT == 0 {
		requestcontrol.ProcessFirstTokenForLatencyPrediction(ctx, t.latencypredictor, reqCtx, now)
	} else {
		requestcontrol.ProcessTokenForLatencyPrediction(ctx, t.latencypredictor, reqCtx, now)
	}

}

func (t *SLOAwareRouter) PostResponseComplete(ctx context.Context, reqCtx *handlers.RequestContext) {
	logger := log.FromContext(ctx)
	request := reqCtx.SchedulingRequest
	targetPod := reqCtx.TargetPod
	if !t.CheckPredictor(logger, targetPod) {
		return
	}

	mapeTTFT := 0.0
	if reqCtx.TTFT > 0 {
		mapeTTFT = math.Abs((reqCtx.TTFT-reqCtx.PredictedTTFT)/reqCtx.TTFT) * 100
		logger.V(logutil.DEBUG).Info("Averages calculated", "avgActualTTFT", reqCtx.TTFT, "avgPredictedTTFT", reqCtx.PredictedTTFT)
		logger.V(logutil.DEBUG).Info("MAPE TTFT computed", "mapeTTFT%", mapeTTFT)
		metrics.RecordRequestTTFT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.TTFT/1000)
		metrics.RecordRequestPredictedTTFT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.PredictedTTFT/1000)
	}

	mapeTPOT := 0.0
	if reqCtx.AvgTPOT > 0 {
		mapeTPOT = math.Abs((reqCtx.AvgTPOT-reqCtx.AvgPredictedTPOT)/reqCtx.AvgTPOT) * 100
		logger.V(logutil.DEBUG).Info("Averages calculated", "avgActualTPOT", reqCtx.AvgTPOT, "avgPredictedTPOT", reqCtx.AvgPredictedTPOT)
		logger.V(logutil.DEBUG).Info("MAPE TPOT computed", "mapeTPOT%", mapeTPOT)
		metrics.RecordRequestTPOT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.AvgTPOT/1000)
		metrics.RecordRequestPredictedTPOT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.AvgPredictedTPOT/1000)
	}
	logger.V(logutil.DEBUG).Info("SLO Aware Routing Mode", "PredictorBasedScheduling", request.PredictorBasedScheduling)

	podName := types.NamespacedName{
		Name:      targetPod.NamespacedName.Name,
		Namespace: targetPod.NamespacedName.Namespace,
	}

	id := request.Headers[requtil.RequestIdHeaderKey]
	podRequestList, ok := t.runningRequestLists[podName]
	if !ok {
		err := fmt.Errorf("No running request list found for pod %s", podName.String())
		logger.V(logutil.DEBUG).Error(err, "SLOAwareRouter: Failed to remove request from queue", "requestID", id)
	}

	_, removed := podRequestList.Remove(id)
	if !removed {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: Item not found in queue", "podName", podName, "requestID", id)
	}
}

func (t *SLOAwareRouter) CheckPredictor(logger logr.Logger, targetPod *backend.Pod) bool {
	if targetPod == nil {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: Skipping PostResponse because no target pod was provided.")
		return false
	}
	if t.latencypredictor == nil {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: Skipping PostResponse because predictor missing")
		return false
	}
	return true
}
