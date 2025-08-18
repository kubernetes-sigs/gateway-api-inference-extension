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

package requestcontrol

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"

	"os"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/latencypredictorasync"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// HeadroomStrategy defines how positive headroom pods should be weighted
type HeadroomStrategy string

const (
	// HeadroomStrategyLeast prioritizes pods with least positive headroom (better packing)
	HeadroomStrategyLeast HeadroomStrategy = "least"
	// HeadroomStrategyMost prioritizes pods with most positive headroom (more conservative)
	HeadroomStrategyMost HeadroomStrategy = "most"
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

// PodPredictionResult holds prediction results for a single pod
type PodPredictionResult struct {
	Pod         schedulingtypes.Pod
	TTFT        float64
	TPOT        float64
	TTFTValid   bool
	TPOTValid   bool
	IsValid     bool
	Error       error
	Headroom    float64 // Headroom for the pod, if applicable
	TTFTHeadroom float64 // TTFT headroom for the pod
}

// PredictionScorer handles prediction-based pod scoring and filtering
type PredictionScorer struct {
	predictor        latencypredictor.PredictorInterface
	headroomStrategy HeadroomStrategy
}

// NewPredictionScorer creates a new PredictionScorer instance
func NewPredictionScorer(predictor latencypredictor.PredictorInterface) *PredictionScorer {
	return &PredictionScorer{
		predictor:        predictor,
		headroomStrategy: HeadroomSelectionStrategy,
	}
}

// NewPredictionScorerWithStrategy creates a new PredictionScorer instance with explicit strategy
func NewPredictionScorerWithStrategy(predictor latencypredictor.PredictorInterface, strategy HeadroomStrategy) *PredictionScorer {
	return &PredictionScorer{
		predictor:        predictor,
		headroomStrategy: strategy,
	}
}

// SetHeadroomStrategy allows runtime configuration of headroom selection strategy
func (ps *PredictionScorer) SetHeadroomStrategy(strategy HeadroomStrategy) {
	ps.headroomStrategy = strategy
}

// GetHeadroomStrategy returns the current headroom selection strategy
func (ps *PredictionScorer) GetHeadroomStrategy() HeadroomStrategy {
	return ps.headroomStrategy
}

// / ScoreAndFilterPods evaluates candidate pods using latency predictions and filters them based on SLO requirements
func (ps *PredictionScorer) ScoreAndFilterPods(ctx context.Context, datastore datastore.Datastore, reqCtx *handlers.RequestContext, candidatePods []schedulingtypes.Pod, result *schedulingtypes.SchedulingResult, requestCriticality v1alpha2.Criticality) (schedulingtypes.Pod, error) {
	logger := log.FromContext(ctx)

	if ps.predictor == nil {
		return nil, fmt.Errorf("predictor is not available")
	}

	// Check if SLOs are provided
	if !reqCtx.SchedulingRequest.PredictorBasedScheduling {
		logger.V(logutil.DEBUG).Info("SLOs not provided, skipping prediction-based filtering")
		return nil, nil
	}

	predictions := ps.generatePredictions(ctx, datastore, candidatePods, result, reqCtx)
	ps.updateRequestContextWithPredictions(reqCtx, predictions)

	var validPreds, invalidPreds []PodPredictionResult
	for _, p := range predictions {
		if p.IsValid || ps.getPodRunningRequestCount(datastore, p.Pod) == 0 { // If the pod is valid or has no running requests, consider it valid
			validPreds = append(validPreds, p)
		} else {
			invalidPreds = append(invalidPreds, p)
		}
	}

	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)

	// 1) if no valid pods, fallback for critical vs non‑critical
	if len(validPreds) == 0 {
		defaultPod := result.ProfileResults[result.PrimaryProfileName].TargetPods[0]
		if requestCriticality == v1alpha2.Critical {
			return defaultPod, nil
		}
		return nil, errutil.Error{
			Code: errutil.InferencePoolResourceExhausted,
			Msg:  "no valid pods after prediction filtering for non-critical request",
		}
	}

	// 2) Tiered selection: positive headroom pods get 99% probability, negative get 1%
	var posHeadroomPods, negHeadroomPods []PodPredictionResult
	for _, p := range validPreds {
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

	// If both positive and negative headroom pods exist, use tiered selection
	if len(posHeadroomPods) > 0 && len(negHeadroomPods) > 0 {
		// 99% chance to select from positive headroom pods, 1% from negative
		if r.Float64() < 0.01 {
			logger.V(logutil.DEBUG).Info("Selecting from negative headroom pods (1% chance)")
			return ps.selectFromNegativeHeadroomPods(ctx, negHeadroomPods, r)
		} else {
			logger.V(logutil.DEBUG).Info("Selecting from positive headroom pods (99% chance)")
			return ps.selectFromPositiveHeadroomPods(ctx, posHeadroomPods, r)
		}
	}

	// If only positive headroom pods exist, select from them
	if len(posHeadroomPods) > 0 {
		logger.V(logutil.DEBUG).Info("Only positive headroom pods available")
		return ps.selectFromPositiveHeadroomPods(ctx, posHeadroomPods, r)
	}

	// If only negative headroom pods exist, select from them
	if len(negHeadroomPods) > 0 {
		logger.V(logutil.DEBUG).Info("Only negative headroom pods available")
		return ps.selectFromNegativeHeadroomPods(ctx, negHeadroomPods, r)
	}

	// fallback (shouldn't happen)
	return validPreds[0].Pod, nil
}

// selectFromPositiveHeadroomPods selects a pod from positive headroom pods using headroom strategy
// Updated to incorporate TTFTHeadroom with a configurable blend vs TPOT headroom.
func (ps *PredictionScorer) selectFromPositiveHeadroomPods(ctx context.Context, posHeadroomPods []PodPredictionResult, r *rand.Rand) (schedulingtypes.Pod, error) {
	logger := log.FromContext(ctx)

	if len(posHeadroomPods) == 1 {
		return posHeadroomPods[0].Pod, nil
	}

	const Wmax = 100
	const minWeight = 1
	const eps = 1e-9

	total := 0
	choices := make([]Choice, 0, len(posHeadroomPods))

	// Find min/max for TPOT (Headroom) and TTFTHeadroom across positive pods to normalize to [0,1]
	minTPOTH, maxTPOTH := math.MaxFloat64, -math.MaxFloat64
	minTTFTH, maxTTFTH := math.MaxFloat64, -math.MaxFloat64

	for _, p := range posHeadroomPods {
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
		"alphaTTFT", alpha, "betaTPOT", beta, "strategy", ps.headroomStrategy)

	for _, p := range posHeadroomPods {
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
		switch ps.headroomStrategy {
		case HeadroomStrategyLeast:
			// prefer smaller combined headroom (pack closer to limits)
			w = int((1.0 - combined)*float64(Wmax-minWeight)) + minWeight + 1
		case HeadroomStrategyMost:
			// prefer larger combined headroom (more conservative / spread)
			w = int(combined*float64(Wmax-minWeight)) + minWeight + 1
		default:
			// Fallback to least
			w = int((1.0 - combined)*float64(Wmax-minWeight)) + minWeight + 1
		}

		choices = append(choices, Choice{PodName: p.Pod, Weight: w})
		total += w

		logger.V(logutil.TRACE).Info("Positive headroom blended weight",
			"pod", p.Pod.GetPod().String(),
			"ttftHeadroom", p.TTFTHeadroom, "normTTFTHeadroom", nTTFTH,
			"tpotHeadroom", p.Headroom,     "normTPOTHeadroom", nTPOTH,
			"combined", combined, "weight", w)
	}

	// Select pod using weighted random
	idx := r.Intn(total)
	for _, c := range choices {
		if idx < c.Weight {
			return c.PodName, nil
		}
		idx -= c.Weight
	}
	return posHeadroomPods[0].Pod, nil
}


// selectFromNegativeHeadroomPods selects a pod from negative headroom pods using hierarchical TTFT/TPOT logic
func (ps *PredictionScorer) selectFromNegativeHeadroomPods(ctx context.Context, negHeadroomPods []PodPredictionResult, r *rand.Rand) (schedulingtypes.Pod, error) {

	if len(negHeadroomPods) == 1 {
		return negHeadroomPods[0].Pod, nil
	}

	const minWeightForNegative = 1
	total := 0
	choices := make([]Choice, 0, len(negHeadroomPods))

	ps.handleNegativeHeadroomPodsHierarchical(ctx, negHeadroomPods, &choices, &total, minWeightForNegative)

	// Select pod using weighted random selection
	idx := r.Intn(total)
	for _, c := range choices {
		if idx < c.Weight {
			return c.PodName, nil
		}
		idx -= c.Weight
	}

	// fallback
	return negHeadroomPods[0].Pod, nil
}

// weightPodsByBlendedDeficit applies blended weighting using TTFT and TPOT deficits.
// Lower blended deficit => higher weight.
func (ps *PredictionScorer) weightPodsByBlendedDeficit(
	ctx context.Context,
	pods []PodPredictionResult,
	choices *[]Choice,
	total *int,
	minWeight int,
	alpha, beta float64,   // weights for TTFT and TPOT deficits
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
		pod         PodPredictionResult
		ttftDef     float64
		tpotDef     float64
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

		if ttftDef < minTTFT { minTTFT = ttftDef }
		if ttftDef > maxTTFT { maxTTFT = ttftDef }
		if tpotDef < minTPOT { minTPOT = tpotDef }
		if tpotDef > maxTPOT { maxTPOT = tpotDef }
	}

	ttftRange := maxTTFT - minTTFT
	tpotRange := maxTPOT - minTPOT

	// Normalize alpha/beta
	if alpha+beta <= 0 {
		alpha, beta = 1.0, 0.0
	} else {
		sum := alpha + beta
		alpha /= sum
		beta  /= sum
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
		w := int((1.0 - blended)*float64(Wrange)) + minWeight + 1

		*choices = append(*choices, Choice{PodName: d.pod.Pod, Weight: w})
		*total += w

		logger.V(logutil.TRACE).Info("Negative bucket blended weighting",
			"pod", d.pod.Pod.GetPod().String(),
			"ttftDef", d.ttftDef, "tpotDef", d.tpotDef,
			"normTTFT", nTTFT, "normTPOT", nTPOT,
			"blendedBadness", blended, "weight", w)
	}
}


func (ps *PredictionScorer) handleNegativeHeadroomPodsHierarchical(
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
		ps.weightPodsByBlendedDeficit(ctx, negTTFTNegTPOT, choices, total, minWeightForNegative,
			NegHeadroomTTFTWeight, NegHeadroomTPOTWeight, "both_negative")
	}

	// Priority 2: TTFT negative, TPOT non-negative -> blended still works (TPOT deficit=0)
	if len(negTTFTNonNegTPOT) > 0 {
		ps.weightPodsByBlendedDeficit(ctx, negTTFTNonNegTPOT, choices, total, minWeightForNegative,
			NegHeadroomTTFTWeight, NegHeadroomTPOTWeight, "ttft_negative")
	}

	// Priority 3: TTFT non-negative, TPOT negative -> blended (TTFT deficit=0)
	if len(nonNegTTFTNegTPOT) > 0 {
		ps.weightPodsByBlendedDeficit(ctx, nonNegTTFTNegTPOT, choices, total, minWeightForNegative,
			NegHeadroomTTFTWeight, NegHeadroomTPOTWeight, "tpot_negative")
	}

	// Priority 4: edge-case bucket -> minimal weight
	for _, p := range nonNegTTFTNonNegTPOT {
		*choices = append(*choices, Choice{PodName: p.Pod, Weight: minWeightForNegative})
		*total += minWeightForNegative
	}
}





// generatePredictions creates prediction results for all candidate pods
func (ps *PredictionScorer) generatePredictions(ctx context.Context, datastore datastore.Datastore, candidatePods []schedulingtypes.Pod, result *schedulingtypes.SchedulingResult, reqCtx *handlers.RequestContext) []PodPredictionResult {
	logger := log.FromContext(ctx)
	predictions := make([]PodPredictionResult, 0, len(candidatePods))

	for _, pod := range candidatePods {
		predResult := PodPredictionResult{Pod: pod}

		logger.V(logutil.TRACE).Info("Candidate pod for scheduling", "pod", pod.GetPod().String(), "metrics", pod.GetMetrics().String())

		// Get prefix cache score for the pod
		prefixCacheScore := GetPrefixCacheScoreForPod(ctx, result, pod, "prefill")

		// Generate prediction
		prediction, err := PredictWithMetrics(ctx, ps.predictor, pod.GetMetrics(), reqCtx.Prompt, 1, prefixCacheScore)
		if err != nil {
			logger.V(logutil.DEBUG).Info("Skipping pod due to prediction error", "pod", pod.GetPod().String(), "error", err)
			predResult.Error = err
			predictions = append(predictions, predResult)
			continue
		}

		predResult.TTFT = prediction.TTFT
		predResult.TPOT = prediction.TPOT
		podMinTPOTSLO := 0.0
		//if pod.GetPod().RunningRequests.Peek() != nil {
		//	podMinTPOTSLO = pod.GetPod().RunningRequests.Peek().TPOT
		//}
		// Do this:
		podMinTPOTSLO = ps.getPodMinTPOTSLO(datastore, pod)
		predResult.TTFTValid, predResult.TPOTValid, predResult.IsValid, predResult.Headroom, predResult.TTFTHeadroom = ps.validatePrediction(prediction, reqCtx.SchedulingRequest, podMinTPOTSLO)

		logger.V(logutil.DEBUG).Info("Prediction for scheduling",
			"pod", pod.GetPod().String(),
			"TTFT", prediction.TTFT,
			"TPOT", prediction.TPOT,
			"buffer", SLOBufferFactor,
			"podMinTPOTSLO", podMinTPOTSLO,
			"ttftSLO", reqCtx.SchedulingRequest.TTFTSLO,
			"requestTPOTSLO", reqCtx.SchedulingRequest.AvgTPOTSLO,
			"tpotHeadroom", predResult.Headroom,
			"ttftHeadroom", predResult.TTFTHeadroom,
			"tpotValid", predResult.TPOTValid,
			"ttftValid", predResult.TTFTValid,
			"headroomStrategy", ps.headroomStrategy)

		predictions = append(predictions, predResult)
	}

	return predictions
}

func (ps *PredictionScorer) getPodMinTPOTSLO(datastore datastore.Datastore, pod schedulingtypes.Pod) float64 {
	podName := types.NamespacedName{
		Name:      pod.GetPod().NamespacedName.Name,
		Namespace: pod.GetPod().NamespacedName.Namespace,
	}
	if runningReqs, err := datastore.PodGetRunningRequests(podName); err == nil && runningReqs != nil {
		if topReq := runningReqs.Peek(); topReq != nil {
			return topReq.TPOT
		}
	}
	return 0
}

func (ps *PredictionScorer) getPodRunningRequestCount(datastore datastore.Datastore, pod schedulingtypes.Pod) int {
	podName := types.NamespacedName{
		Name:      pod.GetPod().NamespacedName.Name,
		Namespace: pod.GetPod().NamespacedName.Namespace,
	}
	if runningReqs, err := datastore.PodGetRequestCount(podName); err == nil {
		return runningReqs
	}
	return 0
}

func (ps *PredictionScorer) validatePrediction(
	pred *latencypredictor.PredictionResponse,
	req *schedulingtypes.LLMRequest,
	podMinTPOTSLO float64,
) (ttftOk, tpotOk, isValid bool, headroom float64, ttftHeadroom float64) {

	bufferedTPOT := req.AvgTPOTSLO * SLOBufferFactor
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

// updateRequestContextWithPredictions updates the request context with prediction data
func (ps *PredictionScorer) updateRequestContextWithPredictions(reqCtx *handlers.RequestContext, predictions []PodPredictionResult) {
	for _, pred := range predictions {
		if pred.Error == nil {
			reqCtx.PredictedTTFTForScheduling = append(reqCtx.PredictedTTFTForScheduling, pred.TTFT)
			reqCtx.PredictedTPOTForScheduling = append(reqCtx.PredictedTPOTForScheduling, pred.TPOT)
		}
	}
}