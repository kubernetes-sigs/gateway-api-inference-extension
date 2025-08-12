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
	Pod       schedulingtypes.Pod
	TTFT      float64
	TPOT      float64
	TTFTValid bool
	TPOTValid bool
	IsValid   bool
	Error     error
	Headroom  float64 // Headroom for the pod, if applicable
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
	if reqCtx.SchedulingRequest.TTFTSLO == 0 || reqCtx.SchedulingRequest.AvgTPOTSLO == 0 {
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

	//1) If there are *any* valid pods, give invalids exactly 0.1% group chance
	if len(validPreds) > 0 && len(invalidPreds) > 0 {
		if r.Float64() < 0.001 {
			// pick one invalid at uniform random
			i := r.Intn(len(invalidPreds))
			return invalidPreds[i].Pod, nil
		}
	}

	// 2) Otherwise, if no valid pods, fallback for critical vs non‑critical
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

	// 3) Headroom-weighted draw among valid pods with configurable strategy:
	var posHeadroomPods, negHeadroomPods []PodPredictionResult
	for _, p := range validPreds {
		if p.Headroom > 0 {
			posHeadroomPods = append(posHeadroomPods, p)
		} else {
			negHeadroomPods = append(negHeadroomPods, p)
		}
	}

	const W_max = 100
	const minWeightForNegative = 1 // Minimal weight for scale-to-zero
	total := 0
	choices := make([]Choice, 0, len(validPreds))

	// Handle positive headroom pods with configurable strategy
	if len(posHeadroomPods) > 0 {
		minPosHeadroom := math.MaxFloat64
		maxPosHeadroom := -math.MaxFloat64

		for _, p := range posHeadroomPods {
			if p.Headroom < minPosHeadroom {
				minPosHeadroom = p.Headroom
			}
			if p.Headroom > maxPosHeadroom {
				maxPosHeadroom = p.Headroom
			}
		}

		sf := 1.0
		posHeadroomRange := maxPosHeadroom - minPosHeadroom
		if posHeadroomRange > 0 {
			sf = float64(W_max-minWeightForNegative) / posHeadroomRange
		}

		// Apply strategy-based weighting
		for _, p := range posHeadroomPods {
			var w int
			switch ps.headroomStrategy {
			case HeadroomStrategyLeast:
				// INVERTED weighting: less headroom = higher weight (better packing)
				w = int((maxPosHeadroom-p.Headroom)*sf) + minWeightForNegative + 1
			case HeadroomStrategyMost:
				// DIRECT weighting: more headroom = higher weight (more conservative)
				w = int((p.Headroom-minPosHeadroom)*sf) + minWeightForNegative + 1
			default:
				// Fallback to least strategy
				w = int((maxPosHeadroom-p.Headroom)*sf) + minWeightForNegative + 1
			}
			
			choices = append(choices, Choice{PodName: p.Pod, Weight: w})
			total += w
		}

		logger.V(logutil.DEBUG).Info("Applied headroom weighting strategy",
			"strategy", ps.headroomStrategy,
			"positivePods", len(posHeadroomPods),
			"minHeadroom", minPosHeadroom,
			"maxHeadroom", maxPosHeadroom)
	}

	// Handle negative headroom pods: minimal weight for scale-to-zero
	for _, p := range negHeadroomPods {
		choices = append(choices, Choice{PodName: p.Pod, Weight: minWeightForNegative})
		total += minWeightForNegative
	}

	// Select pod using weighted random selection
	idx := r.Intn(total)
	for _, c := range choices {
		if idx < c.Weight {
			return c.PodName, nil
		}
		idx -= c.Weight
	}

	// fallback (shouldn't happen)
	return validPreds[0].Pod, nil
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
		predResult.TTFTValid, predResult.TPOTValid, predResult.IsValid, predResult.Headroom = ps.validatePrediction(prediction, reqCtx.SchedulingRequest, podMinTPOTSLO)

		logger.V(logutil.DEBUG).Info("Prediction for scheduling",
			"pod", pod.GetPod().String(),
			"TTFT", prediction.TTFT,
			"TPOT", prediction.TPOT,
			"buffer", SLOBufferFactor,
			"podMinTPOTSLO", podMinTPOTSLO,
			"ttftSLO", reqCtx.SchedulingRequest.TTFTSLO,
			"requestTPOTSLO", reqCtx.SchedulingRequest.AvgTPOTSLO,
			"headroom", predResult.Headroom,
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
) (ttftOk, tpotOk, isValid bool, headroom float64) {

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