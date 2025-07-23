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
	"math/rand"
	"time"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/latencypredictorasync"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

import "os"
import "strconv"

var SLOBufferFactor = func() float64 {
	if value, exists := os.LookupEnv("SLO_BUFFER_FACTOR"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil {
			return parsedValue
		}
	}
	return 1.0 // default value
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
	predictor latencypredictor.PredictorInterface
}

// NewPredictionScorer creates a new PredictionScorer instance
func NewPredictionScorer(predictor latencypredictor.PredictorInterface) *PredictionScorer {
	return &PredictionScorer{
		predictor: predictor,
	}
}

// ScoreAndFilterPods evaluates candidate pods using latency predictions and filters them based on SLO requirements
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
		if p.IsValid {
			validPreds = append(validPreds, p)
		} else {
			invalidPreds = append(invalidPreds, p)
		}
	}
	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)
	//1) If there are *any* valid pods, give invalids exactly 1% group chance
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

	// 3) Headroom‑weighted draw among valid pods:
	//    (your existing logic)
	maxHeadroom := 0.0
	for _, p := range validPreds {
		if p.Headroom > maxHeadroom {
			maxHeadroom = p.Headroom
		}
	}
	const W_max = 100
	sf := 1.0
	if maxHeadroom > 0 {
		sf = float64(W_max-1) / maxHeadroom
	}

	// Build and draw weighted choices
	total := 0
	choices := make([]Choice, 0, len(validPreds))
	for _, p := range validPreds {
		w := int((maxHeadroom-p.Headroom)*sf) + 1
		choices = append(choices, Choice{PodName: p.Pod, Weight: w})
		total += w
	}

	idx := r.Intn(total)
	for _, c := range choices {
		if idx < c.Weight {
			return c.PodName, nil
		}
		idx -= c.Weight
	}

	// fallback (shouldn’t happen)
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
		podName := types.NamespacedName{
			Name:      pod.GetPod().NamespacedName.Name,
			Namespace: pod.GetPod().NamespacedName.Namespace,
		}
		if runningReqs, err := datastore.PodGetRunningRequests(podName); err == nil && runningReqs != nil {
			if topReq := runningReqs.Peek(); topReq != nil {
				podMinTPOTSLO = topReq.TPOT
			}
		}
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
			"ttftValid", predResult.TTFTValid)

		predictions = append(predictions, predResult)
	}

	return predictions
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
