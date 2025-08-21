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

package scorer

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/latencypredictorasync"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	requestcontrol "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	SLOScorerPluginType = "slo-scorer"
	MinScore            = 0
	MaxScore            = 100
)

var SLOBufferFactor = func() float64 {
	if value, exists := os.LookupEnv("SLO_BUFFER_FACTOR"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil {
			return parsedValue
		}
	}
	return 1.0 // default value
}()

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

type SLOScorer struct {
	tn        plugins.TypedName
	predictor latencypredictor.PredictorInterface
	datastore datastore.Datastore
}

var _ framework.Scorer = &SLOScorer{}

// SLOScorerFactory defines the factory function for SLOScorer.
func SLOScorerFactory(name string, predictor latencypredictor.PredictorInterface, datastore datastore.Datastore, _ plugins.Handle) (plugins.Plugin, error) {
	return NewSLOScorer(predictor, datastore).WithName(name), nil
}

func NewSLOScorer(predictor latencypredictor.PredictorInterface, datastore datastore.Datastore) *SLOScorer {
	return &SLOScorer{
		tn:        plugins.TypedName{Type: SLOScorerPluginType, Name: SLOScorerPluginType},
		predictor: predictor,
		datastore: datastore,
	}
}

func (s *SLOScorer) TypedName() plugins.TypedName {
	return s.tn
}

func (s *SLOScorer) WithName(name string) *SLOScorer {
	s.tn.Name = name
	return s
}

func (s *SLOScorer) Score(ctx context.Context, state *schedulingtypes.CycleState, request *schedulingtypes.LLMRequest, pods []schedulingtypes.Pod) map[schedulingtypes.Pod]float64 {
	logger := log.FromContext(ctx)
	predictions := s.generatePredictions(ctx, state, request, pods)

	scores := make(map[schedulingtypes.Pod]float64, len(pods))
	var validPreds, invalidPreds []PodPredictionResult
	for _, p := range predictions {
		if p.Error != nil {
			invalidPreds = append(invalidPreds, p)
			continue
		}
		// A pod is valid if the prediction is valid OR if it's idle (scale-to-zero)
		if p.IsValid || s.getPodRunningRequestCount(p.Pod) == 0 {
			validPreds = append(validPreds, p)
		} else {
			invalidPreds = append(invalidPreds, p)
		}
	}

	for _, p := range invalidPreds {
		scores[p.Pod] = MinScore
	}

	var posHeadroomPods, negHeadroomPods []PodPredictionResult
	for _, p := range validPreds {
		if p.Headroom > 0 {
			posHeadroomPods = append(posHeadroomPods, p)
		} else {
			negHeadroomPods = append(negHeadroomPods, p)
		}
	}

	// Handle positive headroom pods: pack pods with LESS headroom first
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

		posHeadroomRange := maxPosHeadroom - minPosHeadroom
		for _, p := range posHeadroomPods {
			// INVERTED weighting: less headroom = higher score (better packing)
			score := float64(MaxScore)
			if posHeadroomRange > 0 {
				// Normalize score between 1 and MaxScore
				score = ((maxPosHeadroom - p.Headroom) / posHeadroomRange * (MaxScore - 1)) + 1
			}
			scores[p.Pod] = math.Round(score)
		}
	}

	// Handle negative headroom pods: minimal weight for scale-to-zero
	for _, p := range negHeadroomPods {
		scores[p.Pod] = 1
	}

	logger.V(logutil.DEBUG).Info("SLO-based scores calculated", "scores", scores)
	return scores
}

func (s *SLOScorer) generatePredictions(ctx context.Context, state *schedulingtypes.CycleState, request *schedulingtypes.LLMRequest, candidatePods []schedulingtypes.Pod) []PodPredictionResult {
	logger := log.FromContext(ctx)
	predictions := make([]PodPredictionResult, 0, len(candidatePods))

	for _, pod := range candidatePods {
		predResult := PodPredictionResult{Pod: pod}

		logger.V(logutil.TRACE).Info("Candidate pod for scoring", "pod", pod.GetPod().String(), "metrics", pod.GetMetrics().String())

		// Get prefix cache score for the pod
		prefixCacheScore := s.getPrefixCacheScoreForPod(ctx, state, pod)

		// Generate prediction
		prediction, err := requestcontrol.PredictWithMetrics(ctx, s.predictor, pod.GetMetrics(), request.Prompt, 1, prefixCacheScore)
		if err != nil {
			logger.V(logutil.DEBUG).Info("Skipping pod due to prediction error", "pod", pod.GetPod().String(), "error", err)
			predResult.Error = err
			predictions = append(predictions, predResult)
			continue
		}

		predResult.TTFT = prediction.TTFT
		predResult.TPOT = prediction.TPOT
		podMinTPOTSLO := s.getPodMinTPOTSLO(pod)
		predResult.TTFTValid, predResult.TPOTValid, predResult.IsValid, predResult.Headroom = s.validatePrediction(prediction, request, podMinTPOTSLO)

		logger.V(logutil.DEBUG).Info("Prediction for scoring",
			"pod", pod.GetPod().String(),
			"TTFT", prediction.TTFT,
			"TPOT", prediction.TPOT,
			"buffer", SLOBufferFactor,
			"podMinTPOTSLO", podMinTPOTSLO,
			"ttftSLO", request.TTFTSLO,
			"requestTPOTSLO", request.AvgTPOTSLO,
			"headroom", predResult.Headroom,
			"tpotValid", predResult.TPOTValid,
			"ttftValid", predResult.TTFTValid)

		predictions = append(predictions, predResult)
	}

	return predictions
}

func (s *SLOScorer) getPodMinTPOTSLO(pod schedulingtypes.Pod) float64 {
	podName := types.NamespacedName{
		Name:      pod.GetPod().NamespacedName.Name,
		Namespace: pod.GetPod().NamespacedName.Namespace,
	}
	if runningReqs, err := s.datastore.PodGetRunningRequests(podName); err == nil && runningReqs != nil {
		if topReq := runningReqs.Peek(); topReq != nil {
			return topReq.TPOT
		}
	}
	return 0
}

func (s *SLOScorer) getPodRunningRequestCount(pod schedulingtypes.Pod) int {
	podName := types.NamespacedName{
		Name:      pod.GetPod().NamespacedName.Name,
		Namespace: pod.GetPod().NamespacedName.Namespace,
	}
	if runningReqs, err := s.datastore.PodGetRequestCount(podName); err == nil {
		return runningReqs
	}
	return 0
}

func (s *SLOScorer) validatePrediction(
	pred *latencypredictor.PredictionResponse,
	req *schedulingtypes.LLMRequest,
	podMinTPOTSLO float64,
) (ttftOk, tpotOk, isValid bool, headroom float64) {

	bufferedTPOT := req.AvgTPOTSLO * SLOBufferFactor
	if podMinTPOTSLO > 0 {
		bufferedTPOT = math.Min(bufferedTPOT, podMinTPOTSLO*SLOBufferFactor)
	}
	tpotOk = pred.TPOT < bufferedTPOT
	ttftOk = pred.TTFT < req.TTFTSLO

	isValid = ttftOk && tpotOk
	headroom = bufferedTPOT - pred.TPOT
	return
}

func (s *SLOScorer) getPrefixCacheScoreForPod(ctx context.Context, cycleState *schedulingtypes.CycleState, pod schedulingtypes.Pod) float64 {
	stateData, err := cycleState.Read(prefix.PrefixCachePluginType)
	if err != nil {
		// The prefix cache plugin might not be enabled, which is a valid scenario.
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
		return 0.0
	}

	matchLen := prefixCacheState.PrefixCacheServers[prefix.ServerID(pod.GetPod().NamespacedName)]
	return float64(matchLen) / float64(total)
}
