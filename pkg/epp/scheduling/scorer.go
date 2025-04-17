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

package scheduling

import (
	"errors"
	"math/rand/v2"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

type PodScore struct {
	Score float64
	Pod   *types.PodMetrics
}

// Scorer is the interface that scorers must implement
type Scorer interface {
	ScoreTargets(ctx *types.Context, pods []*types.PodMetrics) ([]PodScore, error)
}

// Scorer is the interface that scorers must implement
type ScorerMng struct {
	scorers []Scorer
}

func NewScorerMng() *ScorerMng {
	return &ScorerMng{
		scorers: make([]Scorer, 0),
	}
}

func (sm *ScorerMng) addScorer(scorer Scorer) {
	sm.scorers = append(sm.scorers, scorer)
}

func (sm *ScorerMng) scoreTargets(ctx *types.Context, pods []*types.PodMetrics) (*types.PodMetrics, error) {
	logger := log.FromContext(ctx)

	podsTotalScore := make(map[*types.PodMetrics]float64)
	validPods := make([]*types.PodMetrics, 0)

	// initialize zero score for all pods + check that pods are valid
	for _, pod := range pods {
		if pod == nil || pod.Pod == nil || pod.Metrics == nil {
			logger.Info("Invalid/empty pod skipped in scoring process")
		} else {
			validPods = append(validPods, pod)
			podsTotalScore[pod] = 0.0
		}
	}

	if len(validPods) == 0 {
		return nil, errors.New("Empty list of valid pods to score")
	}

	// add scores from all scorers
	for _, scorer := range sm.scorers {
		scoredPods, err := scorer.ScoreTargets(ctx, validPods)
		if err != nil {
			// in case scorer failed - don't use it in the total score, but continue to other scorers
			logger.Error(err, "Score targets returned error in scorer")
		} else {
			for _, scoredPod := range scoredPods {
				podsTotalScore[scoredPod.Pod] += scoredPod.Score
			}
		}
	}

	// select pod with maximum score, if more than one with the max score - use random pods from the list
	var highestScoreTargets []*types.PodMetrics
	// score weights cound be negative
	maxScore := 0.0
	isFirst := true

	for pod, score := range podsTotalScore {
		if isFirst {
			maxScore = score
			highestScoreTargets = []*types.PodMetrics{pod}
		} else {
			if score > maxScore {
				maxScore = score
				highestScoreTargets = []*types.PodMetrics{pod}
			} else if score == maxScore {
				highestScoreTargets = append(highestScoreTargets, pod)
			}
		}
	}

	// single pod with max score
	if len(highestScoreTargets) == 1 {
		return highestScoreTargets[0], nil
	}

	// select random pod from list of pods with max score
	return highestScoreTargets[rand.IntN(len(highestScoreTargets))], nil
}
