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

package picker

import (
	"context"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// compile-time type assertion
var _ framework.Picker = &MaxScorePicker{}

// NewMaxScorePicker initializes a new MaxScorePicker and returns its pointer.
func NewMaxScorePicker() *MaxScorePicker {
	return &MaxScorePicker{
		random: NewRandomPicker(),
	}
}

// MaxScorePicker picks the pod with the maximum score from the list of candidates.
type MaxScorePicker struct {
	random *RandomPicker
}

// Name returns the name of the picker.
func (p *MaxScorePicker) Name() string {
	return "max_score"
}

// Pick selects the pod with the maximum score from the list of candidates.
func (p *MaxScorePicker) Pick(ctx context.Context, cycleState *types.CycleState, scoredPods []*types.ScoredPod) *types.Result {
	log.FromContext(ctx).V(logutil.DEBUG).Info(fmt.Sprintf("Selecting a pod with the max score from %d candidates: %+v", len(scoredPods), scoredPods))

	highestScorePods := []*types.ScoredPod{}
	maxScore := -1.0 // pods min score is 0, putting value lower than 0 in order to find at least one pod as highest
	for _, pod := range scoredPods {
		if pod.Score > maxScore {
			maxScore = pod.Score
			highestScorePods = []*types.ScoredPod{pod}
		} else if pod.Score == maxScore {
			highestScorePods = append(highestScorePods, pod)
		}
	}

	if len(highestScorePods) > 1 {
		return p.random.Pick(ctx, cycleState, highestScorePods) // pick randomly from the highest score pods
	}

	return &types.Result{TargetPod: highestScorePods[0]}
}
