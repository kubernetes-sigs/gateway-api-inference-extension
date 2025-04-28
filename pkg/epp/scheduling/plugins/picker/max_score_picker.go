package picker

import (
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// MaxScorePicker picks the pod with the maximum score from the list of
// candidates.
type MaxScorePicker struct{}

var _ plugins.Picker = &MaxScorePicker{}

// Name returns the name of the picker.
func (msp *MaxScorePicker) Name() string {
	return "max_score"
}

// Pick selects the pod with the maximum score from the list of candidates.
func (msp *MaxScorePicker) Pick(ctx *types.SchedulingContext, scoredPods []*types.ScoredPod) *types.Result {
	debugLogger := ctx.Logger.V(logutil.DEBUG).WithName("max-score-picker")
	debugLogger.Info(fmt.Sprintf("Selecting the pod with the max score from %d candidates: %+v",
		len(scoredPods), scoredPods))

	winners := make([]*types.ScoredPod, 0)

	maxScore := 0.0
	for _, pod := range scoredPods {
		score := pod.Score
		if score > maxScore {
			maxScore = score
			winners = []*types.ScoredPod{pod}
		} else if score == maxScore {
			winners = append(winners, pod)
		}
	}

	if len(winners) == 0 {
		return nil
	}

	if len(winners) > 1 {
		debugLogger.Info(fmt.Sprintf("Multiple pods have the same max score (%f): %+v",
			maxScore, winners))

		randomPicker := RandomPicker{}
		return randomPicker.Pick(ctx, winners)
	}

	debugLogger.Info(fmt.Sprintf("Selected pod with max score (%f): %+v", maxScore, winners[0]))
	return &types.Result{TargetPod: winners[0]}
}
