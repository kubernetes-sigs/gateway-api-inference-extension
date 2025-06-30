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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	k8stypes "k8s.io/apimachinery/pkg/types"
)

func TestPickMaxScorePicker(t *testing.T) {
	pod1 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}}
	pod2 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}}
	pod3 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}}

	tests := []struct {
		name   string
		picker framework.Picker
		input  []*types.ScoredPod
		output []types.Pod
	}{
		{
			name:   "Single max score",
			picker: NewMaxScorePicker(1),
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 10},
				{Pod: pod2, Score: 25},
				{Pod: pod3, Score: 15},
			},
			output: []types.Pod{
				&types.ScoredPod{Pod: pod2, Score: 25},
			},
		},
		{
			name:   "Multiple max scores, all are equally scored",
			picker: NewMaxScorePicker(2),
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 50},
				{Pod: pod2, Score: 50},
				{Pod: pod3, Score: 30},
			},
			output: []types.Pod{
				&types.ScoredPod{Pod: pod1, Score: 50},
				&types.ScoredPod{Pod: pod2, Score: 50},
			},
		},
		{
			name:   "Multiple results sorted by highest score, more pods than needed",
			picker: NewMaxScorePicker(2),
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 20},
				{Pod: pod2, Score: 25},
				{Pod: pod3, Score: 30},
			},
			output: []types.Pod{
				&types.ScoredPod{Pod: pod3, Score: 30},
				&types.ScoredPod{Pod: pod2, Score: 25},
			},
		},
		{
			name:   "Multiple results sorted by highest score, less pods than needed",
			picker: NewMaxScorePicker(4), // picker is required to return 4 pods at most, but we have only 3.
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 20},
				{Pod: pod2, Score: 25},
				{Pod: pod3, Score: 30},
			},
			output: []types.Pod{
				&types.ScoredPod{Pod: pod3, Score: 30},
				&types.ScoredPod{Pod: pod2, Score: 25},
				&types.ScoredPod{Pod: pod2, Score: 20},
			},
		},
		{
			name:   "Multiple results sorted by highest score, num of pods exactly needed",
			picker: NewMaxScorePicker(3), // picker is required to return 3 pods at most, we have only 3.
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 20},
				{Pod: pod2, Score: 25},
				{Pod: pod3, Score: 30},
			},
			output: []types.Pod{
				&types.ScoredPod{Pod: pod3, Score: 30},
				&types.ScoredPod{Pod: pod2, Score: 25},
				&types.ScoredPod{Pod: pod2, Score: 20},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := test.picker.Pick(context.Background(), types.NewCycleState(), test.input)
			got := result.TargetPods

			diff := cmp.Diff(test.output, got, cmpopts.SortSlices(func(a, b types.Pod) bool {
				return a.GetPod().NamespacedName.String() < b.GetPod().NamespacedName.String()
			}))
			if diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
