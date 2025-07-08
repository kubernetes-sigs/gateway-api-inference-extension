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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	k8stypes "k8s.io/apimachinery/pkg/types"
)

var _ framework.Picker = &TestRandomPicker{}

// NewTestRandomPicker initializes a new NewTestRandomPicker and returns its pointer.
func NewTestRandomPicker(pickRes string) *TestRandomPicker {
	return &TestRandomPicker{
		Picked: pickRes,
		tn:     plugins.TypedName{Type: "random-test", Name: "random-test"},
	}
}

// NewTestRandomPicker picks the selected pod from the list of candidates.
type TestRandomPicker struct {
	tn     plugins.TypedName
	Picked string
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *TestRandomPicker) TypedName() plugins.TypedName {
	return p.tn
}

// WithName sets the name of the picker.
func (p *TestRandomPicker) WithName(name string) *TestRandomPicker {
	p.tn.Name = name
	return p
}

// Type returns the type of the picker.
func (p *TestRandomPicker) Type() string {
	return RandomPickerType
}

func (tp *TestRandomPicker) Pick(_ context.Context, _ *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult {
	fallbackPods := []*types.ScoredPod{}

	var winnerPod types.Pod
	for _, scoredPod := range scoredPods {
		if scoredPod.GetPod().NamespacedName.String() == tp.Picked {
			winnerPod = scoredPod
		} else {
			fallbackPods = append(fallbackPods, scoredPod)
		}
	}

	return &types.ProfileRunResult{TargetPod: winnerPod, FallbackPods: fallbackPods}
}

func TestPickMaxScorePicker(t *testing.T) {
	tests := []struct {
		name              string
		scoredPods        []*types.ScoredPod
		wantPodName       string
		wantFallbackNames []string
		shouldPanic       bool
		picker            framework.Picker
	}{
		{
			name: "Single max score",
			scoredPods: []*types.ScoredPod{
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}}, Score: 10},
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}}, Score: 25},
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}}, Score: 15},
			},
			wantPodName:       "pod2",
			wantFallbackNames: []string{"pod3", "pod1"},
			picker:            NewMaxScorePicker(),
		},
		{
			name: "Multiple max scores",
			scoredPods: []*types.ScoredPod{
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "podA"}}}, Score: 50},
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "podB"}}}, Score: 50},
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "podC"}}}, Score: 30},
			},
			wantPodName:       "podA",
			wantFallbackNames: []string{"podB", "podC"},
			picker:            NewTestRandomPicker(k8stypes.NamespacedName{Name: "podA"}.String()),
		},
		{
			name:              "Empty pod list",
			scoredPods:        []*types.ScoredPod{},
			wantFallbackNames: nil,
			shouldPanic:       true,
			picker:            NewMaxScorePicker(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.shouldPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Errorf("expected panic but did not get one")
					}
				}()
			}

			result := tt.picker.Pick(context.Background(), nil, tt.scoredPods)

			if len(tt.scoredPods) == 0 && result != nil {
				t.Errorf("expected nil result for empty input, got %+v", result)
				return
			}

			if result != nil {
				got := result.TargetPod.GetPod().NamespacedName.Name
				if diff := cmp.Diff(tt.wantPodName, got); diff != "" {
					t.Errorf("Unexpected target pod name (-want +got): %v", diff)
				}
				gotFallback := result.FallbackPods
				for i, wantName := range tt.wantFallbackNames {
					if diff := cmp.Diff(wantName, gotFallback[i].GetPod().NamespacedName.Name); diff != "" {
						t.Errorf("Unexpected target pod name (-want +got): %v", diff)
					}
				}
			}
		})
	}
}
