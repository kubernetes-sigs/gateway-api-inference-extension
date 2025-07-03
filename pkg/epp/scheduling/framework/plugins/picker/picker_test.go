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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	k8stypes "k8s.io/apimachinery/pkg/types"
)

func TestPickMaxScorePicker(t *testing.T) {
	tests := []struct {
		name        string
		scoredPods  []*types.ScoredPod
		wantNames   []string
		shouldPanic bool
	}{
		{
			name: "Single max score",
			scoredPods: []*types.ScoredPod{
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}}, Score: 10},
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}}, Score: 25},
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}}, Score: 15},
			},
			wantNames: []string{"pod2"},
		},
		{
			name: "Multiple max scores",
			scoredPods: []*types.ScoredPod{
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "podA"}}}, Score: 50},
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "podB"}}}, Score: 50},
				{Pod: &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "podC"}}}, Score: 30},
			},
			wantNames: []string{"podA", "podB"},
		},
		{
			name:        "Empty pod list",
			scoredPods:  []*types.ScoredPod{},
			wantNames:   nil,
			shouldPanic: true,
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

			p := NewMaxScorePicker()
			result := p.Pick(context.Background(), nil, tt.scoredPods)

			if len(tt.scoredPods) == 0 && result != nil {
				t.Errorf("expected nil result for empty input, got %+v", result)
				return
			}

			if result != nil {
				got := result.TargetPod.GetPod().NamespacedName.Name
				match := false
				for _, want := range tt.wantNames {
					if got == want {
						match = true
						break
					}
				}
				if !match {
					t.Errorf("got %q, want one of %v", got, tt.wantNames)
				}
			}
		})
	}
}
