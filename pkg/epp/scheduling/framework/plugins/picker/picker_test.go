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

package picker_test

import (
	"context"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"testing"

	k8stypes "k8s.io/apimachinery/pkg/types"
)

// Picks the pod with the highest unique score.
func TestPickMaxScorePicker_SingleMax(t *testing.T) {
	p := picker.NewMaxScorePicker()
	ctx := context.Background()

	scoredPods := []*types.ScoredPod{
		{
			Pod: &types.PodMetrics{
				Pod: &backend.Pod{
					NamespacedName: k8stypes.NamespacedName{Name: "pod1"},
				},
			},
			Score: 10,
		},
		{
			Pod: &types.PodMetrics{
				Pod: &backend.Pod{
					NamespacedName: k8stypes.NamespacedName{Name: "pod2"},
				},
			},
			Score: 25,
		},
		{
			Pod: &types.PodMetrics{
				Pod: &backend.Pod{
					NamespacedName: k8stypes.NamespacedName{Name: "pod3"},
				},
			},
			Score: 15,
		},
	}

	result := p.Pick(ctx, nil, scoredPods)

	if result == nil {
		t.Fatal("expected a result but got nil")
	}

	picked := result.TargetPod.GetPod().NamespacedName.Name
	if picked != "pod2" {
		t.Errorf("expected pod2, but got %s", picked)
	}
}

// Picks randomly when multiple pods share top score.
func TestPickMaxScorePicker_MultipleMax(t *testing.T) {
	p := picker.NewMaxScorePicker()
	ctx := context.Background()

	scoredPods := []*types.ScoredPod{
		{
			Pod: &types.PodMetrics{
				Pod: &backend.Pod{
					NamespacedName: k8stypes.NamespacedName{Name: "podA"},
				},
			},
			Score: 50,
		},
		{
			Pod: &types.PodMetrics{
				Pod: &backend.Pod{
					NamespacedName: k8stypes.NamespacedName{Name: "podB"},
				},
			},
			Score: 50,
		},
		{
			Pod: &types.PodMetrics{
				Pod: &backend.Pod{
					NamespacedName: k8stypes.NamespacedName{Name: "podC"},
				},
			},
			Score: 30,
		},
	}

	result := p.Pick(ctx, nil, scoredPods)

	if result == nil {
		t.Fatal("expected a result but got nil")
	}

	picked := result.TargetPod.GetPod().NamespacedName.Name
	if picked != "podA" && picked != "podB" {
		t.Errorf("expected podA or podB, but got %s", picked)
	}
}

// Returns nil or panics on empty pod list.
func TestPickMaxScorePicker_EmptyList(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Logf("plugin panicked as expected for empty input: %v", r)
		}
	}()

	p := picker.NewMaxScorePicker()
	ctx := context.Background()

	var scoredPods []*types.ScoredPod

	result := p.Pick(ctx, nil, scoredPods)
	if result != nil {
		t.Errorf("expected nil result for empty input, got %+v", result)
	}
}
