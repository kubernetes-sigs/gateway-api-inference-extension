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

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrlatency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func makeEndpoint(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{},
		nil,
	)
}

func setPrefixScore(ep fwksched.Endpoint, matchBlocks, totalBlocks int) {
	ep.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(matchBlocks, totalBlocks, 16))
}

func TestAffinityWeightedPickerGlobalGate(t *testing.T) {
	config := defaultAffinityWeightedPickerConfig
	config.EpsilonExplore = 0

	picker := NewAffinityWeightedPicker(config)

	ep1 := makeEndpoint("sticky")
	ep2 := makeEndpoint("other")
	setPrefixScore(ep1, 100, 100) // 1.0 >= 0.99
	setPrefixScore(ep2, 50, 100)  // 0.5 < 0.99

	scored := []*fwksched.ScoredEndpoint{
		{Endpoint: ep1, Score: 0.5},
		{Endpoint: ep2, Score: 0.8},
	}

	for range 50 {
		result := picker.Pick(context.Background(), fwksched.NewCycleState(), scored)
		if len(result.TargetEndpoints) != 1 {
			t.Fatalf("expected 1 endpoint, got %d", len(result.TargetEndpoints))
		}
		name := result.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		if name != "sticky" {
			t.Fatalf("global gate should stick to 'sticky', got %q", name)
		}
	}
}

func TestAffinityWeightedPickerLocalGateFallback(t *testing.T) {
	config := defaultAffinityWeightedPickerConfig
	config.EpsilonExplore = 0

	picker := NewAffinityWeightedPicker(config)

	ep1 := makeEndpoint("local-sticky")
	ep2 := makeEndpoint("other")
	setPrefixScore(ep1, 85, 100) // 0.85 >= 0.80, < 0.99
	setPrefixScore(ep2, 30, 100) // 0.30 < 0.80

	scored := []*fwksched.ScoredEndpoint{
		{Endpoint: ep1, Score: 0.3},
		{Endpoint: ep2, Score: 0.9},
	}

	for range 50 {
		result := picker.Pick(context.Background(), fwksched.NewCycleState(), scored)
		name := result.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		if name != "local-sticky" {
			t.Fatalf("local gate should stick to 'local-sticky', got %q", name)
		}
	}
}

func TestAffinityWeightedPickerNoGate(t *testing.T) {
	config := defaultAffinityWeightedPickerConfig
	config.EpsilonExplore = 0

	picker := NewAffinityWeightedPicker(config)

	ep1 := makeEndpoint("pod1")
	ep2 := makeEndpoint("pod2")
	setPrefixScore(ep1, 50, 100) // 0.5 < 0.80
	setPrefixScore(ep2, 30, 100) // 0.3 < 0.80

	scored := []*fwksched.ScoredEndpoint{
		{Endpoint: ep1, Score: 0.9},
		{Endpoint: ep2, Score: 0.1},
	}

	counts := map[string]int{}
	for range 1000 {
		result := picker.Pick(context.Background(), fwksched.NewCycleState(), scored)
		name := result.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		counts[name]++
	}

	if counts["pod1"] == 0 || counts["pod2"] == 0 {
		t.Fatalf("both pods should be picked when no gate applies: %v", counts)
	}
	if counts["pod1"] < counts["pod2"] {
		t.Errorf("pod1 (score=0.9) should be picked more often than pod2 (score=0.1): %v", counts)
	}
}

func TestAffinityWeightedPickerEpsilonExplore(t *testing.T) {
	config := defaultAffinityWeightedPickerConfig
	config.EpsilonExplore = 1.0

	picker := NewAffinityWeightedPicker(config)

	ep1 := makeEndpoint("sticky")
	ep2 := makeEndpoint("other")
	setPrefixScore(ep1, 100, 100)
	setPrefixScore(ep2, 0, 100)

	scored := []*fwksched.ScoredEndpoint{
		{Endpoint: ep1, Score: 0.1},
		{Endpoint: ep2, Score: 0.9},
	}

	counts := map[string]int{}
	for range 1000 {
		result := picker.Pick(context.Background(), fwksched.NewCycleState(), scored)
		name := result.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		counts[name]++
	}

	if counts["other"] < counts["sticky"] {
		t.Errorf("with epsilon=1.0, higher-scored 'other' should dominate: %v", counts)
	}
}

func TestAffinityWeightedPickerMaxMode(t *testing.T) {
	config := defaultAffinityWeightedPickerConfig
	config.EpsilonExplore = 0
	config.SelectionMode = "max"

	picker := NewAffinityWeightedPicker(config)

	ep1 := makeEndpoint("low")
	ep2 := makeEndpoint("high")
	ep3 := makeEndpoint("sticky")
	setPrefixScore(ep1, 10, 100) // 0.1
	setPrefixScore(ep2, 10, 100) // 0.1
	setPrefixScore(ep3, 90, 100) // 0.9 >= 0.80 — sticky

	scored := []*fwksched.ScoredEndpoint{
		{Endpoint: ep1, Score: 0.3},
		{Endpoint: ep2, Score: 0.9}, // highest score but not sticky
		{Endpoint: ep3, Score: 0.5}, // sticky via local gate
	}

	// Local gate filters to ep3, then max mode picks it (only candidate).
	for range 50 {
		result := picker.Pick(context.Background(), fwksched.NewCycleState(), scored)
		name := result.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		if name != "sticky" {
			t.Fatalf("max mode with local gate should pick 'sticky', got %q", name)
		}
	}
}

func TestAffinityWeightedPickerMaxModeNoGate(t *testing.T) {
	config := defaultAffinityWeightedPickerConfig
	config.EpsilonExplore = 0
	config.SelectionMode = "max"
	config.GlobalTau = 0 // disable
	config.LocalTau = 0  // disable

	picker := NewAffinityWeightedPicker(config)

	ep1 := makeEndpoint("low")
	ep2 := makeEndpoint("high")

	scored := []*fwksched.ScoredEndpoint{
		{Endpoint: ep1, Score: 0.3},
		{Endpoint: ep2, Score: 0.9},
	}

	// No gate, max mode — always picks highest score.
	for range 50 {
		result := picker.Pick(context.Background(), fwksched.NewCycleState(), scored)
		name := result.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		if name != "high" {
			t.Fatalf("max mode should pick 'high' (score=0.9), got %q", name)
		}
	}
}

func TestAffinityWeightedPickerTTFTLoadGate(t *testing.T) {
	// Sticky endpoint has much worse TTFT — load gate should break stickiness.
	config := defaultAffinityWeightedPickerConfig
	config.EpsilonExplore = 0
	config.MaxTTFTPenaltyMs = 100

	picker := NewAffinityWeightedPicker(config)

	ep1 := makeEndpoint("sticky")
	ep2 := makeEndpoint("fast")
	setPrefixScore(ep1, 100, 100) // 1.0 — perfect match
	setPrefixScore(ep2, 10, 100)  // 0.1

	ep1.Put(attrlatency.LatencyPredictionInfoKey,
		attrlatency.NewLatencyPredictionInfo(true, true, 0, 0, 500, 0)) // slow
	ep2.Put(attrlatency.LatencyPredictionInfoKey,
		attrlatency.NewLatencyPredictionInfo(true, true, 0, 0, 50, 0)) // fast — penalty 450 > 100

	scored := []*fwksched.ScoredEndpoint{
		{Endpoint: ep1, Score: 0.5},
		{Endpoint: ep2, Score: 0.8},
	}

	counts := map[string]int{}
	for range 1000 {
		result := picker.Pick(context.Background(), fwksched.NewCycleState(), scored)
		name := result.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		counts[name]++
	}

	if counts["fast"] == 0 {
		t.Errorf("TTFT load gate should allow 'fast' endpoint to be selected: %v", counts)
	}
}

func TestAffinityWeightedPickerTTFTLoadGateSkippedWithoutPredictions(t *testing.T) {
	// No LatencyPredictionInfo — load gate should be skipped, stickiness holds.
	config := defaultAffinityWeightedPickerConfig
	config.EpsilonExplore = 0
	config.MaxTTFTPenaltyMs = 100

	picker := NewAffinityWeightedPicker(config)

	ep1 := makeEndpoint("sticky")
	ep2 := makeEndpoint("other")
	setPrefixScore(ep1, 100, 100)
	setPrefixScore(ep2, 10, 100)
	// No LatencyPredictionInfo set on either endpoint.

	scored := []*fwksched.ScoredEndpoint{
		{Endpoint: ep1, Score: 0.5},
		{Endpoint: ep2, Score: 0.8},
	}

	for range 50 {
		result := picker.Pick(context.Background(), fwksched.NewCycleState(), scored)
		name := result.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		if name != "sticky" {
			t.Fatalf("without predictions, load gate should be skipped, sticky should hold, got %q", name)
		}
	}
}

func TestAffinityWeightedPickerNoPrefixAttributes(t *testing.T) {
	config := defaultAffinityWeightedPickerConfig
	config.EpsilonExplore = 0

	picker := NewAffinityWeightedPicker(config)

	// No prefix attributes set — gates should not apply.
	ep1 := makeEndpoint("pod1")
	ep2 := makeEndpoint("pod2")

	scored := []*fwksched.ScoredEndpoint{
		{Endpoint: ep1, Score: 0.9},
		{Endpoint: ep2, Score: 0.1},
	}

	counts := map[string]int{}
	for range 1000 {
		result := picker.Pick(context.Background(), fwksched.NewCycleState(), scored)
		name := result.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		counts[name]++
	}

	if counts["pod1"] == 0 || counts["pod2"] == 0 {
		t.Fatalf("both pods should be selectable without prefix attributes: %v", counts)
	}
}
