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

package latency

import (
	"context"
	"testing"

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrlatency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
)

func makeLatencyScorerEndpoint(name string, kvCache float64, queueSize, runningReqs int) framework.Endpoint {
	return framework.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{
			KVCacheUsagePercent: kvCache,
			WaitingQueueSize:    queueSize,
			RunningRequestsSize: runningReqs,
		},
		nil,
	)
}

func setLatencyPrediction(ep framework.Endpoint, ttftValid, tpotValid bool, ttftHeadroom, tpotHeadroom, ttft, tpot float64) {
	ep.Put(attrlatency.LatencyPredictionInfoKey,
		attrlatency.NewLatencyPredictionInfo(ttftValid, tpotValid, ttftHeadroom, tpotHeadroom, ttft, tpot, 0))
}

// noExploreConfig returns default config for deterministic tests.
func noExploreConfig() Config {
	return DefaultConfig
}

func TestScorePositiveHeadroom(t *testing.T) {
	scorer := NewPlugin(noExploreConfig())

	ep1 := makeLatencyScorerEndpoint("pod1", 0.3, 0, 5)
	ep2 := makeLatencyScorerEndpoint("pod2", 0.5, 0, 8)

	// pod1: lots of headroom, pod2: less headroom. Both positive.
	setLatencyPrediction(ep1, true, true, 100, 20, 50, 10)
	setLatencyPrediction(ep2, true, true, 30, 5, 120, 25)

	endpoints := []framework.Endpoint{ep1, ep2}
	scores := scorer.Score(context.Background(), framework.NewCycleState(), nil, endpoints)

	s1, s2 := scores[ep1], scores[ep2]

	// Both should have non-zero scores (positive tier selected).
	if s1 == 0 || s2 == 0 {
		t.Fatalf("positive endpoints should have non-zero scores: pod1=%f, pod2=%f", s1, s2)
	}

	t.Logf("pod1 score=%f, pod2 score=%f", s1, s2)
}

func TestScoreNegativeOnly(t *testing.T) {
	scorer := NewPlugin(noExploreConfig())

	ep1 := makeLatencyScorerEndpoint("pod1", 0.5, 0, 5)
	ep2 := makeLatencyScorerEndpoint("pod2", 0.6, 0, 8)

	// Both negative headroom — only negative tier available.
	setLatencyPrediction(ep1, false, false, -10, -5, 110, 35)
	setLatencyPrediction(ep2, false, false, -100, -30, 200, 60)

	endpoints := []framework.Endpoint{ep1, ep2}
	scores := scorer.Score(context.Background(), framework.NewCycleState(), nil, endpoints)

	s1, s2 := scores[ep1], scores[ep2]

	// Both should have non-zero scores.
	if s1 == 0 || s2 == 0 {
		t.Fatalf("negative endpoints should have non-zero scores: pod1=%f, pod2=%f", s1, s2)
	}

	// pod1 has less violation → should score higher.
	if s1 <= s2 {
		t.Errorf("pod1 (less violation) should score higher: pod1=%f, pod2=%f", s1, s2)
	}

	t.Logf("pod1 score=%f, pod2 score=%f", s1, s2)
}

func TestScoreTierSplit(t *testing.T) {
	// The scorer expects upstream filters to have separated tiers.
	// Test with only positive endpoints (typical after tier filter selects positive).
	scorer := NewPlugin(noExploreConfig())

	ep1 := makeLatencyScorerEndpoint("ep1", 0.3, 0, 5)
	ep2 := makeLatencyScorerEndpoint("ep2", 0.5, 0, 8)

	setLatencyPrediction(ep1, true, true, 10, 2, 90, 28)
	setLatencyPrediction(ep2, true, true, 50, 10, 50, 20)

	endpoints := []framework.Endpoint{ep1, ep2}
	scores := scorer.Score(context.Background(), framework.NewCycleState(), nil, endpoints)

	s1, s2 := scores[ep1], scores[ep2]

	// Both should be scored.
	if s1 == 0 {
		t.Errorf("ep1 should have non-zero score: %f", s1)
	}
	if s2 == 0 {
		t.Errorf("ep2 should have non-zero score: %f", s2)
	}

	// ep1 has less headroom (10 vs 50), so with "least" strategy it should score higher.
	if s1 <= s2 {
		t.Errorf("ep1 (less headroom) should score higher with 'least' strategy: ep1=%f, ep2=%f", s1, s2)
	}

	t.Logf("ep1 score=%f, ep2 score=%f", s1, s2)
}

func TestScoreIdlePodPreference(t *testing.T) {
	scorer := NewPlugin(noExploreConfig())

	epBusy := makeLatencyScorerEndpoint("busy", 0.5, 0, 5)
	epIdle := makeLatencyScorerEndpoint("idle", 0.5, 0, 0) // 0 running requests

	// Same predictions — both negative, same deficit.
	// Use dispatch count for idle detection (matches EPP internal queue).
	epBusy.Put(attrlatency.LatencyPredictionInfoKey,
		attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 5))
	epIdle.Put(attrlatency.LatencyPredictionInfoKey,
		attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 0))

	endpoints := []framework.Endpoint{epBusy, epIdle}
	scores := scorer.Score(context.Background(), framework.NewCycleState(), nil, endpoints)

	sBusy, sIdle := scores[epBusy], scores[epIdle]

	// Idle pod should be strictly preferred (selected tier, busy excluded).
	if sIdle == 0 {
		t.Errorf("idle pod should have non-zero score: %f", sIdle)
	}
	if sBusy != 0 {
		t.Errorf("busy pod should have zero score when idle pods available: %f", sBusy)
	}

	t.Logf("busy score=%f, idle score=%f", sBusy, sIdle)
}

func TestScoreHierarchicalBuckets(t *testing.T) {
	scorer := NewPlugin(noExploreConfig())

	// All negative, all busy — hierarchical buckets apply.
	epBothNeg := makeLatencyScorerEndpoint("both-neg", 0.5, 0, 5)
	epTTFTNeg := makeLatencyScorerEndpoint("ttft-neg", 0.4, 0, 3)
	epTPOTNeg := makeLatencyScorerEndpoint("tpot-neg", 0.6, 0, 4)

	setLatencyPrediction(epBothNeg, false, false, -50, -10, 150, 40) // both negative
	setLatencyPrediction(epTTFTNeg, false, true, -30, 5, 130, 25)    // only TTFT negative
	setLatencyPrediction(epTPOTNeg, true, false, 10, -8, 90, 38)     // only TPOT negative

	endpoints := []framework.Endpoint{epBothNeg, epTTFTNeg, epTPOTNeg}
	scores := scorer.Score(context.Background(), framework.NewCycleState(), nil, endpoints)

	// All should have non-zero scores (they're all in the negative tier).
	for _, ep := range endpoints {
		if scores[ep] == 0 {
			t.Errorf("%s should have non-zero score", ep.GetMetadata().NamespacedName.Name)
		}
	}

	t.Logf("both-neg=%f, ttft-neg=%f, tpot-neg=%f",
		scores[epBothNeg], scores[epTTFTNeg], scores[epTPOTNeg])
}

func TestScoreCompositeFallback(t *testing.T) {
	scorer := NewPlugin(noExploreConfig())

	// No predictions set — should use composite scoring.
	ep1 := makeLatencyScorerEndpoint("pod1", 0.2, 0, 3)
	ep2 := makeLatencyScorerEndpoint("pod2", 0.8, 5, 10)

	endpoints := []framework.Endpoint{ep1, ep2}
	scores := scorer.Score(context.Background(), framework.NewCycleState(), nil, endpoints)

	s1, s2 := scores[ep1], scores[ep2]

	// pod1 has better composite metrics → should score higher.
	if s1 <= s2 {
		t.Errorf("pod1 (lower KV, no queue) should score higher: pod1=%f, pod2=%f", s1, s2)
	}

	t.Logf("pod1 score=%f, pod2 score=%f", s1, s2)
}

// Note: EpsilonExploreNeg (tier selection) is now handled by the
// slo-headroom-tier-filter, not the scorer. See filter tests.
