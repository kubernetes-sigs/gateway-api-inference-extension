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
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	stdpicker "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/picker"
)

// TestPipelineEquivalence verifies the Plugin + WeightedRandomPicker pipeline
// produces correct endpoint selection behavior.
//
// The pipeline:
//   1. Plugin scores all endpoints proportionally by headroom
//   2. WeightedRandomPicker selects using weighted random sampling from scores
//
// Key invariants that must hold:
//   - When all headroom is positive, endpoints with tighter packing get higher scores
//   - When all headroom is negative, idle pods are strictly preferred
//   - When mixed, positive tier dominates (unless epsilon explore triggers)
//   - Composite fallback works when no predictions exist

func ep(name string, kvCache float64, queue, running int) framework.Endpoint {
	return framework.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{
			KVCacheUsagePercent: kvCache,
			WaitingQueueSize:    queue,
			RunningRequestsSize: running,
		},
		nil,
	)
}

func setLatency(ep framework.Endpoint, ttftValid, tpotValid bool, ttftH, tpotH, ttft, tpot float64) {
	ep.Put(attrlatency.LatencyPredictionInfoKey,
		attrlatency.NewLatencyPredictionInfo(ttftValid, tpotValid, ttftH, tpotH, ttft, tpot))
}

func setLatencyWithDispatch(ep framework.Endpoint, ttftValid, tpotValid bool, ttftH, tpotH, ttft, tpot float64, dispatchedCount int) {
	ep.Put(attrlatency.LatencyPredictionInfoKey,
		attrlatency.NewLatencyPredictionInfoWithDispatch(ttftValid, tpotValid, ttftH, tpotH, ttft, tpot, dispatchedCount))
}

func TestPipelinePositiveTierSelectsTighterPacking(t *testing.T) {
	// With "least" headroom strategy, tighter packing is preferred.
	// (1-combined)*wMax gives higher weight to tighter headroom.
	// Verify: endpoint with less headroom gets a higher score.

	s := NewLatencyScorer(noExploreConfig())

	tight := ep("tight", 0.3, 0, 5)
	loose := ep("loose", 0.5, 0, 8)

	setLatency(tight, true, true, 30, 5, 120, 25)  // less headroom
	setLatency(loose, true, true, 100, 20, 50, 10) // more headroom

	scores := s.Score(context.Background(), framework.NewCycleState(), nil, []framework.Endpoint{tight, loose})

	if scores[tight] <= scores[loose] {
		t.Errorf("tighter packing should score higher: tight=%f, loose=%f", scores[tight], scores[loose])
	}
}

func TestPipelineNegativeTierPrefersIdlePods(t *testing.T) {
	// Strict idle-pod preference in negative tier.
	// Verify: idle pod gets non-zero score, busy pod gets zero.

	s := NewLatencyScorer(noExploreConfig())

	busy := ep("busy", 0.5, 0, 5)
	idle := ep("idle", 0.5, 0, 0)

	// Use dispatch count (from EPP's internal queue) for idle detection
	setLatencyWithDispatch(busy, false, false, -50, -10, 150, 40, 5)
	setLatencyWithDispatch(idle, false, false, -50, -10, 150, 40, 0)

	scores := s.Score(context.Background(), framework.NewCycleState(), nil, []framework.Endpoint{busy, idle})

	if scores[idle] == 0 {
		t.Error("idle pod should have non-zero score in negative tier")
	}
	if scores[busy] != 0 {
		t.Errorf("busy pod should have zero score when idle pods available: %f", scores[busy])
	}
}

func TestPipelinePositiveTierDominatesOverNegative(t *testing.T) {
	// With EpsilonExploreNeg=0, positive tier is always selected.
	// Verify: positive endpoint scores non-zero, negative scores zero.

	s := NewLatencyScorer(noExploreConfig())

	pos := ep("pos", 0.3, 0, 5)
	neg := ep("neg", 0.5, 0, 8)

	setLatency(pos, true, true, 50, 10, 50, 20)
	setLatency(neg, false, false, -20, -5, 120, 35)

	scores := s.Score(context.Background(), framework.NewCycleState(), nil, []framework.Endpoint{pos, neg})

	if scores[pos] == 0 {
		t.Error("positive endpoint should have non-zero score")
	}
	if scores[neg] != 0 {
		t.Errorf("negative endpoint should have zero score: %f", scores[neg])
	}
}

func TestPipelineCompositeFallback(t *testing.T) {
	// Falls back to composite scoring when no predictions are available.
	// Verify: endpoint with better metrics scores higher.

	s := NewLatencyScorer(noExploreConfig())

	good := ep("good", 0.2, 0, 3) // low KV, no queue
	bad := ep("bad", 0.8, 5, 10)  // high KV, queued

	scores := s.Score(context.Background(), framework.NewCycleState(), nil, []framework.Endpoint{good, bad})

	if scores[good] <= scores[bad] {
		t.Errorf("better metrics should score higher: good=%f, bad=%f", scores[good], scores[bad])
	}
}

func TestPipelineEndToEndWithPicker(t *testing.T) {
	// Full pipeline: scorer + picker. With deterministic config (no exploration),
	// the picker should select the highest-scored endpoint.

	s := NewLatencyScorer(noExploreConfig())
	p := stdpicker.NewWeightedRandomPicker(1)

	strong := ep("strong", 0.3, 0, 5)
	weak := ep("weak", 0.5, 0, 8)

	setLatency(strong, true, true, 30, 5, 120, 25) // tighter → higher score
	setLatency(weak, true, true, 100, 20, 50, 10)  // looser → lower score

	endpoints := []framework.Endpoint{strong, weak}
	scores := s.Score(context.Background(), framework.NewCycleState(), nil, endpoints)

	// Build scored endpoints for picker
	scored := make([]*framework.ScoredEndpoint, len(endpoints))
	for i, ep := range endpoints {
		scored[i] = &framework.ScoredEndpoint{Endpoint: ep, Score: scores[ep]}
	}

	// Run picker many times — strong should be picked more often
	strongCount := 0
	for range 200 {
		result := p.Pick(context.Background(), framework.NewCycleState(), scored)
		if result.TargetEndpoints[0].GetMetadata().NamespacedName.Name == "strong" {
			strongCount++
		}
	}

	// Strong has higher score → should be picked majority of the time
	if strongCount < 100 {
		t.Errorf("higher-scored endpoint should be picked most often: strong picked %d/200", strongCount)
	}
	t.Logf("strong picked %d/200, scores: strong=%f, weak=%f", strongCount, scores[strong], scores[weak])
}

func TestPipelineNegativeDeficitOrdering(t *testing.T) {
	// Endpoints with less SLO violation should score higher in negative tier.

	s := NewLatencyScorer(noExploreConfig())

	mild := ep("mild", 0.5, 0, 5)
	severe := ep("severe", 0.6, 0, 8)

	setLatency(mild, false, false, -10, -5, 110, 35)     // small violation
	setLatency(severe, false, false, -100, -30, 200, 60) // large violation

	scores := s.Score(context.Background(), framework.NewCycleState(), nil, []framework.Endpoint{mild, severe})

	if scores[mild] <= scores[severe] {
		t.Errorf("less violation should score higher: mild=%f, severe=%f", scores[mild], scores[severe])
	}
}

func TestPipelineAttributeDataFlow(t *testing.T) {
	// Verify the scorer correctly reads predictions from endpoint attributes
	// (as written by the predictor's PrepareRequestData).
	// This tests the data contract between predictor and scorer.

	s := NewLatencyScorer(noExploreConfig())

	ep1 := ep("pod1", 0.3, 0, 5)
	ep2 := ep("pod2", 0.5, 0, 8)

	// Simulate what predictor.PrepareRequestData writes to attributes:
	// valid predictions with positive headroom
	ep1.Put(attrlatency.LatencyPredictionInfoKey,
		attrlatency.NewLatencyPredictionInfo(true, true, 50, 10, 50, 20))
	ep2.Put(attrlatency.LatencyPredictionInfoKey,
		attrlatency.NewLatencyPredictionInfo(true, true, 100, 30, 30, 10))

	endpoints := []framework.Endpoint{ep1, ep2}
	scores := s.Score(context.Background(), framework.NewCycleState(), nil, endpoints)

	// Both should have non-zero scores (predictions present, positive tier)
	if scores[ep1] == 0 || scores[ep2] == 0 {
		t.Fatalf("endpoints with predictions should have non-zero scores: ep1=%f, ep2=%f", scores[ep1], scores[ep2])
	}

	// Now test with NO attributes — should fall back to composite
	ep3 := ep("pod3", 0.2, 0, 3)
	ep4 := ep("pod4", 0.8, 5, 10)
	// No Put() calls — no attributes set

	endpoints2 := []framework.Endpoint{ep3, ep4}
	scores2 := s.Score(context.Background(), framework.NewCycleState(), nil, endpoints2)

	// Should use composite fallback, pod3 should score higher (lower KV, no queue)
	if scores2[ep3] <= scores2[ep4] {
		t.Errorf("composite fallback: better metrics should score higher: ep3=%f, ep4=%f", scores2[ep3], scores2[ep4])
	}
}

func TestPipelineAffinityGateNormalizationScope(t *testing.T) {
	// Verify that the within-tier affinity gate narrows BEFORE normalization,
	// so scores are relative to the sticky subset only.
	//
	// Setup: 3 positive endpoints. A and C are sticky (prefix >= 0.80). B is not.
	// B has extreme headroom that would stretch the normalization range if included.
	//
	// The affinity gate filters to [A, C], and normalization occurs within [A, C] only.

	config := noExploreConfig()
	config.AffinityGateTau = 0.80
	config.EpsilonExploreSticky = 0     // never explore
	config.AffinityMaxTTFTPenaltyMs = 0 // disable TTFT load gate
	s := NewLatencyScorer(config)

	a := ep("A", 0.3, 0, 5)
	b := ep("B", 0.5, 0, 8)
	c := ep("C", 0.4, 0, 6)

	// A: sticky, TTFT headroom=200
	// B: NOT sticky, TTFT headroom=500 (would stretch range)
	// C: sticky, TTFT headroom=100
	setLatency(a, true, true, 200, 50, 50, 10)
	setLatency(b, true, true, 500, 100, 10, 5)
	setLatency(c, true, true, 100, 30, 100, 20)

	// Set prefix scores: A and C are sticky, B is not
	setPrefixScore := func(endpoint framework.Endpoint, match, total int) {
		info := attrprefix.NewPrefixCacheMatchInfo(match, total, 16)
		endpoint.Put(attrprefix.PrefixCacheMatchInfoKey, info)
	}
	setPrefixScore(a, 9, 10) // 0.90 >= 0.80 → sticky
	setPrefixScore(b, 3, 10) // 0.30 < 0.80 → not sticky
	setPrefixScore(c, 8, 10) // 0.80 >= 0.80 → sticky

	endpoints := []framework.Endpoint{a, b, c}
	scores := s.Score(context.Background(), framework.NewCycleState(), nil, endpoints)

	// B should have score=0 (filtered out by affinity gate)
	if scores[b] != 0 {
		t.Errorf("non-sticky B should have score 0, got %f", scores[b])
	}

	// A and C should have non-zero scores
	if scores[a] == 0 {
		t.Error("sticky A should have non-zero score")
	}
	if scores[c] == 0 {
		t.Error("sticky C should have non-zero score")
	}

	// With normalization within [A, C] only (range 100-200):
	// C (headroom=100, tighter) should score higher than A (headroom=200, looser)
	// because the "least" strategy prefers tighter packing.
	if scores[c] <= scores[a] {
		t.Errorf("tighter sticky C should score higher than looser A: C=%f, A=%f", scores[c], scores[a])
	}

	t.Logf("scores: A=%f, B=%f, C=%f", scores[a], scores[b], scores[c])
}
