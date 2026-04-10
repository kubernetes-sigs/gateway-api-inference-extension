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
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

const (
	LatencyPredictionInfoKey = "LatencyPredictionInfoKey"
)

// TODO: Split LatencyPredictionInfo into two attribute keys:
//
//   - LatencyPredictionInfo — pure ML output: ttftValid, tpotValid, ttftHeadroom,
//     tpotHeadroom, ttft, tpot.
//   - EndpointRoutingContext — operational state the predictor passes to downstream
//     plugins: dispatchedRequestCount, prefillTokensInFlight, minRunningTPOTSLO,
//     endpointRole. Scales cleanly for disaggregated serving additions.

// LatencyPredictionInfo contains latency predictions for an endpoint
type LatencyPredictionInfo struct {
	// Individual validity checks
	ttftValid bool // TTFT within SLO?
	tpotValid bool // TPOT within SLO?

	// Headroom values (positive = within SLO, negative = violates SLO)
	ttftHeadroom float64 // TTFT headroom in ms
	tpotHeadroom float64 // TPOT headroom in ms

	// Raw prediction values
	ttft float64 // Predicted time to first token (ms)
	tpot float64 // Predicted time per output token (ms)

	// Dispatched request count from EPP's internal tracking.
	// More accurate than model server's RunningRequestsSize for routing,
	// as it reflects requests dispatched by this EPP instance.
	dispatchedRequestCount int
}

func NewLatencyPredictionInfo(
	ttftValid, tpotValid bool,
	ttftHeadroom, tpotHeadroom float64,
	ttft, tpot float64,
	dispatchedRequestCount int,
) *LatencyPredictionInfo {
	return &LatencyPredictionInfo{
		ttftValid:              ttftValid,
		tpotValid:              tpotValid,
		ttftHeadroom:           ttftHeadroom,
		tpotHeadroom:           tpotHeadroom,
		ttft:                   ttft,
		tpot:                   tpot,
		dispatchedRequestCount: dispatchedRequestCount,
	}
}

// Getters
func (l *LatencyPredictionInfo) TTFTValid() bool             { return l.ttftValid }
func (l *LatencyPredictionInfo) TPOTValid() bool             { return l.tpotValid }
func (l *LatencyPredictionInfo) IsValid() bool               { return l.ttftValid && l.tpotValid }
func (l *LatencyPredictionInfo) TTFTHeadroom() float64       { return l.ttftHeadroom }
func (l *LatencyPredictionInfo) TPOTHeadroom() float64       { return l.tpotHeadroom }
func (l *LatencyPredictionInfo) TTFT() float64               { return l.ttft }
func (l *LatencyPredictionInfo) TPOT() float64               { return l.tpot }
func (l *LatencyPredictionInfo) DispatchedRequestCount() int { return l.dispatchedRequestCount }

// Clone implements fwkdl.Cloneable
func (l *LatencyPredictionInfo) Clone() fwkdl.Cloneable {
	if l == nil {
		return nil
	}
	return &LatencyPredictionInfo{
		ttftValid:              l.ttftValid,
		tpotValid:              l.tpotValid,
		ttftHeadroom:           l.ttftHeadroom,
		tpotHeadroom:           l.tpotHeadroom,
		ttft:                   l.ttft,
		tpot:                   l.tpot,
		dispatchedRequestCount: l.dispatchedRequestCount,
	}
}
