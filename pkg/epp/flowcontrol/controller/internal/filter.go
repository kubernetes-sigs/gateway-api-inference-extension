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

package internal

import (
	"context"

	"github.com/go-logr/logr"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// BandFilter is a function that acts as a pre-policy gate. It takes a complete view of a priority band and returns a
// subset of flow IDs that are considered viable candidates for a subsequent policy decision. It can also return a
// boolean signal to pause the entire operation for the band.
//
// This abstraction decouples the logic of determining *viability* (e.g., is a flow subject to backpressure?) from the
// logic of *selection* (e.g., which of the viable flows is the fairest to pick next?). This separation simplifies the
// mental model for policy authors, who can focus solely on selection logic without needing to account for external
// gating signals.
//
// Because filters are applied before the band is passed to a policy, they are inherently composable. Multiple filters
// can be chained to apply different viability criteria. For example, a future filter could be developed to temporarily
// exclude a "misbehaving" flow that is causing repeated errors, quarantining it from policy consideration.
//
// A nil `allowedFlows` map indicates that no filtering is necessary and all flows in the band are visible.
// This provides a zero-allocation fast path for the common case where no flows are being filtered.
type BandFilter func(
	ctx context.Context,
	band framework.PriorityBandAccessor,
	logger logr.Logger,
) (allowedFlows map[string]struct{}, shouldPause bool)

// NewSaturationFilter creates a `BandFilter` that uses the provided `contracts.SaturationDetector` to determine which
// flows are dispatchable. This is the standard filter used in the production `FlowController` for the dispatch
// operation.
func NewSaturationFilter(sd contracts.SaturationDetector) BandFilter {
	return func(
		ctx context.Context,
		band framework.PriorityBandAccessor,
		logger logr.Logger,
	) (map[string]struct{}, bool) {
		// Phase 1: Implement the current global saturation check.
		if sd.IsSaturated(ctx) {
			logger.V(logutil.VERBOSE).Info("System saturated, pausing dispatch for this shard.")
			return nil, true // Pause dispatching for all bands.
		}

		// Phase 2 (Future): This is where per-flow saturation logic would go.
		// It would iterate `band`, call `IsSaturated(ctx, flowID)`, and build a filtered map of allowed flows.
		// For now, no per-flow filtering is done. Return nil to signal the fast path.
		return nil, false // Do not pause, and do not filter any flows.
	}
}

// subsetPriorityBandAccessor provides a view of a priority band that is restricted to a specific subset of flows.
// It implements the `framework.PriorityBandAccessor` interface, ensuring that any policy operating on it will only
// see the allowed flows, regardless of which accessor method is used. This provides correctness by construction.
//
// For performance, it pre-computes a slice of the allowed flow IDs at creation time, making subsequent calls to
// `FlowIDs()` an O(1) operation with zero allocations.
type subsetPriorityBandAccessor struct {
	originalAccessor  framework.PriorityBandAccessor
	allowedFlows      map[string]struct{}
	allowedFlowsSlice []string
}

var _ framework.PriorityBandAccessor = &subsetPriorityBandAccessor{}

// newSubsetPriorityBandAccessor creates a new filtered view of a priority band.
func newSubsetPriorityBandAccessor(
	original framework.PriorityBandAccessor,
	allowed map[string]struct{},
) *subsetPriorityBandAccessor {
	// Pre-compute the slice of flow IDs for performance.
	ids := make([]string, 0, len(allowed))
	for id := range allowed {
		ids = append(ids, id)
	}

	return &subsetPriorityBandAccessor{
		originalAccessor:  original,
		allowedFlows:      allowed,
		allowedFlowsSlice: ids,
	}
}

// Priority returns the numerical priority level of this band.
func (s *subsetPriorityBandAccessor) Priority() uint {
	return s.originalAccessor.Priority()
}

// PriorityName returns the human-readable name of this priority band.
func (s *subsetPriorityBandAccessor) PriorityName() string {
	return s.originalAccessor.PriorityName()
}

// FlowIDs returns a slice of all flow IDs within this priority band that are in the allowed subset.
// This is an O(1) operation because the slice is pre-computed at creation.
func (s *subsetPriorityBandAccessor) FlowIDs() []string {
	return s.allowedFlowsSlice
}

// Queue returns a `framework.FlowQueueAccessor` for the specified `flowID` within this priority band, but only if it is
// in the allowed subset. This is an O(1) map lookup. If the flow is not in the allowed subset, it returns nil.
func (s *subsetPriorityBandAccessor) Queue(flowID string) framework.FlowQueueAccessor {
	if _, ok := s.allowedFlows[flowID]; !ok {
		return nil
	}
	return s.originalAccessor.Queue(flowID)
}

// IterateQueues executes the given `callback` for each `framework.FlowQueueAccessor` in the allowed subset of this
// priority band. The iteration stops if the callback returns false.
// This implementation delegates to the original accessor's iterator and applies the filter, which is more robust and
// efficient than iterating over a pre-computed slice of IDs.
func (s *subsetPriorityBandAccessor) IterateQueues(callback func(queue framework.FlowQueueAccessor) bool) {
	s.originalAccessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
		if _, ok := s.allowedFlows[queue.FlowSpec().ID]; ok {
			// This queue is in the allowed set, so execute the callback.
			if !callback(queue) {
				return false // The callback requested to stop, so we stop the outer iteration too.
			}
		}
		return true // Continue iterating over the original set.
	})
}
