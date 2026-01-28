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

package contracts

import (
	"context"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
)

// PodLocator defines the contract for a component that resolves the set of candidate pods for a request based on its
// metadata (e.g., subsetting).
//
// This interface allows the Flow Controller to fetch a fresh list of pods dynamically during the dispatch cycle,
// enabling support for "Scale-from-Zero" scenarios where pods may not exist when the request is first enqueued.
type PodLocator interface {
	// Locate returns a list of pod metrics that match the criteria defined in the request metadata.
	Locate(ctx context.Context, requestMetadata map[string]any) []metrics.PodMetrics
}

// SaturationDetector defines the contract for a component that provides real-time load signals to the FlowController.
type SaturationDetector interface {
	// Saturation returns the saturation level of the pool
	// - A value >= 1.0 indicates that the system is fully saturated.
	// - A value < 1.0 indicates the ratio of used capacity to total capacity.
	//
	// FlowController consumes this signal to make dispatch decisions:
	// - If Saturation() >= 1.0: Stop dispatching (enforce HoL blocking).
	// - If Saturation() < 1.0: Continue dispatching.
	Saturation(ctx context.Context, candidatePods []metrics.PodMetrics) float64
}
