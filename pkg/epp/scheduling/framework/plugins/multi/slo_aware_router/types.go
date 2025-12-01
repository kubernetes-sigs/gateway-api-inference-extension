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

// Package requestcontrol contains helpers to decouple latency-predictor logic.
package slo_aware_router

import schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

type headroomStrategy string

type choice struct {
	podName schedulingtypes.Pod
	weight  int
}

const (
	// headroomStrategyLeast prioritizes pods with least positive headroom (better packing)
	headroomStrategyLeast headroomStrategy = "least"
	// headroomStrategyMost prioritizes pods with most positive headroom (more conservative)
	headroomStrategyMost headroomStrategy = "most"

	headroomStrategyCompositeLeast headroomStrategy = "composite-least"
	headroomStrategyCompositeMost  headroomStrategy = "composite-most"
	headroomStrategyCompositeOnly  headroomStrategy = "composite-only"

	// TTFT header string
	ttftSLOHeaderKey = "x-slo-ttft-ms"
	// TPOT header string
	tpotSLOHeaderKey = "x-slo-tpot-ms"
)

const (
	SLOAwareRouterPluginType = "predicted-latency-scorer"
	eps                      = 1e-9
	wMax                     = 100
	minWeight                = 1
)

type podSelectionMode string

const (
	podSelectionLinear podSelectionMode = "linear" // weighted-random (current behavior)
	podSelectionMax    podSelectionMode = "max"    // pick argmax weight
)
