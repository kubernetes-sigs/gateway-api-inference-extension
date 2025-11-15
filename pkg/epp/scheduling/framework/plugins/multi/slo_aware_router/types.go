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

type HeadroomStrategy string

type Choice struct {
	PodName schedulingtypes.Pod
	Weight  int
}

const (
	// HeadroomStrategyLeast prioritizes pods with least positive headroom (better packing)
	HeadroomStrategyLeast HeadroomStrategy = "least"
	// HeadroomStrategyMost prioritizes pods with most positive headroom (more conservative)
	HeadroomStrategyMost HeadroomStrategy = "most"

	HeadroomStrategyCompositeLeast HeadroomStrategy = "composite-least"
	HeadroomStrategyCompositeMost  HeadroomStrategy = "composite-most"
	HeadroomStrategyCompositeOnly  HeadroomStrategy = "composite-only"

	// TTFT header string
	TTFTSLOHeaderKey = "x-slo-ttft-ms"
	// TPOT header string
	TPOTSLOHeaderKey = "x-slo-tpot-ms"
)

const (
	SLOAwareRouterPluginType = "slo-aware-routing"
	eps                      = 1e-9
	Wmax                     = 100
	minWeight                = 1
)

type PodSelectionMode string

const (
	PodSelectionLinear PodSelectionMode = "linear" // weighted-random (current behavior)
	PodSelectionMax    PodSelectionMode = "max"    // pick argmax weight
)
