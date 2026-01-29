/*
Copyright 2026 The Kubernetes Authors.

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

// Copied from Gateway API https://github.com/kubernetes-sigs/gateway-api/tree/28de270b3139a8e172d93d23cb41faa0bf5e4ec8/conformance/utils/suite

package suite

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/utils/features"
)

// -----------------------------------------------------------------------------
// Conformance Profiles - Public Types
// -----------------------------------------------------------------------------

// ConformanceProfile is a group of features that have a related purpose, e.g.
// to cover specific protocol support or a specific feature present in Gateway
// API.
//
// For more details see the relevant GEP: https://gateway-api.sigs.k8s.io/geps/gep-1709/
type ConformanceProfile struct {
	Name             ConformanceProfileName
	CoreFeatures     sets.Set[features.FeatureName]
	ExtendedFeatures sets.Set[features.FeatureName]
}

type ConformanceProfileName string

const (
	// GatewayLayerProfileName indicates the name of the conformance profile
	// which covers the Gateway API layer aspects of the Inference Extension.
	GatewayLayerProfileName ConformanceProfileName = "Gateway"
)

// -----------------------------------------------------------------------------
// Conformance Profiles - Public Vars
// -----------------------------------------------------------------------------

var (
	// GatewayLayerProfile is a ConformanceProfile that covers testing
	// the Gateway API layer aspects of the Inference Extension (e.g., InferencePool,
	// InferenceObjective CRDs).
	GatewayLayerProfile = ConformanceProfile{
		Name: GatewayLayerProfileName,
		CoreFeatures: sets.New(
			features.SupportGateway, // This is needed to ensure manifest gets applied during setup.
			features.SupportHTTPRoute,
			features.SupportInferencePool,
		),
	}
	// Future profiles will cover EPP and ModelServer layers.
)

// -----------------------------------------------------------------------------
// Conformance Profiles - Private Profile Mapping Helpers
// -----------------------------------------------------------------------------

// conformanceProfileMap maps short human-readable names to their respective
// ConformanceProfiles.
var conformanceProfileMap = map[ConformanceProfileName]ConformanceProfile{
	// In the future, other profiles (EPP, ModelServer) will also be registered here,
	// and the suite runner will execute tests based on the selected profiles.
	GatewayLayerProfileName: GatewayLayerProfile,
}

// getConformanceProfileForName retrieves a known ConformanceProfile by its simple
// human readable ConformanceProfileName.
func getConformanceProfileForName(name ConformanceProfileName) (ConformanceProfile, error) {
	profile, ok := conformanceProfileMap[name]
	if !ok {
		return profile, fmt.Errorf("%s is not a valid conformance profile", name)
	}

	return profile, nil
}

// getConformanceProfilesForTest retrieves the ConformanceProfiles a test belongs to.
func getConformanceProfilesForTest(test ConformanceTest, conformanceProfiles sets.Set[ConformanceProfileName]) sets.Set[*ConformanceProfile] {
	matchingConformanceProfiles := sets.New[*ConformanceProfile]()
	for _, conformanceProfileName := range conformanceProfiles.UnsortedList() {
		cp := conformanceProfileMap[conformanceProfileName]
		hasAllFeatures := true
		for _, feature := range test.Features {
			if !cp.CoreFeatures.Has(feature) && !cp.ExtendedFeatures.Has(feature) {
				hasAllFeatures = false
				break
			}
		}
		if hasAllFeatures {
			matchingConformanceProfiles.Insert(&cp)
		}
	}

	return matchingConformanceProfiles
}
