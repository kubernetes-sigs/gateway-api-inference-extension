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

// Package profile demonstrates how to implement a custom ProfileHandler plugin.
//
// A ProfileHandler controls the multi-profile scheduling loop. It has two jobs:
//
//  1. Pick — decide which SchedulerProfiles to run in this iteration.
//     The framework calls Pick repeatedly until it returns an empty map.
//  2. ProcessResults — after all profiles finish, aggregate their results
//     and choose the "primary" profile whose result becomes the final answer.
//
// The built-in SingleProfileHandler works for one-profile setups. This example
// shows a LoggingProfileHandler that wraps the same logic with tracing output
// so developers can see the scheduling lifecycle.
//
// To implement your own ProfileHandler:
//
//  1. Define a struct for any handler state.
//  2. Implement plugin.Plugin              — return a TypedName.
//  3. Implement scheduling.ProfileHandler:
//     - Pick()           — return profiles to run (empty map = done).
//     - ProcessResults() — return the final SchedulingResult.
//  4. Provide a FactoryFunc and register it in RegisterAllPlugins() (see plugins/register.go).
package profile

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const PluginType = "logging-profile-handler"

// compile-time interface check
var _ fwksched.ProfileHandler = &LoggingProfileHandler{}

// LoggingProfileHandler runs all registered profiles exactly once (like
// SingleProfileHandler) but prints trace information at each step. This is
// useful for understanding how the framework drives the profile lifecycle.
type LoggingProfileHandler struct {
	typedName fwkplugin.TypedName
}

// New creates a LoggingProfileHandler with default type and name.
func New() *LoggingProfileHandler {
	return &LoggingProfileHandler{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: PluginType},
	}
}

func (h *LoggingProfileHandler) TypedName() fwkplugin.TypedName { return h.typedName }

// Pick decides which profiles to run. The framework calls this in a loop:
//   - First call: profileResults is empty → return all profiles so they run.
//   - Second call: profileResults has results for every profile → return
//     empty map to signal "done".
func (h *LoggingProfileHandler) Pick(
	_ context.Context,
	_ *fwksched.CycleState,
	request *fwksched.LLMRequest,
	profiles map[string]fwksched.SchedulerProfile,
	profileResults map[string]*fwksched.ProfileRunResult,
) map[string]fwksched.SchedulerProfile {
	if len(profiles) == len(profileResults) {
		fmt.Printf("  [profile/%s] all %d profile(s) completed, stopping loop\n",
			PluginType, len(profiles))
		return map[string]fwksched.SchedulerProfile{}
	}

	names := make([]string, 0, len(profiles))
	for name := range profiles {
		names = append(names, name)
	}
	fmt.Printf("  [profile/%s] picking profiles to run for model=%s: %v\n",
		PluginType, request.TargetModel, names)
	return profiles
}

// ProcessResults aggregates the profile outcomes and selects the primary one.
// For a single-profile setup the logic is trivial: the one profile IS the primary.
// A multi-profile handler could compare results, log shadow profiles, etc.
func (h *LoggingProfileHandler) ProcessResults(
	_ context.Context,
	_ *fwksched.CycleState,
	_ *fwksched.LLMRequest,
	profileResults map[string]*fwksched.ProfileRunResult,
) (*fwksched.SchedulingResult, error) {
	if len(profileResults) == 0 {
		return nil, errors.New("no profile results to process")
	}

	var primaryName string
	for name, result := range profileResults {
		if result == nil {
			fmt.Printf("  [profile/%s] profile %q failed (nil result)\n", PluginType, name)
			return nil, fmt.Errorf("profile %q failed", name)
		}
		fmt.Printf("  [profile/%s] profile %q selected %d endpoint(s)\n",
			PluginType, name, len(result.TargetEndpoints))
		primaryName = name
	}

	fmt.Printf("  [profile/%s] primary profile → %q\n", PluginType, primaryName)
	return &fwksched.SchedulingResult{
		ProfileResults:     profileResults,
		PrimaryProfileName: primaryName,
	}, nil
}

// Factory creates a LoggingProfileHandler from a YAML config entry.
func Factory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	h := New()
	h.typedName.Name = name
	return h, nil
}
