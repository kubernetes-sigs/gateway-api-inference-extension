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

// This example demonstrates how to extend the IGW scheduling framework with
// custom out-of-tree plugins. It follows the same package layout as
// https://github.com/llm-d/llm-d-inference-scheduler/tree/main/pkg/plugins:
//
//	examples/scheduler/
//	├── main.go                              # This file — wires plugins together
//	├── plugins/
//	│   ├── register.go                      # RegisterAllPlugins() — explicit registration
//	│   ├── filter/
//	│   │   └── model_affinity.go            # Custom Filter plugin
//	│   ├── scorer/
//	│   │   └── least_loaded.go              # Custom Scorer plugin
//	│   ├── picker/
//	│   │   └── top_k_random.go              # Custom Picker plugin
//	│   └── profile/
//	│       └── logging_profile_handler.go    # Custom ProfileHandler plugin
//
// Each plugin package:
//  1. Defines a struct implementing the relevant scheduling interface.
//  2. Provides a FactoryFunc for YAML-driven instantiation.
//
// The central plugins/register.go calls plugin.Register() for every factory,
// following the same explicit-registration pattern as llm-d-inference-scheduler.
//
// Usage:
//
//	make run-example EXAMPLE=scheduler
package main

import (
	"context"
	"fmt"
	"os"

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"

	// Direct imports for programmatic instantiation in this example.
	"sigs.k8s.io/gateway-api-inference-extension/examples/scheduler/plugins"
	customfilter "sigs.k8s.io/gateway-api-inference-extension/examples/scheduler/plugins/filter"
	custompicker "sigs.k8s.io/gateway-api-inference-extension/examples/scheduler/plugins/picker"
	customprofile "sigs.k8s.io/gateway-api-inference-extension/examples/scheduler/plugins/profile"
	customscorer "sigs.k8s.io/gateway-api-inference-extension/examples/scheduler/plugins/scorer"
)

func main() {
	ctx := context.Background()

	// ----- Register all custom plugin factories -----
	// Same pattern as llm-d-inference-scheduler:
	//   https://github.com/llm-d/llm-d-inference-scheduler/blob/main/cmd/epp/main.go
	plugins.RegisterAllPlugins()

	// ----- Show registered plugin factories -----
	fmt.Println("=== Registered Plugin Factories ===")
	for pluginType := range fwkplugin.Registry {
		fmt.Printf("  %s\n", pluginType)
	}

	// ----- 1. Instantiate custom plugins -----
	// Direct instantiation — useful for programmatic setups and tests.
	// In production, plugins are instantiated by the config loader using the
	// registered FactoryFunc and a YAML EndpointPickerConfig.
	modelFilter := customfilter.New()
	leastLoadedScorer := customscorer.New()
	profileHandler := customprofile.New()

	// Custom picker: TopKRandomPicker picks randomly from the top K candidates,
	// balancing quality with load distribution.
	topKPicker := custompicker.New(custompicker.DefaultK)

	// ----- 2. Build a SchedulerProfile -----
	// A profile defines the Filter → Score → Pick pipeline.
	defaultProfile := scheduling.NewSchedulerProfile().
		WithFilters(modelFilter).
		WithScorers(scheduling.NewWeightedScorer(leastLoadedScorer, 1)).
		WithPicker(topKPicker)

	// ----- 3. Create the Scheduler -----
	// The ProfileHandler + profile map form the SchedulerConfig.
	config := scheduling.NewSchedulerConfig(
		profileHandler,
		map[string]fwksched.SchedulerProfile{"default": defaultProfile},
	)
	scheduler := scheduling.NewSchedulerWithConfig(config)

	// ----- 4. Simulate inference endpoints -----
	endpoints := []fwksched.Endpoint{
		fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "vllm-pod-1"}},
			&fwkdl.Metrics{
				WaitingQueueSize:    3,
				KVCacheUsagePercent: 0.6,
				MaxActiveModels:     4,
				ActiveModels:        map[string]int{"llama-70b": 1, "mistral-7b": 1, "codellama-34b": 1},
			}, nil),
		fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "vllm-pod-2"}},
			&fwkdl.Metrics{
				WaitingQueueSize:    1,
				KVCacheUsagePercent: 0.3,
				MaxActiveModels:     4,
				ActiveModels:        map[string]int{"llama-70b": 1},
			}, nil),
		fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "vllm-pod-3"}},
			&fwkdl.Metrics{
				WaitingQueueSize:    0,
				KVCacheUsagePercent: 0.1,
				MaxActiveModels:     4,
				ActiveModels:        map[string]int{"mistral-7b": 1},
			}, nil),
		fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "vllm-pod-4"}},
			&fwkdl.Metrics{
				WaitingQueueSize:    8,
				KVCacheUsagePercent: 0.9,
				MaxActiveModels:     4,
				ActiveModels:        map[string]int{"llama-70b": 1, "mistral-7b": 1, "codellama-34b": 1, "phi-3": 1},
			}, nil),
	}

	// ----- 5. Create the scheduling request -----
	request := &fwksched.LLMRequest{
		RequestId:   "req-001",
		TargetModel: "llama-70b",
	}

	// ----- Print input -----
	fmt.Printf("\n=== Input ===\n")
	fmt.Printf("Request: model=%s, id=%s\n\n", request.TargetModel, request.RequestId)
	fmt.Printf("Candidate endpoints (%d):\n", len(endpoints))
	for _, ep := range endpoints {
		m := ep.GetMetrics()
		fmt.Printf("  %-12s  queue=%-3d  kvcache=%3.0f%%  models=%d/%d  %v\n",
			ep.GetMetadata().NamespacedName.Name,
			m.WaitingQueueSize,
			m.KVCacheUsagePercent*100,
			len(m.ActiveModels), m.MaxActiveModels,
			modelNames(m.ActiveModels))
	}

	// ----- 6. Run the scheduler -----
	fmt.Printf("\n=== Scheduling Pipeline ===\n")
	result, err := scheduler.Schedule(ctx, request, endpoints)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nScheduling failed: %v\n", err)
		os.Exit(1)
	}

	// ----- Print result -----
	fmt.Printf("\n=== Result ===\n")
	fmt.Printf("Primary profile: %s\n", result.PrimaryProfileName)
	for name, pr := range result.ProfileResults {
		fmt.Printf("Profile %q → %d endpoint(s):\n", name, len(pr.TargetEndpoints))
		for _, ep := range pr.TargetEndpoints {
			if scored, ok := ep.(*fwksched.ScoredEndpoint); ok {
				fmt.Printf("  → %s  (score=%.4f)\n",
					scored.GetMetadata().NamespacedName.Name, scored.Score)
			} else {
				fmt.Printf("  → %s\n", ep.GetMetadata().NamespacedName.Name)
			}
		}
	}
}

func modelNames(m map[string]int) []string {
	names := make([]string, 0, len(m))
	for name := range m {
		names = append(names, name)
	}
	return names
}
