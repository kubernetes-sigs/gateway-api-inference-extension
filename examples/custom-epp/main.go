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

// This example shows how to build a custom EPP binary that includes
// out-of-tree scheduling plugins alongside the standard in-tree ones.
//
// It follows the same pattern as llm-d-inference-scheduler:
// https://github.com/llm-d/llm-d-inference-scheduler/blob/main/cmd/epp/main.go
//
// The pattern is:
//
//  1. Call plugins.RegisterAllPlugins() to register your custom plugin factories.
//  2. Call runner.NewRunner().Run() — the standard Runner registers all
//     in-tree plugins and then loads the YAML config, which can now
//     reference both in-tree AND your out-of-tree plugin types.
//
// To build and run:
//
//	go build -o custom-epp ./examples/custom-epp/
//	./custom-epp --config-file=my-config.yaml
//
// The YAML config can reference the custom plugins by type:
//
//	apiVersion: inference.networking.x-k8s.io/v1alpha1
//	kind: EndpointPickerConfig
//	plugins:
//	  - type: model-affinity-filter
//	    name: my-model-filter
//	  - type: least-loaded-scorer
//	    name: my-least-loaded
//	  - type: logging-profile-handler
//	    name: my-profile-handler
//	  # ... plus any in-tree plugins like queue-scorer, max-score-picker, etc.
//	schedulingProfiles:
//	  - name: default
//	    plugins:
//	      - pluginRef: my-model-filter
//	      - pluginRef: my-least-loaded
//	        weight: 2
//	      - pluginRef: max-score-picker
package main

import (
	"os"

	ctrl "sigs.k8s.io/controller-runtime"

	"sigs.k8s.io/gateway-api-inference-extension/cmd/epp/runner"
	"sigs.k8s.io/gateway-api-inference-extension/examples/scheduler/plugins"
)

func main() {
	// Register custom out-of-tree plugins before starting the Runner.
	// This is identical to how llm-d-inference-scheduler does it:
	//   plugins.RegisterAllPlugins()
	//
	// In a real out-of-tree project this would be:
	//   myplugins "github.com/my-org/my-epp-plugins/plugins"
	//   myplugins.RegisterAllPlugins()
	plugins.RegisterAllPlugins()

	// Note: GIE built-in plugins are automatically registered by the runner
	// when it processes configuration in runner.parsePluginsConfiguration().

	// The Runner will:
	//   1. Register all in-tree plugins   (registerInTreePlugins)
	//   2. Load the YAML config           (LoadRawConfig)
	//   3. Instantiate plugins by type    (InstantiateAndConfigure)
	//   4. Build the scheduler            (buildSchedulerConfig)
	//   5. Start the gRPC/HTTP server
	if err := runner.NewRunner().Run(ctrl.SetupSignalHandler()); err != nil {
		os.Exit(1)
	}
}
