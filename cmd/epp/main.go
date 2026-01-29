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

package main

import (
	"fmt"
	"os"
	"runtime/debug"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	"sigs.k8s.io/gateway-api-inference-extension/cmd/epp/runner"
)

func main() {
	// Delegate to a run function.
	// This ensures that if run() returns an error, we print it and exit.
	// Crucially, it ensures that any defers inside run() execute BEFORE os.Exit is called.
	if err := run(); err != nil {
		// We use the global logger here assuming it was set during bootstrap or runner init.
		// If setup failed completely, this writes to stderr.
		ctrl.Log.Error(err, "Application exited with error")
		os.Exit(1)
	}
}

func run() error {
	// Setup bootstrap logger.
	// This logger is used for initialization errors before the Runner configures the user-specified logging format
	// (JSON/Console, Verbosity).
	bootstrapLog := zap.New(zap.UseDevMode(true))
	ctrl.SetLogger(bootstrapLog)

	// Panic Recovery: This catches panics on the main goroutine during initialization.
	// Note: It will NOT catch panics in child goroutines spawned by the Manager.
	defer func() {
		if r := recover(); r != nil {
			err, ok := r.(error)
			if !ok {
				err = fmt.Errorf("%v", r)
			}
			bootstrapLog.Error(err, "CRITICAL: Process panic recovered", "stack", string(debug.Stack()))
			os.Exit(1)
		}
	}()

	ctx := ctrl.SetupSignalHandler()

	// Execute Runner.
	// For adding out-of-tree plugins to the plugins registry, use the following:
	// plugins.Register(name, factory)
	return runner.NewRunner().Run(ctx)
}
