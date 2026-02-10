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

package profiling

import (
	"net/http/pprof"
	"runtime"

	ctrl "sigs.k8s.io/controller-runtime"
)

// setupPprofHandlers only implements the pre-defined profiles:
// https://cs.opensource.google/go/go/+/refs/tags/go1.24.4:src/runtime/pprof/pprof.go;l=108
func SetupPprofHandlers(mgr ctrl.Manager) error {
	profiles := []string{
		"heap",
		"goroutine",
		"allocs",
		"threadcreate",
		"block",
		"mutex",
	}
	for _, p := range profiles {
		if err := mgr.AddMetricsServerExtraHandler("/debug/pprof/"+p, pprof.Handler(p)); err != nil {
			return err
		}
	}

	runtime.SetMutexProfileFraction(1)
	runtime.SetBlockProfileRate(1)

	return nil
}
