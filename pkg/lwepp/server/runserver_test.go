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

package server_test

import (
	"testing"

	"sigs.k8s.io/controller-runtime/pkg/manager"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/server"
)

func TestRunnable(t *testing.T) {
	// Make sure AsRunnable() does not use leader election.
	runner := server.NewDefaultExtProcServerRunner().AsRunnable(logutil.NewTestLogger())
	r, ok := runner.(manager.LeaderElectionRunnable)
	if !ok {
		t.Fatal("runner is not LeaderElectionRunnable")
	}
	if r.NeedLeaderElection() {
		t.Error("runner returned NeedLeaderElection = true, expected false")
	}
}

func TestNewDefaultExtProcServerRunnerAppliesOptionDefaults(t *testing.T) {
	opts := server.NewOptions()
	runner := server.NewDefaultExtProcServerRunner()

	if runner.GrpcPort != opts.GRPCPort {
		t.Errorf("GrpcPort = %d, want %d", runner.GrpcPort, opts.GRPCPort)
	}
	if runner.GrpcHealthPort != opts.GRPCHealthPort {
		t.Errorf("GrpcHealthPort = %d, want %d", runner.GrpcHealthPort, opts.GRPCHealthPort)
	}
	if runner.HealthChecking != opts.HealthChecking {
		t.Errorf("HealthChecking = %t, want %t", runner.HealthChecking, opts.HealthChecking)
	}
	// SecureServing defaults to true, so a runner built from the defaults must serve TLS.
	if runner.SecureServing != opts.SecureServing {
		t.Errorf("SecureServing = %t, want %t", runner.SecureServing, opts.SecureServing)
	}
}
