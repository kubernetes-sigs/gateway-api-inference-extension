package server_test

import (
	"testing"

	"sigs.k8s.io/controller-runtime/pkg/manager"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/server"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
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
