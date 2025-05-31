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


package runnable

import "sigs.k8s.io/controller-runtime/pkg/manager"

type leaderElection struct {
	manager.Runnable
	needsLeaderElection bool
}

// LeaderElection wraps the given runnable to implement manager.LeaderElectionRunnable.
func LeaderElection(runnable manager.Runnable, needsLeaderElection bool) manager.Runnable {
	return &leaderElection{
		Runnable:            runnable,
		needsLeaderElection: needsLeaderElection,
	}
}

// RequireLeaderElection wraps the given runnable, marking it as requiring leader election.
func RequireLeaderElection(runnable manager.Runnable) manager.Runnable {
	return LeaderElection(runnable, true)
}

// RequireLeaderElection wraps the given runnable, marking it as not requiring leader election.
func NoLeaderElection(runnable manager.Runnable) manager.Runnable {
	return LeaderElection(runnable, false)
}

// NeedLeaderElection implements manager.NeedLeaderElection interface.
func (r *leaderElection) NeedLeaderElection() bool {
	return r.needsLeaderElection
}
