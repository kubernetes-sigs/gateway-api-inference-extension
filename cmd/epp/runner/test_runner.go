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

package runner

import (
	"context"

	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
)

// NewTestRunnerSetup creates a setup runner dedicated for integration test and its corresponding dataStore.
func NewTestRunnerSetup(ctx context.Context, cfg *rest.Config, opts *runserver.Options, pmc backendmetrics.PodMetricsClient) (ctrl.Manager, datastore.Datastore, error) {
	runner := NewRunner()
	runner.testOverrideSkipNameValidation = true
	return runner.setup(ctx, cfg, opts, pmc)
}
