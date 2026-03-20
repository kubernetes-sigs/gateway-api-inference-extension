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

package test

import (
	"fmt"

	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	crconfig "sigs.k8s.io/controller-runtime/pkg/config"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/basemodelextractor"
)

// NewTestBaseModelPlugin creates a BaseModelToHeaderPlugin using the real constructor
// with a lightweight manager that doesn't require a running API server.
func NewTestBaseModelPlugin() (*basemodelextractor.BaseModelToHeaderPlugin, error) {
	skipValidation := true
	mgr, err := ctrl.NewManager(&rest.Config{Host: "http://dummy:0"}, ctrl.Options{
		Metrics:    metricsserver.Options{BindAddress: "0"},
		Controller: crconfig.Controller{SkipNameValidation: &skipValidation},
	})
	if err != nil {
		return nil, fmt.Errorf("create test manager: %w", err)
	}

	fakeClient := fake.NewClientBuilder().Build()
	plugin, err := basemodelextractor.NewBaseModelToHeaderPlugin(
		func() *builder.Builder { return ctrl.NewControllerManagedBy(mgr) },
		fakeClient,
	)
	if err != nil {
		return nil, fmt.Errorf("create BaseModelToHeaderPlugin: %w", err)
	}

	return plugin, nil
}
