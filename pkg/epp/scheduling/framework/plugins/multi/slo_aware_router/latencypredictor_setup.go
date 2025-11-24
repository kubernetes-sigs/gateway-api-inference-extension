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

package slo_aware_router

import (
	"context"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

// predictorRunnable implements controller-runtime's Runnable interface to manage the predictor's lifecycle.
type predictorRunnable struct {
	predictor *latencypredictor.Predictor
}

func (p *predictorRunnable) Start(ctx context.Context) error {
	logger := log.FromContext(ctx)
	logger.V(logutil.DEBUG).Info("Starting latency predictor...")
	if err := p.predictor.Start(ctx); err != nil {
		logger.V(logutil.DEBUG).Error(err, "Failed to start latency predictor")
		return err
	}
	logger.V(logutil.DEBUG).Info("Latency predictor started.")
	<-ctx.Done()
	logger.V(logutil.DEBUG).Info("Stopping latency predictor...")
	p.predictor.Stop()
	return nil
}
