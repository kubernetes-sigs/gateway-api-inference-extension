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

package requestcontrol

import (
	"context"
	"errors"
	"time"

	fwk "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// executePluginsAsDAG executes PrepareData plugins as a DAG based on their dependencies asynchronously.
// So, a plugin is executed only after all its dependencies have been executed.
// If there is a cycle or any plugin fails with error, it returns an error.
func executePluginsAsDAG(plugins []fwk.DataProducer, ctx context.Context, request *schedulingtypes.InferenceRequest, endpoints []schedulingtypes.Endpoint) error {
	for _, plugin := range plugins {
		if err := plugin.PrepareRequestData(ctx, request, endpoints); err != nil {
			return errors.New("prepare data plugin " + plugin.TypedName().String() + " failed: " + err.Error())
		}
	}
	return nil
}

// prepareDataPluginsWithTimeout executes the PrepareRequestData plugins with retries and timeout.
// The child context is cancelled when the timeout fires so plugins can observe cancellation
// (e.g. abort outbound HTTP calls) and avoid committing state after the director has moved on.
func prepareDataPluginsWithTimeout(timeout time.Duration, plugins []fwk.DataProducer,
	ctx context.Context, request *schedulingtypes.InferenceRequest, endpoints []schedulingtypes.Endpoint) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		errCh <- executePluginsAsDAG(plugins, ctx, request, endpoints)
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return errors.New("prepare data plugin timed out")
		}
		return ctx.Err()
	}
}
