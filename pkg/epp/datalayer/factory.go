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

package datalayer

import (
	"context"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

const (
	ExperimentalDatalayerFeatureGate = "dataLayer"
	PrepareDataPluginsFeatureGate    = "prepareDataPlugins"
)

// PoolInfo represents the DataStore information needed for endpoints.
// TODO:
// Consider if to remove/simplify in follow-ups. This is mostly for backward
// compatibility with backend.metrics' expectations and allowing a shared
// implementation during the transition.
//   - Endpoint metric scraping uses PoolGet to access the pool's Port and Name.
//   - Global metrics logging uses PoolGet solely for error return and PodList to enumerate
//     all endpoints for metrics summarization.
type PoolInfo interface {
	PoolGet() (*EndpointPool, error)
	PodList(func(fwkdl.Endpoint) bool) []fwkdl.Endpoint
	// EndpointSetHealthy marks an endpoint as healthy or unhealthy based on metrics scraping results.
	// When healthy is false, the endpoint is removed from PodList results.
	// When healthy is true, the endpoint is added back.
	EndpointSetHealthy(ep fwkdl.Endpoint, healthy bool)
}

// EndpointFactory defines an interface for managing Endpoint lifecycle. Specifically,
// providing methods to allocate and retire endpoints. This can potentially be used for
// pooled memory or other management chores in the implementation.
type EndpointFactory interface {
	NewEndpoint(parent context.Context, inEnpointMetadata *fwkdl.EndpointMetadata, poolinfo PoolInfo) fwkdl.Endpoint
	ReleaseEndpoint(ep fwkdl.Endpoint)
}
