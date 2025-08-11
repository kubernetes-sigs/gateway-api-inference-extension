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

	corev1 "k8s.io/api/core/v1"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
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
	PoolGet() (*v1.InferencePool, error)
	PodList(func(Endpoint) bool) []Endpoint
}

// EndpointFactory defines an interface for managing Endpoint lifecycle. Specifically,
// providing methods to allocate and retire endpoints. This can potentially be used for
// pooled memory or other management chores in the implementation.
type EndpointFactory interface {
	NewEndpoint(parent context.Context, inpod *corev1.Pod, poolinfo PoolInfo) Endpoint
	ReleaseEndpoint(ep Endpoint)
}
