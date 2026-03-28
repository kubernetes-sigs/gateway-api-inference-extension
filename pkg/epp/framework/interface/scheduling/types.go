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

package scheduling

import (
	"context"
	"fmt"
	"reflect"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
)

const nilString = "<nil>"

type Endpoint interface {
	GetMetadata() *fwkdl.EndpointMetadata
	GetMetrics() *fwkdl.Metrics
	String() string
	Get(string) (fwkdl.Cloneable, bool)
	Put(string, fwkdl.Cloneable)
	Keys() []string
}

func (ep *endpoint) String() string {
	if ep == nil {
		return nilString
	}

	return fmt.Sprintf("%+v", *ep)
}

func (ep *endpoint) GetMetadata() *fwkdl.EndpointMetadata {
	return ep.EndpointMetadata
}

func (ep *endpoint) GetMetrics() *fwkdl.Metrics {
	return ep.Metrics
}

type endpoint struct {
	*fwkdl.EndpointMetadata
	*fwkdl.Metrics
	fwkdl.AttributeMap
}

func NewEndpoint(meta *fwkdl.EndpointMetadata, metrics *fwkdl.Metrics, attr fwkdl.AttributeMap) Endpoint {
	if attr == nil {
		attr = fwkdl.NewAttributes()
	}

	return &endpoint{
		EndpointMetadata: meta.Clone(),
		Metrics:          metrics.Clone(),
		AttributeMap:     attr.Clone(),
	}
}

func EndpointComparer(a, b Endpoint) bool {
	a_ep := a.(*endpoint)
	b_ep := b.(*endpoint)
	return reflect.DeepEqual(a_ep, b_ep)
}

func ScoredEndpointComparer(a, b ScoredEndpoint) bool {
	return a.Score == b.Score && EndpointComparer(a.Endpoint, b.Endpoint)
}

type ScoredEndpoint struct {
	Endpoint
	Score float64
}

// ProfileRunResult captures the profile run result.
type ProfileRunResult struct {
	TargetEndpoints []Endpoint
}

// SchedulingResult captures the result of the scheduling cycle.
type SchedulingResult struct {
	ProfileResults     map[string]*ProfileRunResult
	PrimaryProfileName string
}

type SchedulerProfile interface {
	Run(ctx context.Context, request *requesthandling.InferenceRequest, cycleState *CycleState, candidateEndpoints []Endpoint) (*ProfileRunResult, error)
}
