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
	"fmt"

	"k8s.io/apimachinery/pkg/types"
)

// Addressable supports getting an IP address and a namespaced name.
type Addressable interface {
	GetIPAddress() string
	GetPort() string
	GetMetricsHost() string
	GetNamespacedName() types.NamespacedName
}

// EndpointMetadata represents the relevant Kubernetes Pod state of an inference server.
type EndpointMetadata struct {
	NamespacedName types.NamespacedName
	PodName        string
	Address        string
	Port           string
	MetricsHost    string
	Labels         map[string]string
}

// String returns a string representation of the endpoint.
func (e *EndpointMetadata) String() string {
	if e == nil {
		return ""
	}
	return fmt.Sprintf("%+v", *e)
}

// Clone returns a full copy of the object.
func (p *EndpointMetadata) Clone() *EndpointMetadata {
	if p == nil {
		return nil
	}

	clonedLabels := make(map[string]string, len(p.Labels))
	for key, value := range p.Labels {
		clonedLabels[key] = value
	}
	return &EndpointMetadata{
		NamespacedName: types.NamespacedName{
			Name:      p.NamespacedName.Name,
			Namespace: p.NamespacedName.Namespace,
		},
		PodName:     p.PodName,
		Address:     p.Address,
		Port:        p.Port,
		MetricsHost: p.MetricsHost,
		Labels:      clonedLabels,
	}
}

// GetNamespacedName gets the namespace name of the Endpoint.
func (e *EndpointMetadata) GetNamespacedName() types.NamespacedName {
	return e.NamespacedName
}

// GetIPAddress returns the Endpoint's IP address.
func (e *EndpointMetadata) GetIPAddress() string {
	return e.Address
}

// GetPort returns the Endpoint's inference port.
func (e *EndpointMetadata) GetPort() string {
	return e.Port
}

// GetMetricsHost returns the Endpoint's metrics host (ip:port)
func (e *EndpointMetadata) GetMetricsHost() string {
	return e.MetricsHost
}
