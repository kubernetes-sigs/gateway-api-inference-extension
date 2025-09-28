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
	GetPort() int32
	GetMetricsPort() int32
	GetNamespacedName() types.NamespacedName
}

// PodInfo represents the relevant Kubernetes Pod state of an inference server.
type PodInfo struct {
	NamespacedName types.NamespacedName
	PodName        string
	Address        string
	Port           int32
	MetricsPort    int32
	Labels         map[string]string
}

// String returns a string representation of the pod.
func (p *PodInfo) String() string {
	if p == nil {
		return ""
	}
	return fmt.Sprintf("%+v", *p)
}

// Clone returns a full copy of the object.
func (p *PodInfo) Clone() *PodInfo {
	if p == nil {
		return nil
	}

	clonedLabels := make(map[string]string, len(p.Labels))
	for key, value := range p.Labels {
		clonedLabels[key] = value
	}
	return &PodInfo{
		NamespacedName: types.NamespacedName{
			Name:      p.NamespacedName.Name,
			Namespace: p.NamespacedName.Namespace,
		},
		PodName:     p.PodName,
		Address:     p.Address,
		Port:        p.Port,
		MetricsPort: p.MetricsPort,
		Labels:      clonedLabels,
	}
}

// GetNamespacedName gets the namespace name of the Pod.
func (p *PodInfo) GetNamespacedName() types.NamespacedName {
	return p.NamespacedName
}

// GetIPAddress returns the Pod's IP address.
func (p *PodInfo) GetIPAddress() string {
	return p.Address
}

// GetPort returns the Pod's inference port.
func (p *PodInfo) GetPort() int32 {
	return p.Port
}

// GetMetricsPort returns the pod's metrics port
func (p *PodInfo) GetMetricsPort() int32 {
	return p.MetricsPort
}
