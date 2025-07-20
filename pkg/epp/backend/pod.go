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

package backend

import (
	"fmt"

	"k8s.io/apimachinery/pkg/types"
)

type Pod struct {
	NamespacedName types.NamespacedName
	Address        string
	Labels         map[string]string
	RunningRequests *RequestPriorityQueue
}

func NewPod(name, namespace, address string, labels map[string]string) *Pod {
	return &Pod{
		NamespacedName: types.NamespacedName{
			Name:      name,
			Namespace: namespace,
		},
		Address:         address,
		Labels:          labels,
		RunningRequests: NewRequestPriorityQueue(),
	}
}

func (p *Pod) String() string {
	if p == nil {
		return ""
	}
	queueSize := 0
	if p.RunningRequests != nil {
		queueSize = p.RunningRequests.GetSize()
	}
	return fmt.Sprintf("Pod{%s, %s, %d running requests}", 
		p.NamespacedName.String(), p.Address, queueSize)
}

func (p *Pod) Clone() *Pod {
	if p == nil {
		return nil
	}
	clonedLabels := make(map[string]string, len(p.Labels))
	for key, value := range p.Labels {
		clonedLabels[key] = value
	}
	
	var clonedRequests *RequestPriorityQueue
	if p.RunningRequests != nil {
		clonedRequests = p.RunningRequests.Clone()
	}
	
	return &Pod{
		NamespacedName: types.NamespacedName{
			Name:      p.NamespacedName.Name,
			Namespace: p.NamespacedName.Namespace,
		},
		Address: p.Address,
		Labels:  clonedLabels,
		RunningRequests: clonedRequests,
	}
}
