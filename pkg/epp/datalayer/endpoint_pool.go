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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
)

type EndpointPool struct {
	EndpointMeta           *EndpointsMeta
	DisableK8sCrdReconcile bool
	GKNN                   common.GKNN
}

// NewEndpointPool creates and returns a new empty instance of EndpointPool.
func NewEndpointPool(disableK8sCrdReconcile bool, gknn common.GKNN) *EndpointPool {
	endpointsMeta := NewEndpointMeta()
	return &EndpointPool{
		GKNN:                   gknn,
		DisableK8sCrdReconcile: disableK8sCrdReconcile,
		EndpointMeta:           endpointsMeta,
	}
}

type EndpointsMeta struct {
	Selector    map[string]string
	TargetPorts []int
}

// NewEndpointMeta creates and returns a new empty instance of EndpointPool.
func NewEndpointMeta() *EndpointsMeta {
	return &EndpointsMeta{
		Selector:    make(map[string]string),
		TargetPorts: []int{},
	}
}
