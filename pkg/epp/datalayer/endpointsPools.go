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

import "sigs.k8s.io/gateway-api-inference-extension/pkg/common"

type EndPointsPool struct {
	EndPoints      *EndPoints
	StandaloneMode bool
	GKNN           common.GKNN
}

// NewEndPointsPool creates and returns a new empty instance of EndPointsPool.
func NewEndPointsPool() *EndPointsPool {
	endPoints := NewEndPoints()
	return &EndPointsPool{
		EndPoints: endPoints,
	}
}

type EndPoints struct {
	Selector    map[string]string
	TargetPorts []int
}

// NewEndPoints creates and returns a new empty instance of EndPointsPool.
func NewEndPoints() *EndPoints {
	return &EndPoints{
		Selector:    make(map[string]string),
		TargetPorts: []int{},
	}
}
