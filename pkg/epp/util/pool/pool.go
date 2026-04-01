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

package pool

import (
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	v1alpha2 "sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

func InferencePoolToEndpointPool(inferencePool *v1.InferencePool) *datalayer.EndpointPool {
	if inferencePool == nil {
		return nil
	}
	targetPorts := make([]int, 0, len(inferencePool.Spec.TargetPorts))
	for _, p := range inferencePool.Spec.TargetPorts {
		targetPorts = append(targetPorts, int(p.Number))

	}
	selector := make(map[string]string, len(inferencePool.Spec.Selector.MatchLabels))
	for k, v := range inferencePool.Spec.Selector.MatchLabels {
		selector[string(k)] = string(v)
	}
	endpointPool := &datalayer.EndpointPool{
		Selector:    selector,
		TargetPorts: targetPorts,
		Namespace:   inferencePool.Namespace,
		Name:        inferencePool.Name,
	}
	return endpointPool
}

func AlphaInferencePoolToEndpointPool(inferencePool *v1alpha2.InferencePool) *datalayer.EndpointPool {
	targetPorts := []int{int(inferencePool.Spec.TargetPortNumber)}
	selector := make(map[string]string, len(inferencePool.Spec.Selector))
	for k, v := range inferencePool.Spec.Selector {
		selector[string(k)] = string(v)
	}

	endpointPool := &datalayer.EndpointPool{
		TargetPorts: targetPorts,
		Selector:    selector,
		Namespace:   inferencePool.Namespace,
		Name:        inferencePool.Name,
	}
	return endpointPool
}
