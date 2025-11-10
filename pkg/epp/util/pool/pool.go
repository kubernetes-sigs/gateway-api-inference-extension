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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	v1alpha2 "sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

func InferencePoolToEndPointsPool(inferencePool *v1.InferencePool) *datalayer.EndPointsPool {
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
	gknn := common.GKNN{
		NamespacedName: types.NamespacedName{Namespace: inferencePool.Namespace, Name: inferencePool.Name},
		GroupKind:      schema.GroupKind{Group: "inference.networking.k8s.io", Kind: "InferencePool"},
	}
	endPoints := &datalayer.EndPoints{
		Selector:    selector,
		TargetPorts: targetPorts,
	}
	endPointsPool := &datalayer.EndPointsPool{
		EndPoints:      endPoints,
		StandaloneMode: false,
		GKNN:           gknn,
	}
	return endPointsPool
}

func AlphaInferencePoolToEndPointsPool(inferencePool *v1alpha2.InferencePool) *datalayer.EndPointsPool {
	targetPorts := []int{int(inferencePool.Spec.TargetPortNumber)}
	selector := make(map[string]string, len(inferencePool.Spec.Selector))
	for k, v := range inferencePool.Spec.Selector {
		selector[string(k)] = string(v)
	}
	gknn := common.GKNN{
		NamespacedName: types.NamespacedName{Namespace: inferencePool.Namespace, Name: inferencePool.Name},
		GroupKind:      schema.GroupKind{Group: "inference.networking.x-k8s.io", Kind: "InferencePool"},
	}
	endPoints := &datalayer.EndPoints{
		Selector:    selector,
		TargetPorts: targetPorts,
	}
	endPointsPool := &datalayer.EndPointsPool{
		EndPoints:      endPoints,
		StandaloneMode: false,
		GKNN:           gknn,
	}
	return endPointsPool
}

func EndPointsPoolToInferencePool(endPointsPool *datalayer.EndPointsPool) *v1.InferencePool {
	targetPorts := make([]v1.Port, 0, len(endPointsPool.EndPoints.TargetPorts))
	for _, p := range endPointsPool.EndPoints.TargetPorts {
		targetPorts = append(targetPorts, v1.Port{Number: v1.PortNumber(p)})
	}
	labels := make(map[v1.LabelKey]v1.LabelValue, len(endPointsPool.EndPoints.Selector))
	for k, v := range endPointsPool.EndPoints.Selector {
		labels[v1.LabelKey(k)] = v1.LabelValue(v)
	}

	inferencePool := &v1.InferencePool{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "inference.networking.k8s.io/v1",
			Kind:       "InferencePool",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      endPointsPool.GKNN.Name,
			Namespace: endPointsPool.GKNN.Namespace,
		},
		Spec: v1.InferencePoolSpec{
			Selector:    v1.LabelSelector{MatchLabels: labels},
			TargetPorts: targetPorts,
		},
	}
	return inferencePool
}

func ToGKNN(ip *v1.InferencePool) common.GKNN {
	if ip == nil {
		return common.GKNN{}
	}
	return common.GKNN{
		NamespacedName: types.NamespacedName{
			Name:      ip.Name,
			Namespace: ip.Namespace,
		},
		GroupKind: schema.GroupKind{
			Group: ip.GroupVersionKind().Group,
			Kind:  ip.GroupVersionKind().Kind,
		},
	}
}
