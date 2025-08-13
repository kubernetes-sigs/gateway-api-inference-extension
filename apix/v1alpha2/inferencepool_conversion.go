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

package v1alpha2

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	runtime "k8s.io/apimachinery/pkg/runtime"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

// ConvertTo converts this InferencePool (v1alpha2) to the v1 version.
func (src *InferencePool) ConvertTo() (*v1.InferencePool, error) {
	if src == nil {
		return nil, nil
	}

	v1Extension, err := convertExtensionRefToV1(*src.Spec.ExtensionRef)
	if err != nil {
		return nil, err
	}
	v1Status, err := converStatusToV1(src.Status)
	if err != nil {
		return nil, err
	}
	dst := &v1.InferencePool{
		TypeMeta:   src.TypeMeta,
		ObjectMeta: src.ObjectMeta,
		Spec: v1.InferencePoolSpec{
			TargetPortNumber: src.Spec.TargetPortNumber,
			ExtensionRef:     v1Extension,
		},
		Status: *v1Status,
	}
	if src.Spec.Selector != nil {
		dst.Spec.Selector.MatchLabels = make(map[v1.LabelKey]v1.LabelValue, len(src.Spec.Selector))
		for k, v := range src.Spec.Selector {
			dst.Spec.Selector.MatchLabels[v1.LabelKey(k)] = v1.LabelValue(v)
		}
	}
	return dst, nil
}

// ConvertFrom converts from the v1 version to this version (v1alpha2).
func ConvertFrom(src *v1.InferencePool) (*InferencePool, error) {
	if src == nil {
		return nil, nil
	}

	extensionRef, err := convertExtensionRefFromV1(*src.Spec.ExtensionRef)
	if err != nil {
		return nil, err
	}
	status, err := converStatusFromV1(src.Status)
	if err != nil {
		return nil, err
	}
	dst := &InferencePool{
		TypeMeta: metav1.TypeMeta{
			Kind:       "InferencePool",
			APIVersion: "inference.networking.x-k8s.io/v1alpha2",
		},
		ObjectMeta: src.ObjectMeta,
		Spec: InferencePoolSpec{
			TargetPortNumber: src.Spec.TargetPortNumber,
			ExtensionRef:     extensionRef,
		},
		Status: *status,
	}

	if src.Spec.Selector.MatchLabels != nil {
		dst.Spec.Selector = make(map[LabelKey]LabelValue, len(src.Spec.Selector.MatchLabels))
		for k, v := range src.Spec.Selector.MatchLabels {
			dst.Spec.Selector[LabelKey(k)] = LabelValue(v)
		}
	}

	return dst, nil
}

func converStatusToV1(src InferencePoolStatus) (*v1.InferencePoolStatus, error) {
	u, err := toUnstructured(&src)
	if err != nil {
		return nil, err
	}
	return convert[v1.InferencePoolStatus](u)
}

func converStatusFromV1(src v1.InferencePoolStatus) (*InferencePoolStatus, error) {
	u, err := toUnstructured(&src)
	if err != nil {
		return nil, err
	}
	return convert[InferencePoolStatus](u)
}

func convertExtensionRefToV1(src Extension) (*v1.Extension, error) {
	u, err := toUnstructured(&src)
	if err != nil {
		return nil, err
	}
	return convert[v1.Extension](u)
}

func convertExtensionRefFromV1(src v1.Extension) (*Extension, error) {
	u, err := toUnstructured(&src)
	if err != nil {
		return nil, err
	}
	extension, err := convert[Extension](u)
	if err != nil {
		return nil, err
	}
	return extension, nil
}

func toUnstructured(obj any) (*unstructured.Unstructured, error) {
	u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	return &unstructured.Unstructured{Object: u}, nil
}

func convert[T any](u *unstructured.Unstructured) (*T, error) {
	var res T
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(u.Object, &res); err != nil {
		return nil, fmt.Errorf("error converting unstructured to T: %v", err)
	}
	return &res, nil
}
