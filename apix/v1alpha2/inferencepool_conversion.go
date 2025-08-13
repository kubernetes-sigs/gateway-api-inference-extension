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
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	runtime "k8s.io/apimachinery/pkg/runtime"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

// ConvertTo converts this InferencePool (v1alpha2) to the v1 version.
func (src *InferencePool) ConvertTo(dst *v1.InferencePool) error {
	if dst == nil {
		return errors.New("dst cannot be nil")
	}
	v1Extension, err := convertExtensionRefToV1(src.Spec.ExtensionRef)
	if err != nil {
		return err
	}
	v1Status, err := convertStatusToV1(&src.Status)
	if err != nil {
		return err
	}
	dst.TypeMeta = src.TypeMeta
	dst.ObjectMeta = src.ObjectMeta
	dst.Spec.TargetPorts = []v1.Port{{Number: src.Spec.TargetPortNumber}}
	dst.Spec.ExtensionRef = v1Extension
	dst.Status = *v1Status
	if src.Spec.Selector != nil {
		dst.Spec.Selector.MatchLabels = make(map[v1.LabelKey]v1.LabelValue, len(src.Spec.Selector))
		for k, v := range src.Spec.Selector {
			dst.Spec.Selector.MatchLabels[v1.LabelKey(k)] = v1.LabelValue(v)
		}
	}
	return nil
}

// ConvertFrom converts from the v1 version to this version (v1alpha2).
func (dst *InferencePool) ConvertFrom(src *v1.InferencePool) error {
	if src == nil {
		return errors.New("src cannot be nil")
	}
	extensionRef, err := convertExtensionRefFromV1(src.Spec.ExtensionRef)
	if err != nil {
		return err
	}
	status, err := convertStatusFromV1(&src.Status)
	if err != nil {
		return err
	}
	dst.TypeMeta = src.TypeMeta
	dst.ObjectMeta = src.ObjectMeta
	dst.Spec.TargetPortNumber = src.Spec.TargetPorts[0].Number
	dst.Spec.ExtensionRef = extensionRef
	dst.Status = *status
	if src.Spec.Selector.MatchLabels != nil {
		dst.Spec.Selector = make(map[LabelKey]LabelValue, len(src.Spec.Selector.MatchLabels))
		for k, v := range src.Spec.Selector.MatchLabels {
			dst.Spec.Selector[LabelKey(k)] = LabelValue(v)
		}
	}
	return nil
}

func convertStatusToV1(src *InferencePoolStatus) (*v1.InferencePoolStatus, error) {
	if src == nil {
		return nil, nil
	}
	u, err := toUnstructured(src)
	if err != nil {
		return nil, err
	}
	return convert[v1.InferencePoolStatus](u)
}

func convertStatusFromV1(src *v1.InferencePoolStatus) (*InferencePoolStatus, error) {
	if src == nil {
		return nil, nil
	}
	u, err := toUnstructured(src)
	if err != nil {
		return nil, err
	}
	return convert[InferencePoolStatus](u)
}

func convertExtensionRefToV1(src *Extension) (*v1.Extension, error) {
	if src == nil {
		return nil, nil
	}
	u, err := toUnstructured(src)
	if err != nil {
		return nil, err
	}
	return convert[v1.Extension](u)
}

func convertExtensionRefFromV1(src *v1.Extension) (*Extension, error) {
	if src == nil {
		return nil, nil
	}
	u, err := toUnstructured(src)
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
