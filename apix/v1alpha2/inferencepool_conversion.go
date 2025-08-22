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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	runtime "k8s.io/apimachinery/pkg/runtime"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

// ConvertTo converts this InferencePool (v1alpha2) to the v1 version.
func (src *InferencePool) ConvertTo(dst *v1.InferencePool) error {
	if dst == nil {
		return errors.New("dst cannot be nil")
	}
	endpointPickRef, err := convertExtensionRefToV1(&src.Spec.ExtensionRef)
	if err != nil {
		return err
	}
	v1Status, err := convertStatusToV1(&src.Status)
	if err != nil {
		return err
	}

	// v1 requires at least one condition per parent.
	ensureV1StatusConditionsMinimum(v1Status)

	meta := metav1.TypeMeta{
		Kind:       src.Kind,
		APIVersion: v1.GroupVersion.String(), // Ensure the API version is set correctly.
	}
	dst.TypeMeta = meta
	dst.ObjectMeta = src.ObjectMeta
	dst.Spec.TargetPorts = []v1.Port{{Number: v1.PortNumber(src.Spec.TargetPortNumber)}}
	dst.Spec.EndpointPickerRef = endpointPickRef
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
	extensionRef, err := convertEndpointPickerRefFromV1(&src.Spec.EndpointPickerRef)
	if err != nil {
		return err
	}
	status, err := convertStatusFromV1(&src.Status)
	if err != nil {
		return err
	}

	meta := metav1.TypeMeta{
		Kind:       src.Kind,
		APIVersion: GroupVersion.String(), // Ensure the API version is set correctly.
	}
	dst.TypeMeta = meta
	dst.ObjectMeta = src.ObjectMeta
	dst.Spec.TargetPortNumber = int32(src.Spec.TargetPorts[0].Number)
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
		return nil, errors.New("src cannot be nil")
	}
	u, err := toUnstructured(src)
	if err != nil {
		return nil, err
	}

	// v1alpha2 uses json:"parent" (singular), v1 uses json:"parents"
	if v, ok := u.Object["parent"]; ok {
		u.Object["parents"] = v
		delete(u.Object, "parent")
	}

	out, err := convert[v1.InferencePoolStatus](u)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func convertStatusFromV1(src *v1.InferencePoolStatus) (*InferencePoolStatus, error) {
	if src == nil {
		return nil, errors.New("src cannot be nil")
	}
	u, err := toUnstructured(src)
	if err != nil {
		return nil, err
	}

	// v1 uses json:"parents", v1alpha2 uses json:"parent" (singular)
	if v, ok := u.Object["parents"]; ok {
		u.Object["parent"] = v
		delete(u.Object, "parents")
	}

	out, err := convert[InferencePoolStatus](u)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func convertExtensionRefToV1(src *Extension) (v1.EndpointPickerRef, error) {
	endpointPickerRef := v1.EndpointPickerRef{}
	if src == nil {
		return endpointPickerRef, errors.New("src cannot be nil")
	}
	if src.Group != nil {
		v1Group := v1.Group(*src.Group)
		endpointPickerRef.Group = &v1Group
	}
	if src.Kind != nil {
		endpointPickerRef.Kind = v1.Kind(*src.Kind)
	}
	endpointPickerRef.Name = v1.ObjectName(src.Name)
	if src.PortNumber != nil {
		v1PortNumber := v1.PortNumber(*src.PortNumber)
		endpointPickerRef.PortNumber = &v1PortNumber
	}
	if src.FailureMode != nil {
		endpointPickerRef.FailureMode = v1.ExtensionFailureMode(*src.FailureMode)
	}

	return endpointPickerRef, nil
}

func convertEndpointPickerRefFromV1(src *v1.EndpointPickerRef) (Extension, error) {
	extension := Extension{}
	if src == nil {
		return extension, errors.New("src cannot be nil")
	}
	if src.Group != nil {
		group := Group(*src.Group)
		extension.Group = &group
	}
	if src.Kind != "" {
		kind := Kind(src.Kind)
		extension.Kind = &kind
	}
	extension.Name = ObjectName(src.Name)
	if src.PortNumber != nil {
		portNumber := PortNumber(*src.PortNumber)
		extension.PortNumber = &portNumber
	}
	if src.FailureMode != "" {
		extensionFailureMode := ExtensionFailureMode(src.FailureMode)
		extension.FailureMode = &extensionFailureMode
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

// Ensure v1's PoolStatus.Conditions has at least one entry.
func ensureV1StatusConditionsMinimum(s *v1.InferencePoolStatus) {
	if s == nil {
		return
	}
	epoch := metav1.NewTime(time.Unix(0, 0))
	for i := range s.Parents {
		ps := &s.Parents[i]
		if len(ps.Conditions) == 0 {
			ps.Conditions = []metav1.Condition{{
				Type:               string(v1.InferencePoolConditionAccepted),
				Status:             metav1.ConditionUnknown,
				Reason:             string(v1.InferencePoolReasonPending),
				Message:            "Waiting for controller",
				LastTransitionTime: epoch,
			}}
		}
	}
}
