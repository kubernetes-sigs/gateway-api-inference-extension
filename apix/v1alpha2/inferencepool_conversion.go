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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

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

	meta := metav1.TypeMeta{
		Kind:       src.Kind,
		APIVersion: v1.GroupVersion.String(), // Ensure the API version is set correctly.
	}
	dst.TypeMeta = meta
	dst.ObjectMeta = src.ObjectMeta
	dst.Spec.TargetPorts = []v1.Port{{Number: v1.PortNumber(src.Spec.TargetPortNumber)}}
	dst.Spec.EndpointPickerRef = endpointPickRef
	dst.Status = v1Status

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
	status, err := convertStatusFromV1(src.Status)
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
	if src.Parents == nil {
		return &v1.InferencePoolStatus{}, nil
	}
	out := &v1.InferencePoolStatus{
		Parents: make([]v1.ParentStatus, 0, len(src.Parents)),
	}
	for _, p := range src.Parents {
		ps := v1.ParentStatus{
			ParentRef:  toV1ParentRef(p.GatewayRef),
			Conditions: make([]metav1.Condition, 0, len(p.Conditions)),
		}
		for _, c := range p.Conditions {
			cc := c
			// v1alpha2: "Accepted" -> v1: "SupportedByParent"
			if cc.Type == string(v1.InferencePoolConditionAccepted) &&
				cc.Reason == string(InferencePoolReasonAccepted) {
				cc.Reason = string(v1.InferencePoolReasonAccepted)
			}
			ps.Conditions = append(ps.Conditions, cc)
		}
		out.Parents = append(out.Parents, ps)
	}
	return out, nil
}

func convertStatusFromV1(src *v1.InferencePoolStatus) (*InferencePoolStatus, error) {
	if src == nil {
		return nil, errors.New("src cannot be nil")
	}
	if src.Parents == nil {
		return &InferencePoolStatus{}, nil
	}
	out := &InferencePoolStatus{
		Parents: make([]PoolStatus, 0, len(src.Parents)),
	}
	for _, p := range src.Parents {
		ps := PoolStatus{
			GatewayRef: fromV1ParentRef(p.ParentRef),
			Conditions: make([]metav1.Condition, 0, len(p.Conditions)),
		}
		for _, c := range p.Conditions {
			cc := c
			// v1: "SupportedByParent" -> v1alpha2: "Accepted"
			if cc.Type == string(v1.InferencePoolConditionAccepted) &&
				cc.Reason == string(v1.InferencePoolReasonAccepted) {
				cc.Reason = string(InferencePoolReasonAccepted)
			}
			ps.Conditions = append(ps.Conditions, cc)
		}
		out.Parents = append(out.Parents, ps)
	}
	return out, nil
}

func toV1ParentRef(in ParentGatewayReference) v1.ParentReference {
	out := v1.ParentReference{
		Name: v1.ObjectName(in.Name),
	}
	if in.Group != nil {
		g := v1.Group(*in.Group)
		out.Group = &g
	}
	if in.Kind != nil {
		k := v1.Kind(*in.Kind)
		out.Kind = &k
	}
	if in.Namespace != nil {
		ns := v1.Namespace(*in.Namespace)
		out.Namespace = &ns
	}
	return out
}

func fromV1ParentRef(in v1.ParentReference) ParentGatewayReference {
	out := ParentGatewayReference{
		Name: ObjectName(in.Name),
	}
	if in.Group != nil {
		g := Group(*in.Group)
		out.Group = &g
	}
	if in.Kind != nil {
		k := Kind(*in.Kind)
		out.Kind = &k
	}
	if in.Namespace != nil {
		ns := Namespace(*in.Namespace)
		out.Namespace = &ns
	}
	return out
}

func convertExtensionRefToV1(src *Extension) (v1.EndpointPickerRef, error) {
	endpointPickerRef := v1.EndpointPickerRef{}
	if src == nil {
		return endpointPickerRef, errors.New("src cannot be nil")
	}
	if src.Group != nil {
		endpointPickerRef.Group = ptr.To(v1.Group(*src.Group))
	}
	if src.Kind != nil {
		endpointPickerRef.Kind = ptr.To(v1.Kind(*src.Kind))
	}
	endpointPickerRef.Name = v1.ObjectName(src.Name)
	if src.PortNumber != nil {
		endpointPickerRef.PortNumber = ptr.To(v1.PortNumber(*src.PortNumber))
	}
	if src.FailureMode != nil {
		endpointPickerRef.FailureMode = ptr.To(v1.EndpointPickerFailureMode(*src.FailureMode))
	}

	return endpointPickerRef, nil
}

func convertEndpointPickerRefFromV1(src *v1.EndpointPickerRef) (Extension, error) {
	extension := Extension{}
	if src == nil {
		return extension, errors.New("src cannot be nil")
	}
	if src.Group != nil {
		extension.Group = ptr.To(Group(*src.Group))
	}
	if src.Kind != nil {
		extension.Kind = ptr.To(Kind(*src.Kind))
	}
	extension.Name = ObjectName(src.Name)
	if src.PortNumber != nil {
		extension.PortNumber = ptr.To(PortNumber(*src.PortNumber))
	}
	if src.FailureMode != nil {
		extension.FailureMode = ptr.To(ExtensionFailureMode(*src.FailureMode))
	}
	return extension, nil
}
