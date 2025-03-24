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
// Code generated by client-gen. DO NOT EDIT.

package fake

import (
	gentype "k8s.io/client-go/gentype"
	v1alpha2 "sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	apiv1alpha2 "sigs.k8s.io/gateway-api-inference-extension/client-go/applyconfiguration/api/v1alpha2"
	typedapiv1alpha2 "sigs.k8s.io/gateway-api-inference-extension/client-go/clientset/versioned/typed/api/v1alpha2"
)

// fakeInferenceModels implements InferenceModelInterface
type fakeInferenceModels struct {
	*gentype.FakeClientWithListAndApply[*v1alpha2.InferenceModel, *v1alpha2.InferenceModelList, *apiv1alpha2.InferenceModelApplyConfiguration]
	Fake *FakeInferenceV1alpha2
}

func newFakeInferenceModels(fake *FakeInferenceV1alpha2, namespace string) typedapiv1alpha2.InferenceModelInterface {
	return &fakeInferenceModels{
		gentype.NewFakeClientWithListAndApply[*v1alpha2.InferenceModel, *v1alpha2.InferenceModelList, *apiv1alpha2.InferenceModelApplyConfiguration](
			fake.Fake,
			namespace,
			v1alpha2.SchemeGroupVersion.WithResource("inferencemodels"),
			v1alpha2.SchemeGroupVersion.WithKind("InferenceModel"),
			func() *v1alpha2.InferenceModel { return &v1alpha2.InferenceModel{} },
			func() *v1alpha2.InferenceModelList { return &v1alpha2.InferenceModelList{} },
			func(dst, src *v1alpha2.InferenceModelList) { dst.ListMeta = src.ListMeta },
			func(list *v1alpha2.InferenceModelList) []*v1alpha2.InferenceModel {
				return gentype.ToPointerSlice(list.Items)
			},
			func(list *v1alpha2.InferenceModelList, items []*v1alpha2.InferenceModel) {
				list.Items = gentype.FromPointerSlice(items)
			},
		),
		fake,
	}
}
