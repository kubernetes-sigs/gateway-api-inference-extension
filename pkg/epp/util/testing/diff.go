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

package testing

import (
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"sigs.k8s.io/gateway-api-inference-extension/apisx/v1alpha2"
)

func DiffModelLists(want, got []*v1alpha2.InferenceModel) string {
	return cmp.Diff(want, got, cmpopts.SortSlices(func(a, b *v1alpha2.InferenceModel) bool { return a.Name < b.Name }))
}
