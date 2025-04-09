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

package inferencemodel

import (
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
)

func IsCritical(model *v1alpha2.InferenceModel) bool {
	if model.Spec.Criticality != nil && *model.Spec.Criticality == v1alpha2.Critical {
		return true
	}
	return false
}
