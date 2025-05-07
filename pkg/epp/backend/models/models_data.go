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

package models

import (
	podinfo "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/pod-info"
)

var _ podinfo.ScrapedData = &ModelsData{}

// NewModelsData returns an empty ModelsData struct.
func NewModelsData() *ModelsData {
	return &ModelsData{
		ModelIDs: make([]string, 0),
	}
}

// ModelsData is a struct that models list of available models for a pod.
type ModelsData struct {
	ModelIDs []string
}

func (md *ModelsData) Clone() podinfo.ScrapedData {
	copy := make([]string, len(md.ModelIDs))
	for i, modelID := range md.ModelIDs {
		copy[i] = modelID
	}

	return &ModelsData{ModelIDs: copy}
}
