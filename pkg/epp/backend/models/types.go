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

// ModelsResponse is the response of /v1/models Request.
type ModelsResponse struct {
	// Object is the object type, which is always "list".
	Object string `json:"object"`
	// Data contains list of ModelInfo objects.
	Data []ModelInfo `json:"data"`
}

// ModelInfo describes an OpenAI compatible model.
type ModelInfo struct {
	// ID is the model identifier, which can be referenced in the API endpoints.
	ID string `json:"id"`
	// Object is the object type, which is always "model".
	Object string `json:"object"`
	// Created is the Unix timestamp (in seconds) when the model was created.
	Created int64 `json:"created"`
	// OwnedBy is the organization that owns the model.
	OwnedBy string `json:"owned_by"`
}
