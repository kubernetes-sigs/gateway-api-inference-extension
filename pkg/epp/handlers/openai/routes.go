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

package handlers

var (
	// PassthroughEndpoints are informational endpoints that do not have a model param,
	// and do NOT run inference, so can be passed to any underlying model server at random.
	PassthroughEndpoints map[string]bool = map[string]bool{
		// https://platform.openai.com/docs/api-reference/models/list
		"/v1/models": true,
	}

	// RoutableEndpoints DO have a model param and DO run inference, and thus need to
	// be tracked and routed intelligently.
	RoutableEndpoints map[string]bool = map[string]bool{
		// https://platform.openai.com/docs/api-reference/completions/create
		"v1/completions": true,
		// https://platform.openai.com/docs/api-reference/chat/create
		"v1/chat/completions": true,
		// https://platform.openai.com/docs/api-reference/embeddings/create
		"v1/embeddings": true,
		// https://platform.openai.com/docs/api-reference/audio/createTranscription
		"v1/audio/transcriptions": true,
	}
)
