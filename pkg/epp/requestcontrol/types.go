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

package requestcontrol

// Response contains information from the response received to be passed to the Response requestcontrol plugins
type Response struct {
	// RequestId is the Envoy generated Id for the request being processed
	RequestId string
	// Headers is a map of the response headers. Nil during body processing
	Headers map[string]string
	// Body Is the body of the response or nil during header processing
	Body string
	// IsStreaming indicates whether or not the response is being streamed by the model
	IsStreaming bool
	// EndOfStream when true indicates that this invocation contains the last chunk of the response
	EndOfStream bool
	// ReqMetadata is a map of metadata that can be passed from Envoy.
	// It is populated with Envoy's dynamic metadata when ext_proc is processing ProcessingRequest_ResponseHeaders.
	// Currently, this is only used by conformance test.
	ReqMetadata map[string]any
}
