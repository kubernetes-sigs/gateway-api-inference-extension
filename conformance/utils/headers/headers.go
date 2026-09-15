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

package headers

const (
	HeaderTestEppEndPointSelectionKey = "test-epp-endpoint-selection"
	ConformanceTestResultHeader       = "x-conformance-test-served-endpoint"
	// ConformanceTestSelectedHeader carries the endpoint the picker selected, as
	// opposed to ConformanceTestResultHeader which carries the endpoint that served.
	ConformanceTestSelectedHeader = "x-conformance-test-selected-endpoint"
	// ConformanceTestPoolHeader carries the name of the InferencePool whose endpoint
	// picker selected the endpoint, so a test spanning several pools can tell which
	// pool's picker was consulted.
	ConformanceTestPoolHeader = "x-conformance-test-epp-pool"
)
