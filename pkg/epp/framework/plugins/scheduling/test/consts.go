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

package test

const (
	// HeaderTestEppEndPointSelectionKey is the request header used in tests to control
	// Endpoint Picker (EPP) behavior deterministically.
	//
	// The header value is a comma-separated list of endpoint identifiers. Each entry
	// may be in one of the following formats:
	//
	//   - "IP"          — selects all pods whose IP address matches the given value.
	//   - "IP:port"     — selects only pods whose IP and port both match exactly.
	//                     Ports correspond to data-parallel ranks or specific targetPorts.
	//
	// IPv6 addresses are supported, with or without brackets (e.g. "fd00::1" or "[fd00::1]:3002").
	// The returned order matches the order of endpoints specified in the header, and duplicates
	// are ignored.
	//
	// Examples:
	//   "test-epp-endpoint-selection": "10.0.0.7,10.0.0.8:3002"
	//   "test-epp-endpoint-selection": "[fd00::1]:3000,fd00::2"
	HeaderTestEppEndPointSelectionKey = "test-epp-endpoint-selection"
)
