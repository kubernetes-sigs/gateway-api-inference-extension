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

package metadata

import "testing"

func TestPathAllowedForMultipartModelExtraction(t *testing.T) {
	tests := []struct {
		path    string
		allowed bool
	}{
		{"/v1/audio/transcriptions", true},
		{"/v1/audio/transcriptions/", true},
		{"/v1/audio/transcriptions?foo=1", true},
		{"/v1/video/test", false},
		{"/v1/completions", false},
		{"", false},
	}
	for _, tt := range tests {
		got := PathAllowedForMultipartModelExtraction(tt.path)
		if got != tt.allowed {
			t.Errorf("PathAllowedForMultipartModelExtraction(%q) = %v, want %v", tt.path, got, tt.allowed)
		}
	}
}
