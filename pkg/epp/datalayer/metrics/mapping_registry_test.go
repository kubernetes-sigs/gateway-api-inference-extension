/*
Copyright 2026 The Kubernetes Authors.

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

package metrics

import (
	"testing"
)

func TestMappingRegistry(t *testing.T) {
	r := NewMappingRegistry()

	mDef := &Mapping{TotalQueuedRequests: &Spec{Name: "vllm:num_requests_waiting"}}
	mVllm := &Mapping{TotalQueuedRequests: &Spec{Name: "vllm_custom_waiting"}}

	// Test Registration
	if err := r.Register(DefaultEngineType, mDef); err != nil {
		t.Errorf("failed to register default: %v", err)
	}
	if err := r.Register("vllm", mVllm); err != nil {
		t.Errorf("failed to register vllm: %v", err)
	}

	// Test Get
	tests := []struct {
		engine   string
		wantName string
		wantOk   bool
	}{
		{"vllm", "vllm_custom_waiting", true},
		{"sglang", "vllm:num_requests_waiting", true}, // fallback
		{"", "vllm:num_requests_waiting", true},       // defaults to "default"
		{"default", "vllm:num_requests_waiting", true},
	}

	for _, tt := range tests {
		m, ok := r.Get(tt.engine)
		if ok != tt.wantOk {
			t.Errorf("Get(%q) ok = %v, want %v", tt.engine, ok, tt.wantOk)
			continue
		}
		if ok && m.TotalQueuedRequests.Name != tt.wantName {
			t.Errorf("Get(%q) name = %q, want %q", tt.engine, m.TotalQueuedRequests.Name, tt.wantName)
		}
	}

	// Test without default
	r2 := NewMappingRegistry()
	_ = r2.Register("vllm", mVllm)
	if _, ok := r2.Get("sglang"); ok {
		t.Error("expected sglang to not be found when no default exists")
	}
}
