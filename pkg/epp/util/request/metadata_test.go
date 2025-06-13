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

package request

import (
	"testing"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestExtractMetadataValues(t *testing.T) {
	var makeFilterMetadata = func(key string) map[string]*structpb.Struct {
		structVal, _ := structpb.NewStruct(map[string]interface{}{
			"hello": "world",
			"random": map[string]any{
				"hi": "mom",
			},
		})

		return map[string]*structpb.Struct{
			key: structVal,
		}
	}

	tests := []struct {
		name     string
		metadata map[string]*structpb.Struct
		key      string
		expected map[string]any
	}{
		{
			name:     "Exact match",
			metadata: makeFilterMetadata("envoy.lb.subset_hint"),
			key:      MetadataSubsetKey,
			expected: map[string]any{
				"envoy.lb.subset_hint": map[string]interface{}{
					"hello": "world",
					"random": map[string]any{
						"hi": "mom",
					},
				},
			},
		},
		{
			name:     "Non-existent key",
			metadata: makeFilterMetadata("random-key"),
			key:      MetadataSubsetKey,
			expected: map[string]any{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &extProcPb.ProcessingRequest{
				MetadataContext: &corev3.Metadata{
					FilterMetadata: tt.metadata,
				},
			}

			result := ExtractMetadataValues(req, tt.key)
			if diff := cmp.Diff(result, tt.expected); diff != "" {
				t.Errorf("ExtractMetadataValues() unexpected response (-want +got): %v", diff)
			}
		})
	}
}
