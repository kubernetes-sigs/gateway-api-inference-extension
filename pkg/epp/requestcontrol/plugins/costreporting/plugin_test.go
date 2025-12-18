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

package costreporting

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"

	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/scheduling"
	handlerstypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
)

// Test interface satisfaction at compile time.
var _ requestcontrol.ResponseComplete = &Plugin{}

func TestPluginCreation(t *testing.T) {
	logger := zap.New(zap.UseDevMode(true))
	tests := []struct {
		name    string
		config  Config
		wantErr bool
		wantNS  string
	}{
		{
			name: "valid config with default namespace",
			config: Config{
				Metric: Metric{
					Name: "test-metric",
				},
				Expression: "usage.prompt_tokens",
			},
			wantErr: false,
			wantNS:  DefaultNamespace,
		},
		{
			name: "valid config with custom namespace",
			config: Config{
				Metric: Metric{
					Name:      "test-metric",
					Namespace: "custom-ns",
				},
				Expression: "usage.prompt_tokens",
			},
			wantErr: false,
			wantNS:  "custom-ns",
		},
		{
			name: "invalid config - missing name",
			config: Config{
				Metric:     Metric{},
				Expression: "usage.prompt_tokens",
			},
			wantErr: true,
		},
		{
			name: "invalid config - missing expression",
			config: Config{
				Metric: Metric{
					Name: "test-metric",
				},
			},
			wantErr: true,
		},
		{
			name: "invalid config - invalid expression CEL syntax",
			config: Config{
				Metric: Metric{
					Name: "test-metric",
				},
				Expression: "usage.prompt_tokens + -",
			},
			wantErr: true,
		},
		{
			name: "invalid config - invalid condition CEL syntax",
			config: Config{
				Metric: Metric{
					Name: "test-metric",
				},
				Expression: "usage.prompt_tokens",
				Condition:  "usage.prompt_tokens > ",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plugin, err := New(tt.config, logger)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewCostReporting() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if plugin == nil {
					t.Errorf("Expected plugin to not be nil")
				}
				metric := plugin.config.Metric
				if metric.Namespace != tt.wantNS {
					t.Errorf("Expected namespace %s, got %s", tt.wantNS, metric.Namespace)
				}
			}
		})
	}
}

func TestCostReporting(t *testing.T) {
	logger := zap.New(zap.UseDevMode(true))
	tests := []struct {
		name       string
		config     Config
		response   *requestcontrol.Response
		wantResult *structpb.Struct
	}{
		{
			name: "request usage expression",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				Expression: "usage.prompt_tokens",
			},
			response: &requestcontrol.Response{
				Usage: handlerstypes.Usage{
					PromptTokens: 15,
				},
			},
			wantResult: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					DefaultNamespace: {
						Kind: &structpb.Value_StructValue{
							StructValue: &structpb.Struct{
								Fields: map[string]*structpb.Value{
									"prompt_tokens": {Kind: &structpb.Value_NumberValue{NumberValue: 15}},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "request usage expression and condition with zero value",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				Expression: "usage.prompt_tokens",
				Condition:  "has(usage.prompt_tokens)",
			},
			response: &requestcontrol.Response{
				Usage: handlerstypes.Usage{
					PromptTokens: 0,
				},
			},
			wantResult: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					DefaultNamespace: {
						Kind: &structpb.Value_StructValue{
							StructValue: &structpb.Struct{
								Fields: map[string]*structpb.Value{
									"prompt_tokens": {Kind: &structpb.Value_NumberValue{NumberValue: 0}},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "condition not met",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				Expression: "usage.prompt_tokens",
				Condition:  "usage.completion_tokens > 0",
			},
			response: &requestcontrol.Response{
				Usage: handlerstypes.Usage{
					PromptTokens: 10,
				},
				DynamicMetadata: &structpb.Struct{
					Fields: map[string]*structpb.Value{
						"pre-existing": {Kind: &structpb.Value_StringValue{StringValue: "data"}},
					},
				},
			},
			wantResult: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					"pre-existing": {Kind: &structpb.Value_StringValue{StringValue: "data"}},
				},
			}, // Expect no changes to DynamicMetadata
		},
		{
			name: "condition evaluates to non-boolean",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				Expression: "usage.prompt_tokens",
				Condition:  "usage.prompt_tokens", // This is an int, not bool
			},
			response: &requestcontrol.Response{
				Usage: handlerstypes.Usage{
					PromptTokens: 10,
				},
			},
			wantResult: nil, // Expect no result, error logged
		},
		{
			name: "expression evaluates to non-numeric",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				Expression: "'not a number'",
			},
			response: &requestcontrol.Response{
				Usage: handlerstypes.Usage{
					PromptTokens: 10,
				},
			},
			wantResult: nil, // Expect no result, error logged
		},
		{
			name: "expression runtime error - field not found",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				Expression: "usage.non_existent_field", // No has() guard
			},
			response: &requestcontrol.Response{
				Usage: handlerstypes.Usage{
					PromptTokens: 10,
				},
			},
			wantResult: nil, // Expect no result, error logged
		},
		{
			name: "usage fields missing with has() guards",
			config: Config{
				Metric: Metric{
					Name: "total_tokens",
				},
				Expression: "(has(usage.prompt_tokens) ? usage.prompt_tokens : 0) + (has(usage.completion_tokens) ? usage.completion_tokens : 0)",
			},
			response: &requestcontrol.Response{
				Usage: handlerstypes.Usage{}, // Empty usage
			},
			wantResult: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					DefaultNamespace: {
						Kind: &structpb.Value_StructValue{
							StructValue: &structpb.Struct{
								Fields: map[string]*structpb.Value{
									"total_tokens": {Kind: &structpb.Value_NumberValue{NumberValue: 0}},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "partial usage fields missing with has() guards",
			config: Config{
				Metric: Metric{
					Name: "total_tokens",
				},
				Expression: "(has(usage.prompt_tokens) ? usage.prompt_tokens : 0) + (has(usage.completion_tokens) ? usage.completion_tokens : 0)",
			},
			response: &requestcontrol.Response{
				Usage: handlerstypes.Usage{
					CompletionTokens: 25,
				},
			},
			wantResult: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					DefaultNamespace: {
						Kind: &structpb.Value_StructValue{
							StructValue: &structpb.Struct{
								Fields: map[string]*structpb.Value{
									"total_tokens": {Kind: &structpb.Value_NumberValue{NumberValue: 25}},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clone the initial response object for each subtest
			currentResponse := &requestcontrol.Response{
				Usage: tt.response.Usage,
			}
			if tt.response.DynamicMetadata != nil {
				currentResponse.DynamicMetadata = proto.Clone(tt.response.DynamicMetadata).(*structpb.Struct)
			}

			plugin, err := New(tt.config, logger)
			if err != nil {
				t.Fatalf("Failed to create cost reporting plugin: %v", err)
			}

			plugin.ResponseComplete(context.Background(), &scheduling.LLMRequest{}, currentResponse, &datalayer.EndpointMetadata{})

			if diff := cmp.Diff(tt.wantResult, currentResponse.DynamicMetadata, protocmp.Transform()); diff != "" {
				t.Errorf("ResponseComplete() DynamicMetadata mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
