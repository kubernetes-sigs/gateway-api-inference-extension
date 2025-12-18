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
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	extproc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

func TestCostReporting_processBody(t *testing.T) {
	logger := zap.New(zap.UseDevMode(true))
	tests := []struct {
		name       string
		config     Config
		body       []byte
		wantResult *extproc.ProcessingResponse
		wantErr    bool
	}{
		{
			name: "simple expression",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				DataSource: "responseBody",
				Expression: "responseBody.usage.prompt_tokens",
			},
			body: []byte(`{"usage": {"prompt_tokens": 10}}`),
			wantResult: &extproc.ProcessingResponse{

				Response: &extproc.ProcessingResponse_ImmediateResponse{
					ImmediateResponse: &extproc.ImmediateResponse{
						Details: "cost reporting plugin",
					},
				},
				DynamicMetadata: &structpb.Struct{
					Fields: map[string]*structpb.Value{
						DefaultNamespace: {
							Kind: &structpb.Value_StructValue{
								StructValue: &structpb.Struct{
									Fields: map[string]*structpb.Value{
										"prompt_tokens": {Kind: &structpb.Value_StringValue{StringValue: "10"}},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "expression with addition",
			config: Config{
				Metric: Metric{
					Name: "total_tokens",
				},
				DataSource: "responseBody",
				Expression: "responseBody.usage.prompt_tokens + responseBody.usage.completion_tokens",
			},
			body: []byte(`{"usage": {"prompt_tokens": 10, "completion_tokens": 20}}`),
			wantResult: &extproc.ProcessingResponse{
				Response: &extproc.ProcessingResponse_ImmediateResponse{
					ImmediateResponse: &extproc.ImmediateResponse{
						Details: "cost reporting plugin",
					},
				},
				DynamicMetadata: &structpb.Struct{
					Fields: map[string]*structpb.Value{
						DefaultNamespace: {
							Kind: &structpb.Value_StructValue{
								StructValue: &structpb.Struct{
									Fields: map[string]*structpb.Value{
										"total_tokens": {Kind: &structpb.Value_StringValue{StringValue: "30"}},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "with condition true",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				DataSource: "responseBody",
				Expression: "responseBody.usage.prompt_tokens",
				Condition:  "has(responseBody.usage)",
			},
			body: []byte(`{"usage": {"prompt_tokens": 10}}`),
			wantResult: &extproc.ProcessingResponse{
				Response: &extproc.ProcessingResponse_ImmediateResponse{
					ImmediateResponse: &extproc.ImmediateResponse{
						Details: "cost reporting plugin",
					},
				},
				DynamicMetadata: &structpb.Struct{
					Fields: map[string]*structpb.Value{
						DefaultNamespace: {
							Kind: &structpb.Value_StructValue{
								StructValue: &structpb.Struct{
									Fields: map[string]*structpb.Value{
										"prompt_tokens": {Kind: &structpb.Value_StringValue{StringValue: "10"}},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "with condition false",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				DataSource: "responseBody",
				Expression: "responseBody.usage.prompt_tokens",
				Condition:  "has(responseBody.other)",
			},
			body:       []byte(`{"usage": {"prompt_tokens": 10}}`),
			wantResult: &extproc.ProcessingResponse{},
		},
		{
			name: "custom namespace",
			config: Config{
				Metric: Metric{
					Namespace: "my.namespace",
					Name:      "prompt_tokens",
				},
				DataSource: "responseBody",
				Expression: "responseBody.usage.prompt_tokens",
			},
			body: []byte(`{"usage": {"prompt_tokens": 10}}`),
			wantResult: &extproc.ProcessingResponse{
				Response: &extproc.ProcessingResponse_ImmediateResponse{
					ImmediateResponse: &extproc.ImmediateResponse{
						Details: "cost reporting plugin",
					},
				},
				DynamicMetadata: &structpb.Struct{
					Fields: map[string]*structpb.Value{
						"my.namespace": {
							Kind: &structpb.Value_StructValue{
								StructValue: &structpb.Struct{
									Fields: map[string]*structpb.Value{
										"prompt_tokens": {Kind: &structpb.Value_StringValue{StringValue: "10"}},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "custom key",
			config: Config{
				Metric: Metric{
					Name: "custom_prompt_tokens",
				},
				DataSource: "responseBody",
				Expression: "responseBody.usage.prompt_tokens",
			},
			body: []byte(`{"usage": {"prompt_tokens": 10}}`),
			wantResult: &extproc.ProcessingResponse{
				Response: &extproc.ProcessingResponse_ImmediateResponse{
					ImmediateResponse: &extproc.ImmediateResponse{
						Details: "cost reporting plugin",
					},
				},
				DynamicMetadata: &structpb.Struct{
					Fields: map[string]*structpb.Value{
						DefaultNamespace: {
							Kind: &structpb.Value_StructValue{
								StructValue: &structpb.Struct{
									Fields: map[string]*structpb.Value{
										"custom_prompt_tokens": {Kind: &structpb.Value_StringValue{StringValue: "10"}},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "invalid JSON body",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				DataSource: "responseBody",
				Expression: "responseBody.usage.prompt_tokens",
			},
			body:       []byte(`{"usage": {"prompt_tokens": 10`),
			wantResult: &extproc.ProcessingResponse{},
		},
		{
			name: "expression error - field not found",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				DataSource: "responseBody",
				Expression: "responseBody.usage.missing_field",
			},
			body:       []byte(`{"usage": {"prompt_tokens": 10}}`),
			wantResult: &extproc.ProcessingResponse{},
		},
		{
			name: "expression error - type error",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				DataSource: "responseBody",
				Expression: "responseBody.usage.prompt_tokens + 'abc'",
			},
			body:       []byte(`{"usage": {"prompt_tokens": 10}}`),
			wantResult: &extproc.ProcessingResponse{},
		},
		{
			name: "condition error",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				DataSource: "responseBody",
				Expression: "responseBody.usage.prompt_tokens",
				Condition:  "responseBody.usage.prompt_tokens > 'abc'",
			},
			body:       []byte(`{"usage": {"prompt_tokens": 10}}`),
			wantResult: &extproc.ProcessingResponse{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plugin, err := New(tt.config, logger)
			if err != nil {
				t.Fatalf("Failed to create cost reporting plugin: %v", err)
			}

			gotResult, err := plugin.processBody(context.Background(), tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("processBody() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if diff := cmp.Diff(tt.wantResult, gotResult, protocmp.Transform()); diff != "" {
				t.Errorf("processBody() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNewCostReporting(t *testing.T) {
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
				Expression: "responseBody.usage.prompt_tokens",
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
				Expression: "responseBody.usage.prompt_tokens",
			},
			wantErr: false,
			wantNS:  "custom-ns",
		},
		{

			name: "invalid config - missing name",
			config: Config{
				Metric:     Metric{},
				Expression: "responseBody.usage.prompt_tokens",
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
			name: "invalid config - unsupported dataSource",
			config: Config{
				Metric: Metric{
					Name: "test-metric",
				},
				Expression: "responseBody.usage.prompt_tokens",
				DataSource: "unsupported",
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
