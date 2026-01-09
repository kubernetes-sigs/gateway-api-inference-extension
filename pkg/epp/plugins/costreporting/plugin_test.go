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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	handlerstypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// Test interface satisfaction at compile time.
var _ requestcontrol.ResponseComplete = &Plugin{}

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
				Expression: "request.usage.prompt_tokens",
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
				Expression: "request.usage.prompt_tokens",
			},
			wantErr: false,
			wantNS:  "custom-ns",
		},
		{

			name: "invalid config - missing name",
			config: Config{
				Metric:     Metric{},
				Expression: "request.usage.prompt_tokens",
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

func TestCostReporting_usage(t *testing.T) {
	logger := zap.New(zap.UseDevMode(true))
	tests := []struct {
		name       string
		config     Config
		response   *requestcontrol.Response
		wantResult *structpb.Struct
		wantErr    bool
	}{
		{
			name: "request usage expression",
			config: Config{
				Metric: Metric{
					Name: "prompt_tokens",
				},
				Expression: "request.usage.prompt_tokens",
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
				Expression: "request.usage.prompt_tokens",
				Condition:  "has(request.usage.prompt_tokens)",
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plugin, err := New(tt.config, logger)
			if err != nil {
				t.Fatalf("Failed to create cost reporting plugin: %v", err)
			}

			// processBody call for request dataSource
			plugin.ResponseComplete(context.Background(), &schedulingtypes.LLMRequest{}, tt.response, &backend.Pod{})
			if (err != nil) != tt.wantErr {
				t.Errorf("processBody() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if diff := cmp.Diff(tt.wantResult, tt.response.DynamicMetadata, protocmp.Transform()); diff != "" {
				t.Errorf("processBody() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
