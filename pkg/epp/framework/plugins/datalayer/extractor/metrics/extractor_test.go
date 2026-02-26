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

package metrics

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/proto"
	"k8s.io/utils/ptr"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	sourcemetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/metrics"
)

const (
	// use hardcoded values - importing causes cycle
	defaultTotalQueuedRequestsMetric    = "vllm:num_requests_waiting"
	defaultTotalRunningRequestsMetric   = "vllm:num_requests_running"
	defaultKvCacheUsagePercentageMetric = "vllm:kv_cache_usage_perc"
	defaultLoraInfoMetric               = "vllm:lora_requests_info"
	defaultCacheInfoMetric              = "vllm:cache_config_info"
)

func TestExtractorExtract(t *testing.T) {
	ctx := context.Background()

	if _, err := NewModelServerExtractor(nil, ""); err == nil {
		t.Error("expected to fail to create extractor with nil registry")
	}

	registry := NewMappingRegistry()
	mapping, err := NewMapping(defaultTotalQueuedRequestsMetric, defaultTotalRunningRequestsMetric,
		defaultKvCacheUsagePercentageMetric, defaultLoraInfoMetric, defaultCacheInfoMetric)
	if err != nil {
		t.Fatalf("failed to create mapping: %v", err)
	}
	if err := registry.Register(DefaultEngineType, mapping); err != nil {
		t.Fatalf("failed to register mapping: %v", err)
	}

	extractor, err := NewModelServerExtractor(registry, "")
	if err != nil {
		t.Fatalf("failed to create extractor: %v", err)
	}

	if exType := extractor.TypedName().Type; exType == "" {
		t.Error("empty extractor type")
	}

	if exName := extractor.TypedName().Name; exName == "" {
		t.Error("empty extractor name")
	}

	if inputType := extractor.ExpectedInputType(); inputType != sourcemetrics.PrometheusMetricType {
		t.Errorf("incorrect expected input type: %v", inputType)
	}

	ep := fwkdl.NewEndpoint(nil, nil)
	if ep == nil {
		t.Fatal("expected non-nil endpoint")
	}

	tests := []struct {
		name    string
		data    any
		wantErr bool
		updated bool // whether metrics are expected to change
	}{
		{
			name:    "nil data",
			data:    nil,
			wantErr: true,
			updated: false,
		},
		{
			name:    "empty PrometheusMetricMap",
			data:    sourcemetrics.PrometheusMetricMap{},
			wantErr: true,  // errors when metrics are missing
			updated: false, // and also not updated...
		},
		{
			name: "single valid metric",
			data: sourcemetrics.PrometheusMetricMap{
				defaultTotalQueuedRequestsMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{Value: ptr.To(5.0)},
						},
					},
				},
			},
			wantErr: true, // missing metrics can return an error
			updated: true, // but should still update
		},
		{
			name: "multiple valid metrics",
			data: sourcemetrics.PrometheusMetricMap{
				defaultTotalQueuedRequestsMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{Value: ptr.To(5.0)},
						},
					},
				},
				defaultTotalRunningRequestsMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{Value: ptr.To(1.0)},
						},
					},
				},
				defaultKvCacheUsagePercentageMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{Value: ptr.To(0.5)},
						},
					},
				},
				defaultLoraInfoMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Label: []*dto.LabelPair{
								{
									Name:  proto.String(LoraInfoRunningAdaptersMetricName),
									Value: proto.String("lora1"),
								},
								{
									Name:  proto.String(LoraInfoWaitingAdaptersMetricName),
									Value: proto.String("lora2"),
								},
								{
									Name:  proto.String(LoraInfoMaxAdaptersMetricName),
									Value: proto.String("1"),
								},
							},
						},
					},
				},
				defaultCacheInfoMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Label: []*dto.LabelPair{
								{
									Name:  proto.String(CacheConfigBlockSizeInfoMetricName),
									Value: proto.String("16"),
								},
								{
									Name:  proto.String(CacheConfigNumGPUBlocksMetricName),
									Value: proto.String("1024"),
								},
							},
							Gauge: &dto.Gauge{Value: ptr.To(1.0)},
						},
					},
				},
			},
			wantErr: false,
			updated: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Extract panicked: %v", r)
				}
			}()

			before := ep.GetMetrics().Clone()
			err := extractor.Extract(ctx, tt.data, ep)
			after := ep.GetMetrics()

			if tt.wantErr && err == nil {
				t.Errorf("expected error but got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if tt.updated {
				if diff := cmp.Diff(before, after); diff == "" {
					t.Errorf("expected metrics to be updated, but no change detected")
				}
			} else {
				if diff := cmp.Diff(before, after); diff != "" {
					t.Errorf("expected no metrics update, but got changes:\n%s", diff)
				}
			}
		})
	}
}

func TestExtractorMultiEngine(t *testing.T) {
	ctx := context.Background()

	registry := NewMappingRegistry()
	// Default mapping (vllm)
	mDef, _ := NewMapping("vllm:num_requests_waiting", "vllm:num_requests_running", "", "", "")
	_ = registry.Register(DefaultEngineType, mDef)
	// SGLang mapping
	mSgl, _ := NewMapping("sglang:num_queue_reqs", "sglang:num_running_reqs", "", "", "")
	_ = registry.Register("sglang", mSgl)

	extractor, _ := NewModelServerExtractor(registry, "")

	// Sample metric data
	data := sourcemetrics.PrometheusMetricMap{
		"vllm:num_requests_waiting": &dto.MetricFamily{
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				{
					Gauge: &dto.Gauge{Value: ptr.To(10.0)},
				},
			},
		},
		"sglang:num_queue_reqs": &dto.MetricFamily{
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				{
					Gauge: &dto.Gauge{Value: ptr.To(20.0)},
				},
			},
		},
	}

	// Case 1: Engine = vllm (uses default)
	epVllm := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		Labels: map[string]string{DefaultEngineTypeLabelKey: "vllm"},
	}, nil)
	_ = extractor.Extract(ctx, data, epVllm)
	if epVllm.GetMetrics().WaitingQueueSize != 10 {
		t.Errorf("vllm: expected queue size 10, got %v", epVllm.GetMetrics().WaitingQueueSize)
	}

	// Case 2: Engine = sglang (uses specific)
	epSgl := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		Labels: map[string]string{DefaultEngineTypeLabelKey: "sglang"},
	}, nil)
	_ = extractor.Extract(ctx, data, epSgl)
	if epSgl.GetMetrics().WaitingQueueSize != 20 {
		t.Errorf("sglang: expected queue size 20, got %v", epSgl.GetMetrics().WaitingQueueSize)
	}
}

func TestBackwardCompatibility(t *testing.T) {
	ctx := context.Background()

	registry := NewMappingRegistry()
	// Default mapping (legacy behavior)
	mDef, _ := NewMapping("vllm:num_requests_waiting", "", "", "", "")
	_ = registry.Register(DefaultEngineType, mDef)

	extractor, _ := NewModelServerExtractor(registry, "")

	data := sourcemetrics.PrometheusMetricMap{
		"vllm:num_requests_waiting": &dto.MetricFamily{
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				{
					Gauge: &dto.Gauge{Value: ptr.To(100.0)},
				},
			},
		},
	}

	// Case 1: No labels at all
	epNone := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Labels: nil}, nil)
	_ = extractor.Extract(ctx, data, epNone)
	if epNone.GetMetrics().WaitingQueueSize != 100 {
		t.Errorf("no labels: expected 100, got %v", epNone.GetMetrics().WaitingQueueSize)
	}

	// Case 2: Different label key or unknown value
	epUnknown := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		Labels: map[string]string{DefaultEngineTypeLabelKey: "unknown-engine"},
	}, nil)
	_ = extractor.Extract(ctx, data, epUnknown)
	if epUnknown.GetMetrics().WaitingQueueSize != 100 {
		t.Errorf("unknown label: expected 100, got %v", epUnknown.GetMetrics().WaitingQueueSize)
	}
}

func TestModelServerExtractorFactoryDefaultEngine(t *testing.T) {
	tests := []struct {
		name         string
		params       map[string]any
		wantErr      bool
		errContains  string
		checkDefault string // engine name that should be default
	}{
		{
			name:         "no params uses vllm as default",
			params:       nil,
			wantErr:      false,
			checkDefault: "vllm",
		},
		{
			name: "defaultEngine sglang",
			params: map[string]any{
				"defaultEngine": "sglang",
			},
			wantErr:      false,
			checkDefault: "sglang",
		},
		{
			name: "defaultEngine vllm explicit",
			params: map[string]any{
				"defaultEngine": "vllm",
			},
			wantErr:      false,
			checkDefault: "vllm",
		},
		{
			name: "defaultEngine not found",
			params: map[string]any{
				"defaultEngine": "unknown-engine",
			},
			wantErr:     true,
			errContains: "not found in engineConfigs",
		},
		{
			name: "engine config name is reserved default",
			params: map[string]any{
				"defaultEngine": "default",
				"engineConfigs": []map[string]any{
					{
						"name":               "default",
						"queuedRequestsSpec": "test:metric",
					},
				},
			},
			wantErr:     true,
			errContains: "reserved",
		},
		{
			name: "custom engineConfigs with defaultEngine",
			params: map[string]any{
				"defaultEngine": "custom",
				"engineConfigs": []map[string]any{
					{
						"name":               "custom",
						"queuedRequestsSpec": "test:metric",
					},
				},
			},
			wantErr:      false,
			checkDefault: "custom",
		},
		{
			name: "custom engineConfigs auto-appends vllm and sglang",
			params: map[string]any{
				"engineConfigs": []map[string]any{
					{
						"name":               "triton",
						"queuedRequestsSpec": "nv_trt_llm:waiting",
					},
				},
			},
			wantErr:      false,
			checkDefault: "vllm", // vllm is auto-appended and becomes default
		},
		{
			name: "custom engineConfigs with custom vllm preserves user config",
			params: map[string]any{
				"engineConfigs": []map[string]any{
					{
						"name":               "vllm",
						"queuedRequestsSpec": "custom:vllm_waiting",
					},
				},
			},
			wantErr:      false,
			checkDefault: "vllm", // user's vllm config is used, sglang is auto-appended
		},
		{
			name: "empty engineConfigs uses defaults",
			params: map[string]any{
				"engineConfigs": []map[string]any{},
			},
			wantErr:      false,
			checkDefault: "vllm", // vllm and sglang are auto-appended
		},
		{
			name: "defaultEngine triton with triton defined",
			params: map[string]any{
				"defaultEngine": "triton",
				"engineConfigs": []map[string]any{
					{
						"name":               "triton",
						"queuedRequestsSpec": "nv_trt_llm:waiting",
					},
				},
			},
			wantErr:      false,
			checkDefault: "triton", // triton is default, vllm/sglang auto-appended
		},
		{
			name: "both vllm and sglang custom defined",
			params: map[string]any{
				"engineConfigs": []map[string]any{
					{
						"name":               "vllm",
						"queuedRequestsSpec": "custom:vllm_metric",
					},
					{
						"name":               "sglang",
						"queuedRequestsSpec": "custom:sglang_metric",
					},
				},
			},
			wantErr:      false,
			checkDefault: "vllm", // no auto-append, user's configs used
		},
		{
			name: "duplicate engine names",
			params: map[string]any{
				"engineConfigs": []map[string]any{
					{
						"name":               "vllm",
						"queuedRequestsSpec": "custom:metric1",
					},
					{
						"name":               "vllm",
						"queuedRequestsSpec": "custom:metric2",
					},
				},
			},
			wantErr:     true,
			errContains: "already exists",
		},
		{
			name: "empty engineLabelKey uses default",
			params: map[string]any{
				"engineLabelKey": "",
			},
			wantErr:      false,
			checkDefault: "vllm",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var params json.RawMessage
			if tt.params != nil {
				var err error
				params, err = json.Marshal(tt.params)
				if err != nil {
					t.Fatalf("failed to marshal params: %v", err)
				}
			}

			plugin, err := ModelServerExtractorFactory("test", params, nil)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error but got nil")
				} else if tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errContains)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if plugin == nil {
				t.Fatal("expected non-nil plugin")
			}

			// Verify the correct default is set
			if tt.checkDefault != "" {
				extractor, ok := plugin.(*Extractor)
				if !ok {
					t.Fatal("plugin is not an Extractor")
				}

				// Verify EngineLabelKey is defaulted correctly
				if extractor.engineLabelKey != DefaultEngineTypeLabelKey {
					t.Errorf("engineLabelKey = %q, want %q", extractor.engineLabelKey, DefaultEngineTypeLabelKey)
				}

				// Check that the default mapping exists and matches expected engine
				defaultMapping, found := extractor.registry.Get(DefaultEngineType)
				if !found {
					t.Fatal("default mapping not found in registry")
				}

				engineMapping, found := extractor.registry.Get(tt.checkDefault)
				if !found {
					t.Fatalf("mapping for %q not found in registry", tt.checkDefault)
				}

				// The default mapping should be the same as the expected engine's mapping
				if defaultMapping != engineMapping {
					t.Errorf("default mapping does not match %q engine mapping", tt.checkDefault)
				}
			}
		})
	}
}
