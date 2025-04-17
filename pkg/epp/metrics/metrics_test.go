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
	"os"
	"testing"
	"time"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	RequestTotalMetric                 = InferenceModelComponent + "_request_total"
	RequestErrorTotalMetric            = InferenceModelComponent + "_request_error_total"
	RequestLatenciesMetric             = InferenceModelComponent + "_request_duration_seconds"
	RequestSizesMetric                 = InferenceModelComponent + "_request_sizes"
	ResponseSizesMetric                = InferenceModelComponent + "_response_sizes"
	InputTokensMetric                  = InferenceModelComponent + "_input_tokens"
	OutputTokensMetric                 = InferenceModelComponent + "_output_tokens"
	NormalizedTimePerOutputTokenMetric = InferenceModelComponent + "_normalized_time_per_output_token_seconds"
	RunningRequestsMetric              = InferenceModelComponent + "_running_requests"
	KVCacheAvgUsageMetric              = InferencePoolComponent + "_average_kv_cache_utilization"
	QueueAvgSizeMetric                 = InferencePoolComponent + "_average_queue_size"
)

func TestRecordRequestCounterandSizes(t *testing.T) {
	type requests struct {
		modelName       string
		targetModelName string
		reqSize         int
	}
	scenarios := []struct {
		name string
		reqs []requests
	}{{
		name: "multiple requests",
		reqs: []requests{
			{
				modelName:       "m10",
				targetModelName: "t10",
				reqSize:         1200,
			},
			{
				modelName:       "m10",
				targetModelName: "t10",
				reqSize:         500,
			},
			{
				modelName:       "m10",
				targetModelName: "t11",
				reqSize:         2480,
			},
			{
				modelName:       "m20",
				targetModelName: "t20",
				reqSize:         80,
			},
		},
	}}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				RecordRequestCounter(req.modelName, req.targetModelName)
				RecordRequestSizes(req.modelName, req.targetModelName, req.reqSize)
			}
			wantRequestTotal, err := os.Open("testdata/request_total_metric")
			defer func() {
				if err := wantRequestTotal.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantRequestTotal, RequestTotalMetric); err != nil {
				t.Error(err)
			}
			wantRequestSizes, err := os.Open("testdata/request_sizes_metric")
			defer func() {
				if err := wantRequestSizes.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantRequestSizes, RequestSizesMetric); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestRecordRequestErrorCounter(t *testing.T) {
	type requests struct {
		modelName       string
		targetModelName string
		error           string
	}
	scenarios := []struct {
		name    string
		reqs    []requests
		invalid bool
	}{
		{
			name: "multiple requests",
			reqs: []requests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					error:           errutil.Internal,
				},
				{
					modelName:       "m10",
					targetModelName: "t10",
					error:           errutil.Internal,
				},
				{
					modelName:       "m10",
					targetModelName: "t11",
					error:           errutil.ModelServerError,
				},
				{
					modelName:       "m20",
					targetModelName: "t20",
					error:           errutil.InferencePoolResourceExhausted,
				},
			},
		},
	}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				RecordRequestErrCounter(req.modelName, req.targetModelName, req.error)
			}

			wantRequestErrorCounter, err := os.Open("testdata/request_error_total_metric")
			defer func() {
				if err := wantRequestErrorCounter.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantRequestErrorCounter, RequestErrorTotalMetric); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestRecordRequestLatencies(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	timeBaseline := time.Now()
	type requests struct {
		modelName       string
		targetModelName string
		receivedTime    time.Time
		completeTime    time.Time
	}
	scenarios := []struct {
		name    string
		reqs    []requests
		invalid bool
	}{
		{
			name: "multiple requests",
			reqs: []requests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 10),
				},
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 1600),
				},
				{
					modelName:       "m10",
					targetModelName: "t11",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 60),
				},
				{
					modelName:       "m20",
					targetModelName: "t20",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 120),
				},
			},
		},
		{
			name: "invalid elapsed time",
			reqs: []requests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline.Add(time.Millisecond * 10),
					completeTime:    timeBaseline,
				},
			},
			invalid: true,
		},
	}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				success := RecordRequestLatencies(ctx, req.modelName, req.targetModelName, req.receivedTime, req.completeTime)
				if success == scenario.invalid {
					t.Errorf("got record success(%v), but the request expects invalid(%v)", success, scenario.invalid)
				}
			}

			wantRequestLatencies, err := os.Open("testdata/request_duration_seconds_metric")
			defer func() {
				if err := wantRequestLatencies.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantRequestLatencies, RequestLatenciesMetric); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestRecordNormalizedTimePerOutputToken(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	timeBaseline := time.Now()
	type tokenRequests struct {
		modelName       string
		targetModelName string
		receivedTime    time.Time
		completeTime    time.Time
		outputTokens    int
	}
	scenarios := []struct {
		name    string
		reqs    []tokenRequests
		invalid bool
	}{
		{
			name: "multiple requests",
			reqs: []tokenRequests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 1000),
					outputTokens:    100, // 10ms per token
				},
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 1600),
					outputTokens:    80, // 20ms per token
				},
				{
					modelName:       "m10",
					targetModelName: "t11",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 6000),
					outputTokens:    300, // 20ms per token
				},
				{
					modelName:       "m20",
					targetModelName: "t20",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 2400),
					outputTokens:    400, // 6ms per token
				},
			},
		},
		{
			name: "invalid elapsed time",
			reqs: []tokenRequests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline.Add(time.Millisecond * 10),
					completeTime:    timeBaseline,
					outputTokens:    100,
				},
			},
			invalid: true,
		},
		{
			name: "invalid token count",
			reqs: []tokenRequests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 1000),
					outputTokens:    0, // Invalid: zero tokens
				},
			},
			invalid: true,
		},
	}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				success := RecordNormalizedTimePerOutputToken(ctx, req.modelName, req.targetModelName, req.receivedTime, req.completeTime, req.outputTokens)
				if success == scenario.invalid {
					t.Errorf("got record success(%v), but the request expects invalid(%v)", success, scenario.invalid)
				}
			}

			wantLatencyPerToken, err := os.Open("testdata/normalized_time_per_output_token_seconds_metric")
			defer func() {
				if err := wantLatencyPerToken.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantLatencyPerToken, NormalizedTimePerOutputTokenMetric); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestRecordResponseMetrics(t *testing.T) {
	type responses struct {
		modelName       string
		targetModelName string
		inputToken      int
		outputToken     int
		respSize        int
	}
	scenarios := []struct {
		name string
		resp []responses
	}{{
		name: "multiple requests",
		resp: []responses{
			{
				modelName:       "m10",
				targetModelName: "t10",
				respSize:        1200,
				inputToken:      10,
				outputToken:     100,
			},
			{
				modelName:       "m10",
				targetModelName: "t10",
				respSize:        500,
				inputToken:      20,
				outputToken:     200,
			},
			{
				modelName:       "m10",
				targetModelName: "t11",
				respSize:        2480,
				inputToken:      30,
				outputToken:     300,
			},
			{
				modelName:       "m20",
				targetModelName: "t20",
				respSize:        80,
				inputToken:      40,
				outputToken:     400,
			},
		},
	}}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, resp := range scenario.resp {
				RecordInputTokens(resp.modelName, resp.targetModelName, resp.inputToken)
				RecordOutputTokens(resp.modelName, resp.targetModelName, resp.outputToken)
				RecordResponseSizes(resp.modelName, resp.targetModelName, resp.respSize)
			}
			wantResponseSize, err := os.Open("testdata/response_sizes_metric")
			defer func() {
				if err := wantResponseSize.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantResponseSize, ResponseSizesMetric); err != nil {
				t.Error(err)
			}

			wantInputToken, err := os.Open("testdata/input_tokens_metric")
			defer func() {
				if err := wantInputToken.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantInputToken, InputTokensMetric); err != nil {
				t.Error(err)
			}

			wantOutputToken, err := os.Open("testdata/output_tokens_metric")
			defer func() {
				if err := wantOutputToken.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantOutputToken, OutputTokensMetric); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestRunningRequestsMetrics(t *testing.T) {
	type request struct {
		modelName string
		complete  bool // true -> request is completed, false -> running request
	}

	scenarios := []struct {
		name     string
		requests []request
	}{
		{
			name: "basic test",
			requests: []request{
				{
					modelName: "m1",
					complete:  false,
				},
				{
					modelName: "m1",
					complete:  false,
				},
				{
					modelName: "m1",
					complete:  true,
				},
				{
					modelName: "m2",
					complete:  false,
				},
			},
		},
	}

	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.requests {
				if req.complete {
					DecRunningRequests(req.modelName)
				} else {
					IncRunningRequests(req.modelName)
				}
			}

			wantRunningRequests, err := os.Open("testdata/running_requests_metrics")
			defer func() {
				if err := wantRunningRequests.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantRunningRequests, RunningRequestsMetric); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestInferencePoolMetrics(t *testing.T) {
	scenarios := []struct {
		name         string
		poolName     string
		kvCacheAvg   float64
		queueSizeAvg float64
	}{
		{
			name:         "basic test",
			poolName:     "p1",
			kvCacheAvg:   0.3,
			queueSizeAvg: 0.4,
		},
	}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			RecordInferencePoolAvgKVCache(scenario.poolName, scenario.kvCacheAvg)
			RecordInferencePoolAvgQueueSize(scenario.poolName, scenario.queueSizeAvg)

			wantKVCache, err := os.Open("testdata/kv_cache_avg_metrics")
			defer func() {
				if err := wantKVCache.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantKVCache, KVCacheAvgUsageMetric); err != nil {
				t.Error(err)
			}

			wantQueueSize, err := os.Open("testdata/queue_avg_size_metrics")
			defer func() {
				if err := wantQueueSize.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantQueueSize, QueueAvgSizeMetric); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestSchedulerPluginProcessingLatencies(t *testing.T) {
	type pluginLatency struct {
		pluginType string
		pluginName string
		duration   time.Duration
	}
	scenarios := []struct {
		name      string
		latencies []pluginLatency
	}{
		{
			name: "multiple plugins",
			latencies: []pluginLatency{
				{
					pluginType: "PreSchedule",
					pluginName: "PluginA",
					duration:   100 * time.Millisecond,
				},
				{
					pluginType: "PostSchedule",
					pluginName: "PluginB",
					duration:   200 * time.Millisecond,
				},
				{
					pluginType: "Filter",
					pluginName: "PluginC",
					duration:   50 * time.Millisecond,
				},
				{
					pluginType: "Scorer",
					pluginName: "PluginD",
					duration:   10 * time.Millisecond,
				},
				{
					pluginType: "Picker",
					pluginName: "PluginE",
					duration:   10 * time.Microsecond,
				},
			},
		},
	}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, latency := range scenario.latencies {
				RecordSchedulerPluginProcessingLatency(latency.pluginType, latency.pluginName, latency.duration)
			}

			wantPluginLatencies, err := os.Open("testdata/scheduler_plugin_processing_latencies_metric")
			defer func() {
				if err := wantPluginLatencies.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantPluginLatencies, "endpoint_picker_scheduler_plugin_processing_latencies"); err != nil {
				t.Error(err)
			}
		})
	}
}
