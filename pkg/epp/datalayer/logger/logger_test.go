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

package logger

import (
	"bytes"
	"context"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap/zapcore"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	poolutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/pool"
)

// Buffer to write the logs to
type buffer struct {
	buf bytes.Buffer
	mu  sync.Mutex
}

func (s *buffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.Write(p)
}

func (s *buffer) read() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.String()
}

func TestLogger(t *testing.T) {
	// Redirect the logger to a buffer
	var b buffer
	opts := &zap.Options{
		DestWriter:  &b,
		Development: true,
		Level:       zapcore.Level(-5),
	}
	logger := zap.New(zap.UseFlagOptions(opts))
	ctrl.SetLogger(logger)
	ctx, cancel := context.WithCancel(context.Background())
	ctx = logr.NewContext(ctx, logger)

	StartMetricsLogger(ctx, &FakeLoggerDataStore{}, 100*time.Millisecond, 100*time.Millisecond)

	time.Sleep(6 * time.Second)
	cancel()

	logOutput := b.read()
	assert.Contains(t, logOutput, "Refreshing Prometheus Metrics	{\"ReadyPods\": 2}")
	assert.Contains(t, logOutput, "Current Pods and metrics gathered	{\"Fresh metrics\": \"[Metadata: {NamespacedName:default/pod1 PodName: Address:1.2.3.4:5678")
	assert.Contains(t, logOutput, "Metrics: {ActiveModels:map[modelA:1] WaitingModels:map[modelB:2] MaxActiveModels:5")
	assert.Contains(t, logOutput, "RunningRequestsSize:3 WaitingQueueSize:7 KVCacheUsagePercent:42.5 KvCacheMaxTokenCapacity:2048")
	assert.Contains(t, logOutput, "Metadata: {NamespacedName:default/pod2 PodName: Address:1.2.3.4:5679")
	assert.Contains(t, logOutput, "\"Stale metrics\": \"[]\"")
}

func TestCalculateTotals(t *testing.T) {
	tests := []struct {
		name      string
		endpoints []fwkdl.Endpoint
		want      totals
	}{
		{
			name:      "empty list",
			endpoints: []fwkdl.Endpoint{},
			want:      totals{},
		},
		{
			name: "single endpoint",
			endpoints: []fwkdl.Endpoint{
				fwkdl.NewEndpoint(pod1, &fwkdl.Metrics{
					KVCacheUsagePercent: 50.0,
					WaitingQueueSize:    3,
					RunningRequestsSize: 5,
					UpdateTime:          time.Now(),
				}),
			},
			want: totals{kvCache: 50.0, queueSize: 3, runningRequests: 5},
		},
		{
			name: "multiple endpoints aggregated",
			endpoints: []fwkdl.Endpoint{
				fwkdl.NewEndpoint(pod1, &fwkdl.Metrics{
					KVCacheUsagePercent: 30.0,
					WaitingQueueSize:    2,
					RunningRequestsSize: 1,
					UpdateTime:          time.Now(),
				}),
				fwkdl.NewEndpoint(pod2, &fwkdl.Metrics{
					KVCacheUsagePercent: 70.0,
					WaitingQueueSize:    5,
					RunningRequestsSize: 3,
					UpdateTime:          time.Now(),
				}),
			},
			want: totals{kvCache: 100.0, queueSize: 7, runningRequests: 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateTotals(tt.endpoints)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestRefreshPrometheusMetricsAvgValues(t *testing.T) {
	metrics.Register()
	metrics.Reset()

	logger := logr.Discard()

	// Use a datastore where pods have odd-sum metrics so that
	// integer division would truncate incorrectly.
	// Pod1: RunningRequests=0, Queue=1
	// Pod2: RunningRequests=1, Queue=2
	// Correct avg: RunningRequests=0.5, Queue=1.5
	// Integer truncation would give: RunningRequests=0, Queue=1
	ds := &FakeOddMetricsDataStore{}

	refreshPrometheusMetrics(logger, ds, 100*time.Millisecond)

	families, err := ctrlmetrics.Registry.Gather()
	assert.NoError(t, err)

	findGauge := func(name string) float64 {
		for _, f := range families {
			if f.GetName() == name {
				for _, m := range f.GetMetric() {
					return m.GetGauge().GetValue()
				}
			}
		}
		t.Fatalf("metric %s not found", name)
		return 0
	}

	avgRunning := findGauge("inference_pool_average_running_requests")
	avgQueue := findGauge("inference_pool_average_queue_size")

	assert.InDelta(t, 0.5, avgRunning, 0.001, "average running requests should be 0.5, not truncated to 0")
	assert.InDelta(t, 1.5, avgQueue, 0.001, "average queue size should be 1.5, not truncated to 1")
}

type FakeOddMetricsDataStore struct{}

func (f *FakeOddMetricsDataStore) PoolGet() (*datalayer.EndpointPool, error) {
	pool := &v1.InferencePool{Spec: v1.InferencePoolSpec{TargetPorts: []v1.Port{{Number: 8000}}}}
	return poolutil.InferencePoolToEndpointPool(pool), nil
}

func (f *FakeOddMetricsDataStore) PodList(predicate func(fwkdl.Endpoint) bool) []fwkdl.Endpoint {
	m1 := &fwkdl.Metrics{
		RunningRequestsSize: 0,
		WaitingQueueSize:    1,
		KVCacheUsagePercent: 10.0,
		UpdateTime:          time.Now(),
	}
	m2 := &fwkdl.Metrics{
		RunningRequestsSize: 1,
		WaitingQueueSize:    2,
		KVCacheUsagePercent: 20.0,
		UpdateTime:          time.Now(),
	}
	ep1 := fwkdl.NewEndpoint(pod1, m1)
	ep2 := fwkdl.NewEndpoint(pod2, m2)
	pods := []fwkdl.Endpoint{ep1, ep2}
	res := []fwkdl.Endpoint{}
	for _, pod := range pods {
		if predicate(pod) {
			res = append(res, pod)
		}
	}
	return res
}

var pod1 = &fwkdl.EndpointMetadata{
	NamespacedName: types.NamespacedName{
		Name:      "pod1",
		Namespace: "default",
	},
	Address: "1.2.3.4:5678",
}
var pod2 = &fwkdl.EndpointMetadata{
	NamespacedName: types.NamespacedName{
		Name:      "pod2",
		Namespace: "default",
	},
	Address: "1.2.3.4:5679",
}

type FakeLoggerDataStore struct{}

func (f *FakeLoggerDataStore) PoolGet() (*datalayer.EndpointPool, error) {
	pool := &v1.InferencePool{Spec: v1.InferencePoolSpec{TargetPorts: []v1.Port{{Number: 8000}}}}
	return poolutil.InferencePoolToEndpointPool(pool), nil
}

func (f *FakeLoggerDataStore) PodList(predicate func(fwkdl.Endpoint) bool) []fwkdl.Endpoint {
	var m = &fwkdl.Metrics{
		ActiveModels:            map[string]int{"modelA": 1},
		WaitingModels:           map[string]int{"modelB": 2},
		MaxActiveModels:         5,
		RunningRequestsSize:     3,
		WaitingQueueSize:        7,
		KVCacheUsagePercent:     42.5,
		KvCacheMaxTokenCapacity: 2048,
		UpdateTime:              time.Now(),
	}
	ep1 := fwkdl.NewEndpoint(pod1, m)
	ep2 := fwkdl.NewEndpoint(pod2, m)
	pods := []fwkdl.Endpoint{ep1, ep2}
	res := []fwkdl.Endpoint{}

	for _, pod := range pods {
		if predicate(pod) {
			res = append(res, pod)
		}
	}
	return res
}
