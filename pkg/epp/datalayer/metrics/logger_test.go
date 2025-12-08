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
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
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

	StartMetricsLogger(ctx, &fakeDataStore{}, 100*time.Millisecond, 100*time.Millisecond)

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

var pod1 = &datalayer.EndpointMetadata{
	NamespacedName: types.NamespacedName{
		Name:      "pod1",
		Namespace: "default",
	},
	Address: "1.2.3.4:5678",
}
var pod2 = &datalayer.EndpointMetadata{
	NamespacedName: types.NamespacedName{
		Name:      "pod2",
		Namespace: "default",
	},
	Address: "1.2.3.4:5679",
}

type fakeDataStore struct{}

func (f *fakeDataStore) PoolGet() (*datalayer.EndpointPool, error) {
	pool := &v1.InferencePool{Spec: v1.InferencePoolSpec{TargetPorts: []v1.Port{{Number: 8000}}}}
	return poolutil.InferencePoolToEndpointPool(pool), nil
}

func (f *fakeDataStore) PodList(predicate func(datalayer.Endpoint) bool) []datalayer.Endpoint {
	var m = &datalayer.Metrics{
		ActiveModels:            map[string]int{"modelA": 1},
		WaitingModels:           map[string]int{"modelB": 2},
		MaxActiveModels:         5,
		RunningRequestsSize:     3,
		WaitingQueueSize:        7,
		KVCacheUsagePercent:     42.5,
		KvCacheMaxTokenCapacity: 2048,
		UpdateTime:              time.Now(),
	}
	ep1 := datalayer.NewEndpoint(pod1, m)
	ep2 := datalayer.NewEndpoint(pod2, m)
	pods := []datalayer.Endpoint{ep1, ep2}
	res := []datalayer.Endpoint{}

	for _, pod := range pods {
		if predicate(pod) {
			res = append(res, pod)
		}
	}
	return res
}
