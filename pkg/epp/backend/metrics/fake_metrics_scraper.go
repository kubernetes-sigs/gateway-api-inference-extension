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
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	podinfo "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/pod-info"
)

var _ podinfo.Scraper = &FakeMetricsScraper{}

type FakeMetricsScraper struct {
	errMu sync.RWMutex
	Err   map[types.NamespacedName]error
	resMu sync.RWMutex
	Res   map[types.NamespacedName]*MetricsData
}

func (s *FakeMetricsScraper) Name() string {
	return MetricsDataKey
}

func (s *FakeMetricsScraper) InitData() podinfo.ScrapedData {
	return newMetrics()
}

func (s *FakeMetricsScraper) Scrape(ctx context.Context, pod *backend.Pod, port int) (podinfo.ScrapedData, error) {
	s.errMu.RLock()
	err, ok := s.Err[pod.NamespacedName]
	s.errMu.RUnlock()
	if ok {
		return nil, err
	}
	s.resMu.RLock()
	res, ok := s.Res[pod.NamespacedName]
	s.resMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("no pod found: %v", pod.NamespacedName)
	}
	return podinfo.CloneScrapedData(res), nil
}

func (s *FakeMetricsScraper) ProcessResult(ctx context.Context, data podinfo.ScrapedData) {} // noop

func (s *FakeMetricsScraper) SetRes(new map[types.NamespacedName]*MetricsData) {
	s.resMu.Lock()
	defer s.resMu.Unlock()
	s.Res = new
}

func (s *FakeMetricsScraper) SetErr(new map[types.NamespacedName]error) {
	s.errMu.Lock()
	defer s.errMu.Unlock()
	s.Err = new
}
