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

package podinfo

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

var _ PodInfo = &podInfo{}

type PodInfo interface {
	GetPod() *backend.Pod
	GetData(key string) (ScrapedData, bool)
	GetDataKeys() []string
	UpdatePod(*corev1.Pod)
	Stop()
	String() string
}

// data encapsulates all data that is scraped from a pod by the different scrapers.
// Each scraper will have an entry in the map, where the key is the scraper.Name()
// and the value is the latest data that was scraped by that scraper.
type data struct {
	data map[string]ScrapedData
}

func newPodInfo(pod *corev1.Pod, scrapers map[Scraper]*ScraperConfig, ds Datastore) *podInfo {
	info := &podInfo{
		scrapers: scrapers,
		ds:       ds,
		dataLock: sync.RWMutex{},
		once:     sync.Once{},
	}
	// initialize scrapers data, each scraper has an entry mapped from its name to its data
	data := &data{data: make(map[string]ScrapedData, 0)}
	for scraper := range scrapers {
		data.data[scraper.Name()] = scraper.InitData()
	}

	info.pod.Store(toInternalPod(pod))
	info.data.Store(data)

	return info
}

type podInfo struct {
	pod      atomic.Pointer[backend.Pod]
	data     atomic.Pointer[data]
	scrapers map[Scraper]*ScraperConfig
	ds       Datastore

	dataLock sync.RWMutex // dataLock is used to synchronize RW access to data map.
	once     sync.Once    // ensure the start function is called only once.
	stopFunc context.CancelFunc
	logger   logr.Logger
}

func (pi *podInfo) GetPod() *backend.Pod {
	return pi.pod.Load()
}

func (pi *podInfo) GetData(key string) (ScrapedData, bool) {
	if data := pi.data.Load(); data != nil && data.data != nil {
		pi.dataLock.RLock()
		defer pi.dataLock.RUnlock()
		value, ok := data.data[key]
		return value, ok
	}
	// if nothing is stored in data field, return nil
	return nil, false
}

func (pi *podInfo) GetDataKeys() []string {
	if data := pi.data.Load(); data != nil && data.data != nil {
		pi.dataLock.RLock()
		defer pi.dataLock.RUnlock()
		result := []string{}
		for key := range data.data {
			result = append(result, key)
		}
		return result
	}
	// if nothing is stored in data field, not data keys
	return []string{}
}

func (pi *podInfo) UpdatePod(pod *corev1.Pod) {
	pi.pod.Store(toInternalPod(pod))
}

func (pi *podInfo) String() string {
	pi.dataLock.RLock()
	defer pi.dataLock.RUnlock()
	return fmt.Sprintf("Pod: %v; Data: %v", pi.GetPod(), pi.data.Load().data)
}

func (pi *podInfo) Stop() {
	pi.logger.V(logutil.DEFAULT).Info("Stopping scrape loop")
	pi.stopFunc()
}

func (pi *podInfo) startScrapers(ctx context.Context) {
	// The returned context's Done channel is closed when the returned cancel function is called or when the parent context's Done channel is closed, whichever happens first.
	newCtx, cancel := context.WithCancel(ctx)
	pi.stopFunc = cancel
	pi.logger = log.FromContext(ctx).WithValues("pod", pi.GetPod())

	pi.once.Do(func() {
		for scraper, config := range pi.scrapers { // TODO should be consolidated into a single go routine for all scrapers
			go pi.startScrapeLoop(newCtx, scraper, config.ScrapeInterval, config.ScrapeTimeout)
		}
	})
}

func (pi *podInfo) startScrapeLoop(ctx context.Context, scraper Scraper, interval time.Duration, timeout time.Duration) {
	pi.logger.V(logutil.DEFAULT).Info("Starting scraper", "ScraperName", scraper.Name())
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C: // scrape periodically
			pool, err := pi.ds.PoolGet()
			if err != nil {
				pi.logger.V(logutil.TRACE).Info("InferencePool not set, skipping scrape", "ScraperName", scraper.Name())
				continue // No inference pool or not initialize, skip this scraping cycle.
			}
			ctxWithTimeout, cancel := context.WithTimeout(ctx, timeout)
			scrapedData, err := scraper.Scrape(ctxWithTimeout, pi.GetPod(), int(pool.Spec.TargetPortNumber))
			cancel() // cancels only ctxWithTimeout, doesn't cancel ctx
			if err != nil {
				pi.logger.V(logutil.TRACE).Error(err, "Failed to scrape", "ScraperName", scraper.Name())
			}
			// allow processing partial results in case there was an error.
			// the scraper can return nil in case of an error and then do nothing in the ProcessResult function in case no partial update is required
			scraper.ProcessResult(ctx, scrapedData)
			// store updated data if its valid
			if scrapedData != nil {
				pi.storeScrapedData(scraper.Name(), scrapedData)
			}
		}
	}
}

func (pi *podInfo) storeScrapedData(key string, data ScrapedData) {
	updated := pi.data.Load()
	pi.dataLock.Lock()
	defer pi.dataLock.Unlock()
	updated.data[key] = data
	pi.data.Store(updated)
}

func toInternalPod(pod *corev1.Pod) *backend.Pod {
	return &backend.Pod{
		NamespacedName: types.NamespacedName{
			Name:      pod.GetName(),
			Namespace: pod.GetNamespace(),
		},
		Address: pod.Status.PodIP,
		Labels:  pod.GetLabels(),
	}
}
