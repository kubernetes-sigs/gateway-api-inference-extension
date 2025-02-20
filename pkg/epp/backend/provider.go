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

package backend

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"go.uber.org/multierr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	fetchMetricsTimeout = 5 * time.Second
)

func NewProvider(pmc PodMetricsClient, datastore datastore.Datastore) *Provider {
	p := &Provider{
		pmc:       pmc,
		datastore: datastore,
	}
	return p
}

// Provider provides backend pods and information such as metrics.
type Provider struct {
	pmc       PodMetricsClient
	datastore datastore.Datastore
}

type PodMetricsClient interface {
	FetchMetrics(ctx context.Context, existing *datastore.PodMetrics) (*datastore.PodMetrics, error)
}

func (p *Provider) Init(ctx context.Context, refreshMetricsInterval, refreshPrometheusMetricsInterval time.Duration) error {
	// periodically refresh metrics
	logger := log.FromContext(ctx)
	go func() {
		for {
			select {
			case <-ctx.Done():
				logger.V(logutil.DEFAULT).Info("Shutting down metrics prober")
				return
			default:
				time.Sleep(refreshMetricsInterval)
				if err := p.refreshMetricsOnce(logger); err != nil {
					logger.V(logutil.DEFAULT).Error(err, "Failed to refresh metrics")
				}
			}
		}
	}()

	// Periodically flush prometheus metrics for inference pool
	go func() {
		for {
			select {
			case <-ctx.Done():
				logger.V(logutil.DEFAULT).Info("Shutting down prometheus metrics thread")
				return
			default:
				time.Sleep(refreshPrometheusMetricsInterval)
				p.flushPrometheusMetricsOnce(logger)
			}
		}
	}()

	// Periodically print out the pods and metrics for DEBUGGING.
	if logger := logger.V(logutil.DEBUG); logger.Enabled() {
		go func() {
			for {
				select {
				case <-ctx.Done():
					logger.V(logutil.DEFAULT).Info("Shutting down metrics logger thread")
					return
				default:
					time.Sleep(5 * time.Second)
					logger.Info("Current Pods and metrics gathered", "metrics", p.datastore.PodGetAll())
				}
			}
		}()
	}

	return nil
}

func (p *Provider) refreshMetricsOnce(logger logr.Logger) error {
	loggerTrace := logger.V(logutil.TRACE)
	ctx, cancel := context.WithTimeout(context.Background(), fetchMetricsTimeout)
	defer cancel()
	start := time.Now()
	defer func() {
		d := time.Since(start)
		// TODO: add a metric instead of logging
		loggerTrace.Info("Metrics refreshed", "duration", d)
	}()
	var wg sync.WaitGroup
	errCh := make(chan error)
	processOnePod := func(key, value any) bool {
		loggerTrace.Info("Pod and metric being processed", "pod", key, "metric", value)
		existing := value.(*datastore.PodMetrics)
		wg.Add(1)
		go func() {
			defer wg.Done()
			updated, err := p.pmc.FetchMetrics(ctx, existing)
			if err != nil {
				errCh <- fmt.Errorf("failed to parse metrics from %s: %v", existing.NamespacedName, err)
				return
			}
			p.datastore.PodUpdateMetricsIfExist(updated.NamespacedName, &updated.Metrics)
			loggerTrace.Info("Updated metrics for pod", "pod", updated.NamespacedName, "metrics", updated.Metrics)
		}()
		return true
	}
	p.datastore.PodRange(processOnePod)

	// Wait for metric collection for all pods to complete and close the error channel in a
	// goroutine so this is unblocking, allowing the code to proceed to the error collection code
	// below.
	// Note we couldn't use a buffered error channel with a size because the size of the podMetrics
	// sync.Map is unknown beforehand.
	go func() {
		wg.Wait()
		close(errCh)
	}()

	var errs error
	for err := range errCh {
		errs = multierr.Append(errs, err)
	}
	return errs
}

func (p *Provider) flushPrometheusMetricsOnce(logger logr.Logger) {
	logger.V(logutil.DEBUG).Info("Flushing Prometheus Metrics")

	pool, _ := p.datastore.PoolGet()
	if pool == nil {
		// No inference pool or not initialize.
		return
	}

	var kvCacheTotal float64
	var queueTotal int

	podMetrics := p.datastore.PodGetAll()
	if len(podMetrics) == 0 {
		return
	}

	for _, pod := range podMetrics {
		kvCacheTotal += pod.KVCacheUsagePercent
		queueTotal += pod.WaitingQueueSize
	}

	podTotalCount := len(podMetrics)
	metrics.RecordInferencePoolAvgKVCache(pool.Name, kvCacheTotal/float64(podTotalCount))
	metrics.RecordInferencePoolAvgQueueSize(pool.Name, float64(queueTotal/podTotalCount))
}
