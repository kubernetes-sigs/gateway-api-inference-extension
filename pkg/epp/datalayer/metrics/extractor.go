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
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"

	dto "github.com/prometheus/client_model/go"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// LoRA metrics based on MSP
	LoraInfoRunningAdaptersMetricName = "running_lora_adapters"
	LoraInfoWaitingAdaptersMetricName = "waiting_lora_adapters"
	LoraInfoMaxAdaptersMetricName     = "max_lora"

	CacheConfigBlockSizeInfoMetricName = "block_size"
	CacheConfigNumGPUBlocksMetricName  = "num_gpu_blocks"
)

// Extractor implements the metrics extraction based on the model
// server protocol standard.
type Extractor struct {
	typedName plugins.TypedName
	mapping   *Mapping
}

// Produces returns the data attributes that are provided by the datalayer.metrics
// package.
func Produces() map[string]any {
	return map[string]any{
		metrics.WaitingQueueSizeKey:    int(0),
		metrics.RunningRequestsSizeKey: int(0),
		metrics.KVCacheUsagePercentKey: float64(0),
		metrics.ActiveModelsKey:        map[string]int{},
		metrics.WaitingModelsKey:       map[string]int{},
		metrics.MaxActiveModelsKey:     int(0),
		metrics.UpdateTimeKey:          time.Time{},
	}
}

// NewModelServerExtractor returns a new model server protocol (MSP) metrics extractor,
// configured with the given metrics' specifications.
// These are mandatory metrics per the MSP specification, and are used
// as the basis for the built-in scheduling plugins.
func NewModelServerExtractor(queueSpec, runningSpec, kvusageSpec, loraSpec, cacheInfoSpec string) (*Extractor, error) {
	mapping, err := NewMapping(queueSpec, runningSpec, kvusageSpec, loraSpec, cacheInfoSpec)
	if err != nil {
		return nil, fmt.Errorf("failed to create extractor metrics Mapping - %w", err)
	}
	return &Extractor{
		typedName: plugins.TypedName{
			Type: MetricsExtractorType,
			Name: MetricsExtractorType,
		},
		mapping: mapping,
	}, nil
}

// TypedName returns the type and name of the metrics.Extractor.
func (ext *Extractor) TypedName() plugins.TypedName {
	return ext.typedName
}

// ExpectedType defines the type expected by the metrics.Extractor - a
// parsed output from a Prometheus metrics endpoint.
func (ext *Extractor) ExpectedInputType() reflect.Type {
	return PrometheusMetricType
}

// Extract transforms the data source output into a concrete attribute that
// is stored on the given endpoint.
func (ext *Extractor) Extract(ctx context.Context, data any, ep datalayer.Endpoint) error {
	families, ok := data.(PrometheusMetricMap)
	if !ok {
		return fmt.Errorf("unexpected input in Extract: %T", data)
	}

	var errs []error
	current := ep.GetMetrics()
	clone := current.Clone()
	updated := false

	if spec := ext.mapping.TotalQueuedRequests; spec != nil { // extract queued requests
		if metric, err := spec.getLatestMetric(families); err != nil {
			errs = append(errs, err)
		} else {
			clone.WaitingQueueSize = int(extractValue(metric))
			updated = true
		}
	}

	if spec := ext.mapping.TotalRunningRequests; spec != nil { // extract running requests
		if metric, err := spec.getLatestMetric(families); err != nil {
			errs = append(errs, err)
		} else {
			clone.RunningRequestsSize = int(extractValue(metric))
			updated = true
		}
	}

	if spec := ext.mapping.KVCacheUtilization; spec != nil { // extract KV cache usage
		if metric, err := spec.getLatestMetric(families); err != nil {
			errs = append(errs, err)
		} else {
			clone.KVCacheUsagePercent = extractValue(metric)
			updated = true
		}
	}

	if spec := ext.mapping.LoraRequestInfo; spec != nil { // extract LoRA-specific metrics
		metric, err := spec.getLatestMetric(families)
		if err != nil {
			errs = append(errs, err)
		} else if metric != nil {
			populateLoRAMetrics(clone, metric, &errs)
			updated = true
		}
	}

	if spec := ext.mapping.CacheInfo; spec != nil { // extract CacheInfo-specific metrics
		metric, err := spec.getLatestMetric(families)
		if err != nil {
			errs = append(errs, err)
		} else if metric != nil {
			populateCacheInfoMetrics(clone, metric, &errs)
			updated = true
		}
	}

	logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().NamespacedName)
	if updated {
		clone.UpdateTime = time.Now()
		logger.V(logutil.TRACE).Info("Refreshed metrics", "updated", clone)
		ep.UpdateMetrics(clone)
	}

	if len(errs) != 0 {
		return errors.Join(errs...)
	}
	return nil
}

// populateLoRAMetrics updates the metrics with LoRA adapter info from the metric labels.
func populateLoRAMetrics(clone *datalayer.Metrics, metric *dto.Metric, errs *[]error) {
	clone.ActiveModels = map[string]int{}
	clone.WaitingModels = map[string]int{}

	for _, label := range metric.GetLabel() {
		switch label.GetName() {
		case LoraInfoRunningAdaptersMetricName:
			addAdapters(clone.ActiveModels, label.GetValue())
		case LoraInfoWaitingAdaptersMetricName:
			addAdapters(clone.WaitingModels, label.GetValue())
		case LoraInfoMaxAdaptersMetricName:
			if label.GetValue() != "" {
				if val, err := strconv.Atoi(label.GetValue()); err == nil {
					clone.MaxActiveModels = val
				} else {
					*errs = append(*errs, err)
				}
			}
		}
	}
}

// populateCacheInfoMetrics updates the metrics with cache info from the metric labels.
func populateCacheInfoMetrics(clone *datalayer.Metrics, metric *dto.Metric, errs *[]error) {
	clone.CacheBlockSize = 0
	for _, label := range metric.GetLabel() {
		switch label.GetName() {
		case CacheConfigBlockSizeInfoMetricName:
			if label.GetValue() != "" {
				if val, err := strconv.Atoi(label.GetValue()); err == nil {
					clone.CacheBlockSize = val
				} else {
					*errs = append(*errs, err)
				}
			}
		case CacheConfigNumGPUBlocksMetricName:
			if label.GetValue() != "" {
				if val, err := strconv.Atoi(label.GetValue()); err == nil {
					clone.CacheNumGPUBlocks = val
				} else {
					*errs = append(*errs, err)
				}
			}
		}
	}
}

// addAdapters splits a comma-separated adapter list and stores keys with default value 0.
func addAdapters(m map[string]int, csv string) {
	for _, name := range strings.Split(csv, ",") {
		if trimmed := strings.TrimSpace(name); trimmed != "" {
			m[trimmed] = 0
		}
	}
}
