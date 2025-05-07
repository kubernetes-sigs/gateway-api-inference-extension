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
	"net/http"
	"strconv"
	"strings"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"go.uber.org/multierr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	podinfo "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/pod-info"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// LoRA metrics based on protocol
	LoraInfoRunningAdaptersMetricName = "running_lora_adapters"
	LoraInfoWaitingAdaptersMetricName = "waiting_lora_adapters"
	LoraInfoMaxAdaptersMetricName     = "max_lora"
	metricsRequestRelativePath        = "/metrics"
)

var _ podinfo.Scraper = &MetricsScraper{}

// MetricsScraper is a scraper that scrapes periodically pod's metrics using the /metrics endpoint.
type MetricsScraper struct {
	MetricMapping   *MetricMapping
	existingMetrics *MetricsData
}

// Name returns name of the scraper.
func (s *MetricsScraper) Name() string {
	return MetricsDataKey // "metrics"
}

func (s *MetricsScraper) InitData() podinfo.ScrapedData {
	return newMetrics()
}

// Scrape scrapes metrics from a given pod, clones the existing metrics object and returns an updated one.
func (s *MetricsScraper) Scrape(ctx context.Context, pod *backend.Pod, port int) (podinfo.ScrapedData, error) {
	// Currently the metrics endpoint is hard-coded, which works with vLLM.
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/16): Consume this from InferencePool config.
	url := fmt.Sprintf("http://%s:%d/%s", pod.Address, port, metricsRequestRelativePath)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create GET request - %w", err)
	}

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("failed to scrape from pod %s - %w", pod.NamespacedName, err)
	}
	defer func() {
		_ = response.Body.Close()
	}()

	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code from pod %s: %v", pod.NamespacedName, response.StatusCode)
	}

	parser := expfmt.TextParser{}
	metricFamilies, err := parser.TextToMetricFamilies(response.Body)
	if err != nil {
		return nil, err
	}
	return s.promToPodMetrics(metricFamilies)
}

func (s *MetricsScraper) ProcessResult(ctx context.Context, data podinfo.ScrapedData) {
	if data == nil {
		return
	}
	metrics := data.(*MetricsData)
	if metrics == nil {
		return
	}
	// if result cannot be converted to *Metrics, we cannot process result, otherwise
	metrics.UpdateTime = time.Now()
	s.existingMetrics = metrics
	log.FromContext(ctx).V(logutil.TRACE).Info("processed scraped metrics", "updated", metrics)
}

// promToPodMetrics updates internal pod metrics with scraped Prometheus metrics.
func (s *MetricsScraper) promToPodMetrics(metricFamilies map[string]*dto.MetricFamily) (podinfo.ScrapedData, error) {
	var errs error
	updated := podinfo.CloneScrapedData(s.existingMetrics)

	if s.MetricMapping.TotalQueuedRequests != nil {
		queued, err := s.getMetric(metricFamilies, *s.MetricMapping.TotalQueuedRequests)
		if err == nil {
			updated.WaitingQueueSize = int(queued.GetGauge().GetValue())
		} else {
			errs = multierr.Append(errs, err)
		}
	}

	if s.MetricMapping.KVCacheUtilization != nil {
		usage, err := s.getMetric(metricFamilies, *s.MetricMapping.KVCacheUtilization)
		if err == nil {
			updated.KVCacheUsagePercent = usage.GetGauge().GetValue()
		} else {
			errs = multierr.Append(errs, err)
		}
	}

	// Handle LoRA metrics (only if all LoRA MetricSpecs are present)
	if s.MetricMapping.LoraRequestInfo != nil {
		loraMetrics, err := s.getLatestLoraMetric(metricFamilies)
		errs = multierr.Append(errs, err)

		if loraMetrics != nil {
			updated.ActiveModels = make(map[string]int)
			updated.WaitingModels = make(map[string]int)
			for _, label := range loraMetrics.GetLabel() {
				if label.GetName() == LoraInfoRunningAdaptersMetricName {
					if label.GetValue() != "" {
						adapterList := strings.Split(label.GetValue(), ",")
						for _, adapter := range adapterList {
							updated.ActiveModels[adapter] = 0
						}
					}
				}
				if label.GetName() == LoraInfoWaitingAdaptersMetricName {
					if label.GetValue() != "" {
						adapterList := strings.Split(label.GetValue(), ",")
						for _, adapter := range adapterList {
							updated.WaitingModels[adapter] = 0
						}
					}
				}
				if label.GetName() == LoraInfoMaxAdaptersMetricName {
					if label.GetValue() != "" {
						updated.MaxActiveModels, err = strconv.Atoi(label.GetValue())
						if err != nil {
							errs = multierr.Append(errs, err)
						}
					}
				}
			}
		}
	}

	return updated, errs
}

// getLatestLoraMetric gets latest lora metric series in gauge metric family `vllm:lora_requests_info`
// reason its specially fetched is because each label key value pair permutation generates new series
// and only most recent is useful. The value of each series is the creation timestamp so we can
// retrieve the latest by sorting the value.
func (s *MetricsScraper) getLatestLoraMetric(metricFamilies map[string]*dto.MetricFamily) (*dto.Metric, error) {
	if s.MetricMapping.LoraRequestInfo == nil {
		return nil, nil // No LoRA metrics configured
	}

	loraRequests, ok := metricFamilies[s.MetricMapping.LoraRequestInfo.MetricName]
	if !ok {
		return nil, fmt.Errorf("metric family %q not found", s.MetricMapping.LoraRequestInfo.MetricName)
	}

	var latest *dto.Metric
	var latestTs float64 // Use float64, as Gauge.Value is float64

	// Iterate over all metrics in the family.
	for _, m := range loraRequests.GetMetric() {
		running := ""
		waiting := ""
		// Check if the metric has the expected LoRA labels.
		for _, lp := range m.GetLabel() {
			switch lp.GetName() {
			case LoraInfoRunningAdaptersMetricName:
				running = lp.GetValue()
			case LoraInfoWaitingAdaptersMetricName:
				waiting = lp.GetValue()
			}
		}
		// Ignore metrics with both labels empty.
		if running == "" && waiting == "" {
			continue
		}

		// Select the metric with the *largest Gauge Value* (which represents the timestamp).
		if m.GetGauge().GetValue() > latestTs {
			latestTs = m.GetGauge().GetValue()
			latest = m
		}
	}
	if latest == nil {
		return nil, nil
	}

	return latest, nil // Convert nanoseconds to time.Time
}

// getMetric retrieves a specific metric based on MetricSpec.
func (s *MetricsScraper) getMetric(metricFamilies map[string]*dto.MetricFamily, spec MetricSpec) (*dto.Metric, error) {
	mf, ok := metricFamilies[spec.MetricName]
	if !ok {
		return nil, fmt.Errorf("metric family %q not found", spec.MetricName)
	}

	if len(mf.GetMetric()) == 0 {
		return nil, fmt.Errorf("no metrics available for %q", spec.MetricName)
	}

	return getLatestMetric(mf, &spec)
}

// getLabeledMetric gets the latest metric with matching labels.
func getLatestMetric(mf *dto.MetricFamily, spec *MetricSpec) (*dto.Metric, error) {
	var latestMetric *dto.Metric
	var latestTimestamp int64 = -1 // Initialize to -1 so any timestamp is greater

	for _, m := range mf.GetMetric() {
		if spec.Labels == nil || labelsMatch(m.GetLabel(), spec.Labels) {
			if m.GetTimestampMs() > latestTimestamp {
				latestTimestamp = m.GetTimestampMs()
				latestMetric = m
			}
		}
	}

	if latestMetric != nil {
		return latestMetric, nil
	}

	return nil, fmt.Errorf("no matching metric found for %q with labels %+v", spec.MetricName, spec.Labels)
}

// labelsMatch checks if a metric's labels contain all the labels in the spec.
func labelsMatch(metricLabels []*dto.LabelPair, specLabels map[string]string) bool {
	if len(specLabels) == 0 {
		return true // No specific labels required
	}

	for specName, specValue := range specLabels {
		found := false
		for _, label := range metricLabels {
			if label.GetName() == specName && label.GetValue() == specValue {
				found = true
				break
			}
		}
		if !found {
			return false // A required label is missing
		}
	}
	return true // All required labels are present
}
