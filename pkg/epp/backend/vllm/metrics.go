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

// Package vllm provides vllm specific pod metrics implementation.
package vllm

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"go.uber.org/multierr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// Metric names used in the vLLM metrics implementation.
// Refer to the protocol doc for more details:
// https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/003-model-server-protocol
const (
	LoraRequestInfoMetricName                = "vllm:lora_requests_info"
	LoraRequestInfoRunningAdaptersMetricName = "running_lora_adapters"
	LoraRequestInfoWaitingAdaptersMetricName = "waiting_lora_adapters"
	LoraRequestInfoMaxAdaptersMetricName     = "max_lora"
	// TODO: Replace these with the num_tokens_running/waiting below once we add those to the fork.
	RunningQueueSizeMetricName = "vllm:num_requests_running"
	WaitingQueueSizeMetricName = "vllm:num_requests_waiting"
	/* TODO: Uncomment this once the following are added to the fork.
	RunningQueueSizeMetricName        = "vllm:num_tokens_running"
	WaitingQueueSizeMetricName        = "vllm:num_tokens_waiting"
	*/
	KVCacheUsagePercentMetricName = "vllm:gpu_cache_usage_perc"
)

type PodMetricsClientImpl struct{}

// FetchMetrics fetches metrics from a given pod.
func (p *PodMetricsClientImpl) FetchMetrics(
	ctx context.Context,
	pod *metrics.Pod,
	existing *metrics.Metrics,
	port int32,
) (*metrics.Metrics, error) {
	logger := log.FromContext(ctx)
	loggerDefault := logger.V(logutil.DEFAULT)

	// Currently the metrics endpoint is hard-coded, which works with vLLM.
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/16): Consume this from InferencePool config.
	url := "http://" + pod.Address + ":" + strconv.Itoa(int(port)) + "/metrics"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		loggerDefault.Error(err, "Failed create HTTP request", "method", http.MethodGet, "url", url)
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		loggerDefault.Error(err, "Failed to fetch metrics", "pod", pod.NamespacedName)
		return nil, fmt.Errorf("failed to fetch metrics from %s: %w", pod.NamespacedName, err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		loggerDefault.Error(nil, "Unexpected status code returned", "pod", pod.NamespacedName, "statusCode", resp.StatusCode)
		return nil, fmt.Errorf("unexpected status code from %s: %v", pod.NamespacedName, resp.StatusCode)
	}

	parser := expfmt.TextParser{}
	metricFamilies, err := parser.TextToMetricFamilies(resp.Body)
	if err != nil {
		return nil, err
	}
	return promToPodMetrics(logger, metricFamilies, existing)
}

// promToPodMetrics updates internal pod metrics with scraped prometheus metrics.
// A combined error is returned if errors occur in one or more metric processing.
// it returns a new PodMetrics pointer which can be used to atomically update the pod metrics map.
func promToPodMetrics(
	logger logr.Logger,
	metricFamilies map[string]*dto.MetricFamily,
	existing *metrics.Metrics,
) (*metrics.Metrics, error) {
	var errs error
	updated := existing.Clone()
	runningQueueSize, err := getLatestMetric(logger, metricFamilies, RunningQueueSizeMetricName)
	errs = multierr.Append(errs, err)
	if err == nil {
		updated.RunningQueueSize = int(runningQueueSize.GetGauge().GetValue())
	}
	waitingQueueSize, err := getLatestMetric(logger, metricFamilies, WaitingQueueSizeMetricName)
	errs = multierr.Append(errs, err)
	if err == nil {
		updated.WaitingQueueSize = int(waitingQueueSize.GetGauge().GetValue())
	}
	cachePercent, err := getLatestMetric(logger, metricFamilies, KVCacheUsagePercentMetricName)
	errs = multierr.Append(errs, err)
	if err == nil {
		updated.KVCacheUsagePercent = cachePercent.GetGauge().GetValue()
	}

	loraMetrics, _, err := getLatestLoraMetric(logger, metricFamilies)
	errs = multierr.Append(errs, err)
	/* TODO: uncomment once this is available in vllm.
	kvCap, _, err := getGaugeLatestValue(metricFamilies, KvCacheMaxTokenCapacityMetricName)
	errs = multierr.Append(errs, err)
	if err != nil {
		updated.KvCacheMaxTokenCapacity = int(kvCap)
	}
	*/

	if loraMetrics != nil {
		updated.ActiveModels = make(map[string]int)
		for _, label := range loraMetrics.GetLabel() {
			if label.GetName() == LoraRequestInfoRunningAdaptersMetricName {
				if label.GetValue() != "" {
					adapterList := strings.Split(label.GetValue(), ",")
					for _, adapter := range adapterList {
						updated.ActiveModels[adapter] = 0
					}
				}
			}
			if label.GetName() == LoraRequestInfoWaitingAdaptersMetricName {
				if label.GetValue() != "" {
					adapterList := strings.Split(label.GetValue(), ",")
					for _, adapter := range adapterList {
						updated.ActiveModels[adapter] = 0
					}
				}
			}
			if label.GetName() == LoraRequestInfoMaxAdaptersMetricName {
				if label.GetValue() != "" {
					updated.MaxActiveModels, err = strconv.Atoi(label.GetValue())
					if err != nil {
						errs = multierr.Append(errs, err)
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
func getLatestLoraMetric(logger logr.Logger, metricFamilies map[string]*dto.MetricFamily) (*dto.Metric, time.Time, error) {
	loraRequests, ok := metricFamilies[LoraRequestInfoMetricName]
	if !ok {
		logger.V(logutil.DEFAULT).Error(nil, "Metric family not found", "name", LoraRequestInfoMetricName)
		return nil, time.Time{}, fmt.Errorf("metric family %q not found", LoraRequestInfoMetricName)
	}

	var latest *dto.Metric
	var latestTs float64

	// Iterate over all metrics in the family.
	for _, m := range loraRequests.GetMetric() {
		var running, waiting string
		// Read the label values for running and waiting adapters.
		for _, lp := range m.GetLabel() {
			switch lp.GetName() {
			case LoraRequestInfoRunningAdaptersMetricName:
				running = lp.GetValue()
			case LoraRequestInfoWaitingAdaptersMetricName:
				waiting = lp.GetValue()
			}
		}

		// Ignore metrics with both labels empty. This happens when there are no running or waiting requests on
		// the server, in this case it is best to use the last set of active adapters.
		if running == "" && waiting == "" {
			continue
		}

		// Select the metric with the latest creation timestamp.
		if m.GetGauge().GetValue() > latestTs {
			latestTs = m.GetGauge().GetValue()
			latest = m
		}
	}

	if latest == nil {
		logger.V(logutil.TRACE).Info("Metric value Empty", "value", latest, "metric", LoraRequestInfoMetricName)
		return nil, time.Time{}, nil
	}

	// Convert the gauge value (creation timestamp) to time.Time.
	return latest, time.Unix(0, int64(latestTs*1000)), nil
}

// getLatestMetric gets the latest metric of a family. This should be used to get the latest Gauge metric.
// Since vllm doesn't set the timestamp in metric, this metric essentially gets the first metric.
func getLatestMetric(logger logr.Logger, metricFamilies map[string]*dto.MetricFamily, metricName string) (*dto.Metric, error) {
	mf, ok := metricFamilies[metricName]
	if !ok {
		logger.V(logutil.DEFAULT).Error(nil, "Metric family not found", "name", metricName)
		return nil, fmt.Errorf("metric family %q not found", metricName)
	}
	if len(mf.GetMetric()) == 0 {
		return nil, fmt.Errorf("no metrics available for %q", metricName)
	}
	var latestTs int64
	var latest *dto.Metric
	for _, m := range mf.GetMetric() {
		if m.GetTimestampMs() >= latestTs {
			latestTs = m.GetTimestampMs()
			latest = m
		}
	}
	logger.V(logutil.TRACE).Info("Metric value selected", "value", latest, "metric", metricName)
	return latest, nil
}
