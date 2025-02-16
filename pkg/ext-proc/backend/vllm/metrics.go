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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

const (
	LoraRequestInfoMetricName                = "vllm:lora_requests_info"
	LoraRequestInfoRunningAdaptersMetricName = "running_lora_adapters"
	LoraRequestInfoMaxAdaptersMetricName     = "max_lora"
	// TODO: Replace these with the num_tokens_running/waiting below once we add those to the fork.
	RunningQueueSizeMetricName = "vllm:num_requests_running"
	WaitingQueueSizeMetricName = "vllm:num_requests_waiting"
	/* TODO: Uncomment this once the following are added to the fork.
	RunningQueueSizeMetricName        = "vllm:num_tokens_running"
	WaitingQueueSizeMetricName        = "vllm:num_tokens_waiting"
	*/
	KVCacheUsagePercentMetricName     = "vllm:gpu_cache_usage_perc"
	KvCacheMaxTokenCapacityMetricName = "vllm:gpu_cache_max_token_capacity"
)

type PodMetricsClientImpl struct{}

// FetchMetrics fetches metrics from a given pod.
func (p *PodMetricsClientImpl) FetchMetrics(
	ctx context.Context,
	existing *backend.PodMetrics,
) (*backend.PodMetrics, error) {
	logger := log.FromContext(ctx)
	loggerDefault := logger.V(logutil.DEFAULT)

	// Currently the metrics endpoint is hard-coded, which works with vLLM.
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/16): Consume this from InferencePool config.
	url := fmt.Sprintf("http://%s/metrics", existing.Address)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		loggerDefault.Error(err, "Failed create HTTP request", "method", http.MethodGet, "url", url)
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		loggerDefault.Error(err, "Failed to fetch metrics", "pod", existing.NamespacedName)
		return nil, fmt.Errorf("failed to fetch metrics from %s: %w", existing.NamespacedName, err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		loggerDefault.Error(nil, "Unexpected status code returned", "pod", existing.NamespacedName, "statusCode", resp.StatusCode)
		return nil, fmt.Errorf("unexpected status code from %s: %v", existing.NamespacedName, resp.StatusCode)
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
	existing *backend.PodMetrics,
) (*backend.PodMetrics, error) {
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
	var latestTs float64
	var latest *dto.Metric
	for _, m := range loraRequests.GetMetric() {
		if m.GetGauge().GetValue() > latestTs {
			latestTs = m.GetGauge().GetValue()
			latest = m
		}
	}
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
