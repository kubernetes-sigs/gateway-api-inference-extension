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

package triton

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/go-logr/logr"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"go.uber.org/multierr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// Triton metrics, see https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html

	TRTLLMRequestMetricsName  = "nv_trt_llm_request_metrics"
	TRTLLMKvCacheMetricsName  = "nv_trt_llm_kv_cache_block_metrics"
	TRTLLMKvCacheMetricsLabel = "kv_cache_block_type"
	TRTLLMRequestMetricsLabel = "request_type"
)

type PodMetricsClientImpl struct{}

// FetchMetrics fetches metrics from a given pod.
func (p *PodMetricsClientImpl) FetchMetrics(
	ctx context.Context,
	existing *datastore.PodMetrics,
	port int32,
) (*datastore.PodMetrics, error) {
	logger := log.FromContext(ctx)
	loggerDefault := logger.V(logutil.DEFAULT)

	// existing.ScrapePort = 8002 // triton has a different port for metrics than the target port for inference
	url := "http://" + existing.Address + ":" + strconv.Itoa(int(port)) + "/metrics"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	// TODO print response and err

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

// promToPodMetrics updates internal pod metrics with scraped Prometheus metrics.
func promToPodMetrics(
	logger logr.Logger,
	metricFamilies map[string]*dto.MetricFamily,
	existing *datastore.PodMetrics,
) (*datastore.PodMetrics, error) {
	var errs error
	updated := existing.Clone()

	//fmt.Print("\n\nDEBUG START\n###### DEBUG getting REQUEST metrics... ######")
	// Get the "nv_trt_llm_request_metrics" metric family
	//requestMetrics, err := getLatestMetric(logger, metricFamilies, TRTLLMRequestMetricsName)
	requestMetrics, ok := metricFamilies[TRTLLMRequestMetricsName]
	//errs = multierr.Append(errs, err)
	if ok {
		if scheduled, err := getTrtLlmGaugeMetric(logger, requestMetrics, TRTLLMRequestMetricsLabel, "scheduled"); err == nil {
			//fmt.Printf("\n###### DEBUG generation_requests: %+v", generation_requests)
			updated.Metrics.RunningQueueSize = int(scheduled)
			if active, err := getTrtLlmGaugeMetric(logger, requestMetrics, TRTLLMRequestMetricsLabel, "active"); err == nil {
				//fmt.Printf("\n###### DEBUG scheduled: %+v", scheduled)
				updated.Metrics.WaitingQueueSize = int(active - scheduled)
				// pendingMetrics, ok := metricFamilies["nv_inference_pending_request_count"]
				// if ok {
				// 	if queued, err := getTrtLlmGaugeMetric(logger, pendingMetrics, "model", "ensemble"); err == nil {
				// 		fmt.Printf("\n###### DEBUG queued requests: %+v", int(queued))
				// 	}
				// }
				//fmt.Printf("\n###### DEBUG active (total) requests: %+v", int(active))
				//fmt.Printf("\n###### DEBUG waiting requests: %+v", int(active-scheduled))
				//fmt.Printf("\n###### DEBUG running requests: %+v", int(scheduled))
			} else {
				errs = multierr.Append(errs, err)
			}
		} else {
			errs = multierr.Append(errs, err)
		}
	}

	//fmt.Print("\n\n###### DEBUG getting KVBLOCK metrics... ######")
	// Get the "nv_trt_llm_kv_cache_block_metrics" metric family
	kvCacheBlocks, ok := metricFamilies[TRTLLMKvCacheMetricsName]
	// errs = multierr.Append(errs, err)
	// fmt.Printf("###### DEBUG (should be nil) getLatestMetric errs: %+v", errs)
	if ok {
		// Calculate the kv-cache usage from the max and used metrics
		if max, err := getTrtLlmGaugeMetric(logger, kvCacheBlocks, TRTLLMKvCacheMetricsLabel, "max"); err == nil {
			//fmt.Printf("\n###### DEBUG max: %+v", max)
			if used, err := getTrtLlmGaugeMetric(logger, kvCacheBlocks, TRTLLMKvCacheMetricsLabel, "used"); err == nil {
				//fmt.Printf("\n###### DEBUG used: %+v", used)
				usage := 0.0
				if max > 0 {
					usage = used / max
				}
				updated.Metrics.KVCacheUsagePercent = usage
			} else {
				errs = multierr.Append(errs, err)
			}
		} else {
			errs = multierr.Append(errs, err)
		}
	}

	//fmt.Printf("\n### DEBUG: %+v", updated)
	//fmt.Printf("\n###### DEBUG ERRORS: %+v", errs)

	return updated, errs
}

// getLatestMetric gets the latest metric of a family.
func getLatestMetric(logger logr.Logger, metricFamilies map[string]*dto.MetricFamily, metricName string) (*dto.MetricFamily, error) {
	mf, ok := metricFamilies[metricName]
	if !ok {
		logger.V(logutil.DEFAULT).Error(nil, "Metric family not found", "name", metricName)
		return nil, fmt.Errorf("metric family %q not found", metricName)
	}
	if len(mf.GetMetric()) == 0 {
		return nil, fmt.Errorf("no metrics available for %q", metricName)
	}

	var latestTs int64
	var latestMf *dto.MetricFamily
	for _, m := range mf.GetMetric() {
		if m.GetTimestampMs() >= latestTs {
			latestTs = m.GetTimestampMs()
			latestMf = &dto.MetricFamily{
				Name:   mf.Name,
				Help:   mf.Help,
				Type:   mf.Type,
				Metric: []*dto.Metric{m},
			}
		}
	}

	logger.V(logutil.TRACE).Info("Metric value selected", "metric Family", latestMf, "metric", metricName)
	return latestMf, nil
}

// getGaugeMetricForPod gets gauge metric value for a given pod.
func getGaugeMetricForPod(logger logr.Logger, mf *dto.MetricFamily, podIdentifier string) (float64, error) {
	for _, m := range mf.GetMetric() {
		for _, label := range m.GetLabel() {
			if (label.GetName() == "pod" || label.GetName() == "gpu_uuid") && strings.Contains(label.GetValue(), podIdentifier) {
				logger.V(logutil.TRACE).Info("Pod metric found", "value", m.GetGauge().GetValue(), "labelName", label.GetName(), "labelValue", label.GetValue())

				return m.GetGauge().GetValue(), nil // Return the value with nil error
			}
		}
	}
	logger.V(logutil.TRACE).Info("Metric Value not found for pod", "pod", podIdentifier, "metric family", mf.GetName())
	return -1, fmt.Errorf("metric value not found for pod %s in metric family %s", podIdentifier, mf.GetName()) // Return an error
}

// getCounterMetricForPod gets counter metric value for a given pod.
func getCounterMetricForPod(logger logr.Logger, mf *dto.MetricFamily, podName string) (int, error) {
	for _, m := range mf.GetMetric() {
		for _, label := range m.GetLabel() {
			if label.GetName() == "pod" && label.GetValue() == podName {
				val := m.GetCounter().GetValue()
				intVal, err := strconv.Atoi(fmt.Sprintf("%v", val)) // Convert float64 to int
				if err != nil {
					return -1, fmt.Errorf("failed to convert counter metric to int: %w", err)
				}
				logger.V(logutil.TRACE).Info("Pod metric found", "value", intVal)

				return intVal, nil
			}
		}
	}
	return -1, nil
}

// TRTLLM metrics

// getTrtLlmMetric gets a TRT LLM metric with the specified type, key, and value.
func getTrtLlmMetric(logger logr.Logger, mf *dto.MetricFamily, metricType dto.MetricType, key, value string) (float64, error) {
	//fmt.Printf("###### DEBUG START GETTRTMERTIC: %+v", mf.GetMetric())
	//fmt.Printf("###### DEBUG METRICS: %+v", len(mf.GetMetric()))
	for _, m := range mf.GetMetric() {
		//fmt.Printf("###### DEBUG ANALYZING METRIC: %+v", m)
		//fmt.Printf("###### DEBUG TIMESTAMP: %+v", m.GetTimestampMs())
		foundKey := false
		foundValue := false
		//fmt.Printf("###### DEBUG LABELS: %+v", m.GetLabel())
		for _, label := range m.GetLabel() {
			//fmt.Printf("###### DEBUG COMPARING label NAME %+v == %+v and label VALUE %+v == %+v", label.GetName(), key, label.GetValue(), value)
			if label.GetName() == key && label.GetValue() == value {
				foundKey = true
			}
			if mf.GetType() == metricType {
				foundValue = true
			}
		}
		if foundKey && foundValue {
			//fmt.Printf("###### DEBUG METRIC FOUND: %+v", m)
			if metricType == dto.MetricType_GAUGE {
				logger.V(logutil.TRACE).Info("TRT LLM gauge metric found", "value", m.GetGauge().GetValue(), "key", key, "value", value)
				return m.GetGauge().GetValue(), nil
			} else if metricType == dto.MetricType_COUNTER {
				val := m.GetCounter().GetValue()
				intVal, err := strconv.Atoi(fmt.Sprintf("%v", val))
				if err != nil {
					return -1, fmt.Errorf("failed to convert counter metric to int: %w", err)
				}
				logger.V(logutil.TRACE).Info("TRT LLM counter metric found", "value", intVal, "key", key, "value", value)
				return float64(intVal), nil
			}
		}
	}
	return -1, fmt.Errorf("TRT LLM metric not found: %s{ %s=\"%s\" }", mf.GetName(), key, value)
}

// getTrtLlmGaugeMetric gets a gauge TRT LLM metric.
func getTrtLlmGaugeMetric(logger logr.Logger, mf *dto.MetricFamily, key, value string) (float64, error) {
	return getTrtLlmMetric(logger, mf, dto.MetricType_GAUGE, key, value)
}

// getTrtLlmCounterMetric gets a counter TRT LLM metric.
func getTrtLlmCounterMetric(logger logr.Logger, mf *dto.MetricFamily, key, value string) (float64, error) {
	return getTrtLlmMetric(logger, mf, dto.MetricType_COUNTER, key, value)
}
