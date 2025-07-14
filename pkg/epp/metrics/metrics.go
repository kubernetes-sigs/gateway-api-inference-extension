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
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/metrics"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	metricsutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/metrics"
)

const (
	InferenceModelComponent = "inference_model"
	InferencePoolComponent  = "inference_pool"
	InferenceExtension      = "inference_extension"
)

var (
	// The git hash of the latest commit in the build.
	CommitSHA string

	// The build ref from the _PULL_BASE_REF from cloud build trigger.
	BuildRef string
)

var (
	// Inference Model Metrics
	requestCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_total",
			Help:      metricsutil.HelpMsgWithStability("Counter of inference model requests broken out for each model and target model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "target_model_name"},
	)

	requestErrCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_error_total",
			Help:      metricsutil.HelpMsgWithStability("Counter of inference model requests errors broken out for each model and target model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "target_model_name", "error_code"},
	)

	requestTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model TTFT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3,
				4, 5, 6, 8, 10, 15, 20, 30, 45, 60, 120, 180, 240, 300, 360, 480, 600, 900, 1200, 1800, 2700, 3600,
			},
		},
		[]string{"model_name", "target_model_name"},
	)

	requestTTFTGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_ttft_seconds_gauge",
			Help:      metricsutil.HelpMsgWithStability("Inference model TTFT gauge in seconds for each model and target model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "target_model_name"},
	)

	requestPredictedTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_predicted_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model Predicted TTFT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3,
				4, 5, 6, 8, 10, 15, 20, 30, 45, 60, 120, 180, 240, 300, 360, 480, 600, 900, 1200, 1800, 2700, 3600,
			},
		},
		[]string{"model_name", "target_model_name"},
	)

	requestPredictedTTFTGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_predicted_ttft_seconds_gauge",
			Help:      metricsutil.HelpMsgWithStability("Inference model Predicted TTFT gauge in seconds for each model and target model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "target_model_name"},
	)

	// New metrics for TTFT prediction duration
	requestTTFTPredictionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_ttft_prediction_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Duration taken to generate TTFT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
			},
		},
		[]string{"model_name", "target_model_name"},
	)

	requestTTFTPredictionDurationGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_ttft_prediction_duration_seconds_gauge",
			Help:      metricsutil.HelpMsgWithStability("Latest duration taken to generate TTFT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "target_model_name"},
	)

	requestTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model TPOT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0005, 0.00205, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.125, 0.15, 0.2, 0.3,
				0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4.5, 6, 12, 18, 24, 30, 36, 48, 60, 90, 120, 180, 270, 360,
			},
		},
		[]string{"model_name", "target_model_name"},
	)

	requestTPOTGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_tpot_seconds_gauge",
			Help:      metricsutil.HelpMsgWithStability("Inference model TPOT gauge in seconds for each model and target model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "target_model_name"},
	)
	requestPredictedTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_predicted_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model Predicted TPOT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0005, 0.00205, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.125, 0.15, 0.2, 0.3,
				0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4.5, 6, 12, 18, 24, 30, 36, 48, 60, 90, 120, 180, 270, 360,
			},
		},
		[]string{"model_name", "target_model_name"},
	)

	requestPredictedTPOTGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_predicted_tpot_seconds_gauge",
			Help:      metricsutil.HelpMsgWithStability("Inference model Predicted TPOT gauge in seconds for each model and target model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "target_model_name"},
	)

	// New metrics for TPOT prediction duration
	requestTPOTPredictionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_tpot_prediction_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Duration taken to generate TPOT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
			},
		},
		[]string{"model_name", "target_model_name"},
	)

	requestTPOTPredictionDurationGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_tpot_prediction_duration_seconds_gauge",
			Help:      metricsutil.HelpMsgWithStability("Latest duration taken to generate TPOT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "target_model_name"},
	)

	requestTPOTPredictionMAPE = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_tpot_predictions_mape",
			Help:      metricsutil.HelpMsgWithStability("Inference model TPOT prediction mape distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets: []float64{
				1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60,
				70, 80, 90, 100,
			},
		},
		[]string{"model_name", "target_model_name"},
	)

	requestTPOTPredictionMAPEGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_tpot_predictions_mape_gauge",
			Help:      metricsutil.HelpMsgWithStability("Inference model TPOT prediction mape gauge in seconds for each model and target model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "target_model_name"},
	)

	requestTTFTPredictionMAPE = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_ttft_predictions_mape",
			Help:      metricsutil.HelpMsgWithStability("Inference model TTFT prediction mape distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets: []float64{
				1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60,
				70, 80, 90, 100,
			},
		},
		[]string{"model_name", "target_model_name"},
	)

	requestTTFTPredictionMAPEGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_ttft_predictions_mape_gauge",
			Help:      metricsutil.HelpMsgWithStability("Inference model TTFT prediction mape gauge in seconds for each model and target model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name", "target_model_name"},
	)

	requestLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model response latency distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3,
				4, 5, 6, 8, 10, 15, 20, 30, 45, 60, 120, 180, 240, 300, 360, 480, 600, 900, 1200, 1800, 2700, 3600,
			},
		},
		[]string{"model_name", "target_model_name"},
	)

	requestSizes = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "request_sizes",
			Help:      metricsutil.HelpMsgWithStability("Inference model requests size distribution in bytes for each model and target model.", compbasemetrics.ALPHA),
			// Use buckets ranging from 1000 bytes (1KB) to 10^9 bytes (1GB).
			Buckets: []float64{
				64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, // More fine-grained up to 64KB
				131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, // Exponential up to 8MB
				16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, // Exponential up to 1GB
			},
		},
		[]string{"model_name", "target_model_name"},
	)

	responseSizes = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "response_sizes",
			Help:      metricsutil.HelpMsgWithStability("Inference model responses size distribution in bytes for each model and target model.", compbasemetrics.ALPHA),
			// Most models have a response token < 8192 tokens. Each token, in average, has 4 characters.
			// 8192 * 4 = 32768.
			Buckets: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536},
		},
		[]string{"model_name", "target_model_name"},
	)

	inputTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "input_tokens",
			Help:      metricsutil.HelpMsgWithStability("Inference model input token count distribution for requests in each model.", compbasemetrics.ALPHA),
			// Most models have a input context window less than 1 million tokens.
			Buckets: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536, 131072, 262144, 524288, 1048576},
		},
		[]string{"model_name", "target_model_name"},
	)

	outputTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "output_tokens",
			Help:      metricsutil.HelpMsgWithStability("Inference model output token count distribution for requests in each model.", compbasemetrics.ALPHA),
			// Most models generates output less than 8192 tokens.
			Buckets: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
		},
		[]string{"model_name", "target_model_name"},
	)

	runningRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceModelComponent,
			Name:      "running_requests",
			Help:      metricsutil.HelpMsgWithStability("Inference model number of running requests in each model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name"},
	)

	// NTPOT - Normalized Time Per Output Token
	NormalizedTimePerOutputToken = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceModelComponent,
			Name:      "normalized_time_per_output_token_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model latency divided by number of output tokens in seconds for each model and target model.", compbasemetrics.ALPHA),
			// From few milliseconds per token to multiple seconds per token
			Buckets: []float64{
				0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
			},
		},
		[]string{"model_name", "target_model_name"},
	)

	// Inference Pool Metrics
	inferencePoolAvgKVCache = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferencePoolComponent,
			Name:      "average_kv_cache_utilization",
			Help:      metricsutil.HelpMsgWithStability("The average kv cache utilization for an inference server pool.", compbasemetrics.ALPHA),
		},
		[]string{"name"},
	)

	inferencePoolAvgQueueSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferencePoolComponent,
			Name:      "average_queue_size",
			Help:      metricsutil.HelpMsgWithStability("The average number of requests pending in the model server queue.", compbasemetrics.ALPHA),
		},
		[]string{"name"},
	)

	inferencePoolReadyPods = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferencePoolComponent,
			Name:      "ready_pods",
			Help:      metricsutil.HelpMsgWithStability("The number of ready pods in the inference server pool.", compbasemetrics.ALPHA),
		},
		[]string{"name"},
	)

	// Scheduler Metrics
	SchedulerE2ELatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceExtension,
			Name:      "scheduler_e2e_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("End-to-end scheduling latency distribution in seconds.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
			// StabilityLevel: prometheus.ALPHA,
		},
		[]string{},
	)
	SchedulerPluginProcessingLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceExtension,
			Name:      "scheduler_plugin_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Scheduler plugin processing latency distribution in seconds for each plugin type and plugin name.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{"plugin_type", "plugin_name"},
	)

	RequestControlPluginProcessingLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceExtension,
			Name:      "request_control_plugin_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("RequestControl plugin processing latency distribution in seconds for each plugin type and plugin name.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{"plugin_type", "plugin_name"},
	)

	// Prefix indexer Metrics
	PrefixCacheSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceExtension,
			Name:      "prefix_indexer_size",
			Help:      metricsutil.HelpMsgWithStability("Size of the prefix indexer.", compbasemetrics.ALPHA),
		},
		[]string{},
	)

	PrefixCacheHitRatio = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceExtension,
			Name:      "prefix_indexer_hit_ratio",
			Help:      metricsutil.HelpMsgWithStability("Ratio of prefix length matched to total prefix length in the cache lookup.", compbasemetrics.ALPHA),
			// Buckets from 0.0 to 1.0 in increments
			Buckets: []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{},
	)

	PrefixCacheHitLength = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceExtension,
			Name:      "prefix_indexer_hit_bytes",
			Help:      metricsutil.HelpMsgWithStability("Length of the prefix match in number of bytes in the cache lookup.", compbasemetrics.ALPHA),
			Buckets:   []float64{0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536},
		},
		[]string{},
	)

	// Info Metrics
	InferenceExtensionInfo = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceExtension,
			Name:      "info",
			Help:      metricsutil.HelpMsgWithStability("General information of the current build of Inference Extension.", compbasemetrics.ALPHA),
		},
		[]string{"commit", "build_ref"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register(customCollectors ...prometheus.Collector) {
	registerMetrics.Do(func() {
		metrics.Registry.MustRegister(requestTPOT)
		metrics.Registry.MustRegister(requestTTFT)

		metrics.Registry.MustRegister(requestTPOTGauge)
		metrics.Registry.MustRegister(requestTTFTGauge)

		metrics.Registry.MustRegister(requestPredictedTPOT)
		metrics.Registry.MustRegister(requestPredictedTTFT)

		metrics.Registry.MustRegister(requestPredictedTPOTGauge)
		metrics.Registry.MustRegister(requestPredictedTTFTGauge)

		// Register new prediction duration metrics
		metrics.Registry.MustRegister(requestTPOTPredictionDuration)
		metrics.Registry.MustRegister(requestTPOTPredictionDurationGauge)
		metrics.Registry.MustRegister(requestTTFTPredictionDuration)
		metrics.Registry.MustRegister(requestTTFTPredictionDurationGauge)

		metrics.Registry.MustRegister(requestTPOTPredictionMAPE)
		metrics.Registry.MustRegister(requestTTFTPredictionMAPE)
		metrics.Registry.MustRegister(requestTPOTPredictionMAPEGauge)
		metrics.Registry.MustRegister(requestTTFTPredictionMAPEGauge)
		metrics.Registry.MustRegister(requestCounter)
		metrics.Registry.MustRegister(requestErrCounter)
		metrics.Registry.MustRegister(requestLatencies)
		metrics.Registry.MustRegister(requestSizes)
		metrics.Registry.MustRegister(responseSizes)
		metrics.Registry.MustRegister(inputTokens)
		metrics.Registry.MustRegister(outputTokens)
		metrics.Registry.MustRegister(runningRequests)
		metrics.Registry.MustRegister(NormalizedTimePerOutputToken)
		metrics.Registry.MustRegister(inferencePoolAvgKVCache)
		metrics.Registry.MustRegister(inferencePoolAvgQueueSize)
		metrics.Registry.MustRegister(inferencePoolReadyPods)
		metrics.Registry.MustRegister(SchedulerPluginProcessingLatencies)
		metrics.Registry.MustRegister(SchedulerE2ELatency)
		metrics.Registry.MustRegister(RequestControlPluginProcessingLatencies)
		metrics.Registry.MustRegister(InferenceExtensionInfo)
		metrics.Registry.MustRegister(PrefixCacheSize)
		metrics.Registry.MustRegister(PrefixCacheHitRatio)
		metrics.Registry.MustRegister(PrefixCacheHitLength)
		for _, collector := range customCollectors {
			metrics.Registry.MustRegister(collector)
		}
	})
}

// Just for integration test
func Reset() {
	requestCounter.Reset()
	requestErrCounter.Reset()
	requestLatencies.Reset()
	requestSizes.Reset()
	responseSizes.Reset()
	inputTokens.Reset()
	outputTokens.Reset()
	runningRequests.Reset()
	NormalizedTimePerOutputToken.Reset()
	inferencePoolAvgKVCache.Reset()
	inferencePoolAvgQueueSize.Reset()
	inferencePoolReadyPods.Reset()
	SchedulerPluginProcessingLatencies.Reset()
	SchedulerE2ELatency.Reset()
	RequestControlPluginProcessingLatencies.Reset()
	InferenceExtensionInfo.Reset()
	PrefixCacheSize.Reset()
	PrefixCacheHitRatio.Reset()
	PrefixCacheHitLength.Reset()

	requestTPOT.Reset()
	requestTTFT.Reset()
	requestTPOTGauge.Reset()
	requestTTFTGauge.Reset()

	requestTPOTPredictionMAPE.Reset()
	requestTPOTPredictionMAPEGauge.Reset()
	requestTTFTPredictionMAPE.Reset()
	requestTTFTPredictionMAPEGauge.Reset()

	requestPredictedTPOT.Reset()
	requestPredictedTTFT.Reset()
	requestPredictedTPOTGauge.Reset()
	requestPredictedTTFTGauge.Reset()

	// Reset new prediction duration metrics
	requestTPOTPredictionDuration.Reset()
	requestTPOTPredictionDurationGauge.Reset()
	requestTTFTPredictionDuration.Reset()
	requestTTFTPredictionDurationGauge.Reset()
}

// RecordRequstCounter records the number of requests.
func RecordRequestCounter(modelName, targetModelName string) {
	requestCounter.WithLabelValues(modelName, targetModelName).Inc()
}

// RecordRequestErrCounter records the number of error requests.
func RecordRequestErrCounter(modelName, targetModelName string, code string) {
	if code != "" {
		requestErrCounter.WithLabelValues(modelName, targetModelName, code).Inc()
	}
}

// RecordRequestSizes records the request sizes.
func RecordRequestSizes(modelName, targetModelName string, reqSize int) {
	requestSizes.WithLabelValues(modelName, targetModelName).Observe(float64(reqSize))
}

// RecordRequestLatencies records duration of request.
func RecordRequestLatencies(ctx context.Context, modelName, targetModelName string, received time.Time, complete time.Time) bool {
	if !complete.After(received) {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "Request latency values are invalid",
			"modelName", modelName, "targetModelName", targetModelName, "completeTime", complete, "receivedTime", received)
		return false
	}
	elapsedSeconds := complete.Sub(received).Seconds()
	requestLatencies.WithLabelValues(modelName, targetModelName).Observe(elapsedSeconds)
	return true
}

func RecordRequestTPOT(ctx context.Context, modelName, targetModelName string, tpot float64) bool {
	if tpot < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TPOT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "tpot", tpot)
		return false
	}
	requestTPOT.WithLabelValues(modelName, targetModelName).Observe(tpot)
	requestTPOTGauge.WithLabelValues(modelName, targetModelName).Set(tpot)
	return true
}

// TPOT records duration of request.
func RecordRequestPredictedTPOT(ctx context.Context, modelName, targetModelName string, predicted_tpot float64) bool {
	if predicted_tpot < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "Predicted TPOT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "tpot", predicted_tpot)
		return false
	}
	requestPredictedTPOT.WithLabelValues(modelName, targetModelName).Observe(predicted_tpot)
	requestPredictedTPOTGauge.WithLabelValues(modelName, targetModelName).Set(predicted_tpot)
	return true
}

// RecordRequestTPOTPredictionDuration records the duration taken to generate TPOT predictions.
func RecordRequestTPOTPredictionDuration(ctx context.Context, modelName, targetModelName string, duration float64) bool {
	if duration < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TPOT prediction duration must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "duration", duration)
		return false
	}
	requestTPOTPredictionDuration.WithLabelValues(modelName, targetModelName).Observe(duration)
	requestTPOTPredictionDurationGauge.WithLabelValues(modelName, targetModelName).Set(duration)
	return true
}

// TTFT records duration of request.
func RecordRequestTTFT(ctx context.Context, modelName, targetModelName string, ttft float64) bool {
	if ttft < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TTFT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "ttft", ttft)
		return false
	}
	requestTTFT.WithLabelValues(modelName, targetModelName).Observe(ttft)
	requestTTFTGauge.WithLabelValues(modelName, targetModelName).Set(ttft)
	return true
}

// TPOT records duration of request.
func RecordRequestPredictedTTFT(ctx context.Context, modelName, targetModelName string, predicted_ttft float64) bool {
	if predicted_ttft < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "Predicted TTFT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "ttft", predicted_ttft)
		return false
	}
	requestPredictedTTFT.WithLabelValues(modelName, targetModelName).Observe(predicted_ttft)
	requestPredictedTTFTGauge.WithLabelValues(modelName, targetModelName).Set(predicted_ttft)
	return true
}

// RecordRequestTTFTPredictionDuration records the duration taken to generate TTFT predictions.
func RecordRequestTTFTPredictionDuration(ctx context.Context, modelName, targetModelName string, duration float64) bool {
	if duration < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TTFT prediction duration must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "duration", duration)
		return false
	}
	requestTTFTPredictionDuration.WithLabelValues(modelName, targetModelName).Observe(duration)
	requestTTFTPredictionDurationGauge.WithLabelValues(modelName, targetModelName).Set(duration)
	return true
}

func RecordRequestTPOTPredictionMape(ctx context.Context, modelName, targetModelName string, mape float64) bool {
	requestTPOTPredictionMAPE.WithLabelValues(modelName, targetModelName).Observe(mape)
	requestTPOTPredictionMAPEGauge.WithLabelValues(modelName, targetModelName).Set(mape)
	return true
}

func RecordRequestTTFTPredictionMape(ctx context.Context, modelName, targetModelName string, mape float64) bool {
	requestTTFTPredictionMAPE.WithLabelValues(modelName, targetModelName).Observe(mape)
	requestTTFTPredictionMAPEGauge.WithLabelValues(modelName, targetModelName).Set(mape)
	return true
}

// RecordResponseSizes records the response sizes.
func RecordResponseSizes(modelName, targetModelName string, size int) {
	responseSizes.WithLabelValues(modelName, targetModelName).Observe(float64(size))
}

// RecordInputTokens records input tokens count.
func RecordInputTokens(modelName, targetModelName string, size int) {
	if size > 0 {
		inputTokens.WithLabelValues(modelName, targetModelName).Observe(float64(size))
	}
}

// RecordOutputTokens records output tokens count.
func RecordOutputTokens(modelName, targetModelName string, size int) {
	if size > 0 {
		outputTokens.WithLabelValues(modelName, targetModelName).Observe(float64(size))
	}
}

// RecordNormalizedTimePerOutputToken (NTPOT) records the normalized time per output token.
func RecordNormalizedTimePerOutputToken(ctx context.Context, modelName, targetModelName string, received time.Time, complete time.Time, outputTokenCount int) bool {
	if !complete.After(received) {
		log.FromContext(ctx).Error(nil, "Request latency values are invalid for NTPOT calculation",
			"modelName", modelName, "targetModelName", targetModelName, "completeTime", complete, "receivedTime", received)
		return false
	}

	if outputTokenCount <= 0 {
		log.FromContext(ctx).Error(nil, "Output token count must be positive for NTPOT calculation",
			"modelName", modelName, "targetModelName", targetModelName, "outputTokenCount", outputTokenCount)
		return false
	}

	elapsedSeconds := complete.Sub(received).Seconds()
	secondsPerToken := elapsedSeconds / float64(outputTokenCount)

	NormalizedTimePerOutputToken.WithLabelValues(modelName, targetModelName).Observe(secondsPerToken)
	return true
}

// IncRunningRequests increases the current running requests.
func IncRunningRequests(modelName string) {
	if modelName != "" {
		runningRequests.WithLabelValues(modelName).Inc()
	}
}

// DecRunningRequests decreases the current running requests.
func DecRunningRequests(modelName string) {
	if modelName != "" {
		runningRequests.WithLabelValues(modelName).Dec()
	}
}

func RecordInferencePoolAvgKVCache(name string, utilization float64) {
	inferencePoolAvgKVCache.WithLabelValues(name).Set(utilization)
}

func RecordInferencePoolAvgQueueSize(name string, queueSize float64) {
	inferencePoolAvgQueueSize.WithLabelValues(name).Set(queueSize)
}

func RecordInferencePoolReadyPods(name string, runningPods float64) {
	inferencePoolReadyPods.WithLabelValues(name).Set(runningPods)
}

// RecordSchedulerPluginProcessingLatency records the processing latency for a scheduler plugin.
func RecordSchedulerPluginProcessingLatency(pluginType, pluginName string, duration time.Duration) {
	SchedulerPluginProcessingLatencies.WithLabelValues(pluginType, pluginName).Observe(duration.Seconds())
}

// RecordSchedulerE2ELatency records the end-to-end scheduling latency.
func RecordSchedulerE2ELatency(duration time.Duration) {
	SchedulerE2ELatency.WithLabelValues().Observe(duration.Seconds())
}

// RecordRequestControlPluginProcessingLatency records the processing latency for a request-control plugin.
func RecordRequestControlPluginProcessingLatency(pluginType, pluginName string, duration time.Duration) {
	RequestControlPluginProcessingLatencies.WithLabelValues(pluginType, pluginName).Observe(duration.Seconds())
}

// RecordPrefixCacheSize records the size of the prefix indexer in megabytes.
func RecordPrefixCacheSize(size int64) {
	PrefixCacheSize.WithLabelValues().Set(float64(size))
}

// RecordPrefixCacheMatch records both the hit ratio and hit length for a prefix indexer match.
// matchedLength is the number of characters that matched, and totalLength is the total prefix length.
func RecordPrefixCacheMatch(matchedLength, totalLength int) {
	// Record the hit length metric
	PrefixCacheHitLength.WithLabelValues().Observe(float64(matchedLength))

	// Record the hit ratio metric if totalLength is positive
	if totalLength > 0 {
		ratio := float64(matchedLength) / float64(totalLength)
		PrefixCacheHitRatio.WithLabelValues().Observe(ratio)
	}
}

func RecordInferenceExtensionInfo() {
	InferenceExtensionInfo.WithLabelValues(CommitSHA, BuildRef).Set(1)
}