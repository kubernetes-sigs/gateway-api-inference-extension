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
	// --- Subsystems ---
	InferenceObjectiveComponent = "inference_objective"
	InferencePoolComponent      = "inference_pool"
	InferenceExtension          = "inference_extension"

	// --- Internal Keys (for Legacy/Gauge Usage) ---
	KVCacheUsagePercentKey = "KVCacheUsagePercent"
	WaitingQueueSizeKey    = "WaitingQueueSize"
	RunningRequestsSizeKey = "RunningRequestsSize"
	MaxActiveModelsKey     = "MaxActiveModels"
	ActiveModelsKey        = "ActiveModels"
	WaitingModelsKey       = "WaitingModels"
	UpdateTimeKey          = "UpdateTime"

	// Metric Type Values
	TypeTPOT                   = "tpot"
	TypePredictedTPOT          = "predicted_tpot"
	TypeTPOTPredictionDuration = "tpot_prediction_duration"
	TypeTPOTSLOViolation       = "tpot_slo_violation"
	TypeTPOTSLOThreshold       = "tpot_slo_threshold"

	TypeTTFT                   = "ttft"
	TypePredictedTTFT          = "predicted_ttft"
	TypeTTFTPredictionDuration = "ttft_prediction_duration"
	TypeTTFTSLOViolation       = "ttft_slo_violation"
	TypeTTFTSLOThreshold       = "ttft_slo_threshold"
)

var (
	// --- Common Label Sets ---
	ModelLabels     = []string{"model_name", "target_model_name"}
	ModelTypeLabels = []string{"model_name", "target_model_name", "type"}
	PoolLabels      = []string{"name"}

	// --- Common Buckets ---

	// GeneralLatencyBuckets for long running inference from 5ms to 1 hour
	GeneralLatencyBuckets = []float64{
		0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3, 4, 5, 6,
		8, 10, 15, 20, 30, 45, 60, 120, 180, 240, 300, 360, 480, 600, 900, 1200,
		1800, 2700, 3600,
	}

	// TPOTBuckets for time-per-output-token (usually milliseconds to seconds)
	TPOTBuckets = []float64{
		0.0005, 0.00205, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.125, 0.15, 0.2,
		0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4.5, 6, 12, 18, 24, 30, 36, 48, 60,
		90, 120, 180, 270, 360,
	}

	// PredictionLatencyBuckets for internal latency (predictions) from 100us to 5s
	PredictionLatencyBuckets = []float64{
		0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
	}
)

// --- Inference Objective Metrics ---
var (
	requestCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "request_total",
			Help:      metricsutil.HelpMsgWithStability("Counter of inference objective requests broken out for each model and target model.", compbasemetrics.ALPHA),
		},
		ModelLabels,
	)

	requestErrCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "request_error_total",
			Help:      metricsutil.HelpMsgWithStability("Counter of inference objective requests errors broken out for each model and target model.", compbasemetrics.ALPHA),
		},
		append(ModelLabels, "error_code"),
	)

	inferenceGauges = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "inference_request_metric",
			Help:      metricsutil.HelpMsgWithStability("Consolidated gauge for various inference request metrics including TTFT, TPOT, SLOs, and prediction durations.", compbasemetrics.ALPHA),
		},
		ModelTypeLabels,
	)

	requestTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "request_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model TTFT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   GeneralLatencyBuckets,
		},
		ModelLabels,
	)

	requestPredictedTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "request_predicted_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model Predicted TTFT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   GeneralLatencyBuckets,
		},
		ModelLabels,
	)

	requestTTFTPredictionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "request_ttft_prediction_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Duration taken to generate TTFT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   PredictionLatencyBuckets,
		},
		ModelLabels,
	)

	requestTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "request_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model TPOT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   TPOTBuckets,
		},
		ModelLabels,
	)

	requestPredictedTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "request_predicted_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference model Predicted TPOT distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   TPOTBuckets,
		},
		ModelLabels,
	)

	requestTPOTPredictionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "request_tpot_prediction_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Duration taken to generate TPOT predictions in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   PredictionLatencyBuckets,
		},
		ModelLabels,
	)

	sloViolationCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "request_slo_violation_total",
			Help:      metricsutil.HelpMsgWithStability("Counter of SLO violations for each model, target model, and violation type.", compbasemetrics.ALPHA),
		},
		ModelTypeLabels,
	)

	requestLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "request_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference objective response latency distribution in seconds for each model and target model.", compbasemetrics.ALPHA),
			Buckets:   GeneralLatencyBuckets,
		},
		ModelLabels,
	)

	requestSizes = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "request_sizes",
			Help:      metricsutil.HelpMsgWithStability("Inference objective requests size distribution in bytes for each model and target model.", compbasemetrics.ALPHA),
			// Use buckets ranging from 1000 bytes (1KB) to 10^9 bytes (1GB).
			Buckets: []float64{
				64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, // More fine-grained up to 64KB
				131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, // Exponential up to 8MB
				16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, // Exponential up to 1GB
			},
		},
		ModelLabels,
	)

	responseSizes = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "response_sizes",
			Help:      metricsutil.HelpMsgWithStability("Inference objective responses size distribution in bytes for each model and target model.", compbasemetrics.ALPHA),
			// Most models have a response token < 8192 tokens. Each token, in average, has 4 characters.
			// 8192 * 4 = 32768.
			Buckets: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536},
		},
		ModelLabels,
	)

	inputTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "input_tokens",
			Help:      metricsutil.HelpMsgWithStability("Inference objective input token count distribution for requests in each model.", compbasemetrics.ALPHA),
			// Most models have a input context window less than 1 million tokens.
			Buckets: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536, 131072, 262144, 524288, 1048576},
		},
		ModelLabels,
	)

	outputTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "output_tokens",
			Help:      metricsutil.HelpMsgWithStability("Inference objective output token count distribution for requests in each model.", compbasemetrics.ALPHA),
			// Most models generates output less than 8192 tokens.
			Buckets: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
		},
		ModelLabels,
	)

	promptCachedTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "prompt_cached_tokens",
			Help:      metricsutil.HelpMsgWithStability("Inference objective prompt cached token count distribution for requests in each model.", compbasemetrics.ALPHA),
			// Most models have a input context window less than 1 million tokens.
			Buckets: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32778, 65536, 131072, 262144, 524288, 1048576},
		},
		ModelLabels,
	)

	runningRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "running_requests",
			Help:      metricsutil.HelpMsgWithStability("Inference objective number of running requests in each model.", compbasemetrics.ALPHA),
		},
		[]string{"model_name"},
	)

	// NTPOT - Normalized Time Per Output Token
	NormalizedTimePerOutputToken = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceObjectiveComponent,
			Name:      "normalized_time_per_output_token_seconds",
			Help:      metricsutil.HelpMsgWithStability("Inference objective latency divided by number of output tokens in seconds for each model and target model.", compbasemetrics.ALPHA),
			// From few milliseconds per token to multiple seconds per token
			Buckets: []float64{
				0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
			},
		},
		ModelLabels,
	)
)

// --- Inference Pool Metrics ---
var (
	inferencePoolAvgKVCache = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferencePoolComponent,
			Name:      "average_kv_cache_utilization",
			Help:      metricsutil.HelpMsgWithStability("The average kv cache utilization for an inference server pool.", compbasemetrics.ALPHA),
		},
		PoolLabels,
	)

	inferencePoolAvgQueueSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferencePoolComponent,
			Name:      "average_queue_size",
			Help:      metricsutil.HelpMsgWithStability("The average number of requests pending in the model server queue.", compbasemetrics.ALPHA),
		},
		PoolLabels,
	)

	inferencePoolReadyPods = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferencePoolComponent,
			Name:      "ready_pods",
			Help:      metricsutil.HelpMsgWithStability("The number of ready pods in the inference server pool.", compbasemetrics.ALPHA),
		},
		PoolLabels,
	)
)

// --- Scheduling Metrics ---
var (
	SchedulerE2ELatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceExtension,
			Name:      "scheduler_e2e_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("End-to-end scheduling latency distribution in seconds.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{},
	)

	SchedulerAttemptsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: InferenceExtension,
			Name:      "scheduler_attempts_total",
			Help:      metricsutil.HelpMsgWithStability("Total number of scheduling attempts.", compbasemetrics.ALPHA),
		},
		[]string{"status"}, // "success", "failure"
	)

	PluginProcessingLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceExtension,
			Name:      "plugin_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Plugin processing latency distribution in seconds for each extension point, plugin type and plugin name.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
			},
		},
		[]string{"extension_point", "plugin_type", "plugin_name"},
	)

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
)

// --- Info Metrics ---
var InferenceExtensionInfo = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: InferenceExtension,
		Name:      "info",
		Help:      metricsutil.HelpMsgWithStability("General information of the current build of Inference Extension.", compbasemetrics.ALPHA),
	},
	[]string{"commit", "build_ref"},
)

// --- Flow Control Metrics ---
var (
	flowControlRequestQueueDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: InferenceExtension,
			Name:      "flow_control_request_queue_duration_seconds",
			Help:      metricsutil.HelpMsgWithStability("Distribution of the total time requests spend in the EPP flow control layer, measured from the start of the EnqueueAndWait call until a final outcome is reached.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
			},
		},
		[]string{"fairness_id", "priority", "outcome"},
	)

	flowControlQueueSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: InferenceExtension,
			Name:      "flow_control_queue_size",
			Help:      metricsutil.HelpMsgWithStability("Current number of requests being actively managed by the EPP flow control layer, from the start of the EnqueueAndWait call until a final outcome is reached.", compbasemetrics.ALPHA),
		},
		[]string{"fairness_id", "priority"},
	)
)

// --- Inference Model Rewrite Metrics ---
var inferenceModelRewriteDecisionsTotal = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: InferenceExtension,
		Name:      "model_rewrite_decisions_total",
		Help:      metricsutil.HelpMsgWithStability("Total number of inference model rewrite decisions.", compbasemetrics.ALPHA),
	},
	[]string{"model_rewrite_name", "model_name", "target_model"},
)

var registerMetrics sync.Once

// Register all metrics.
func Register(customCollectors ...prometheus.Collector) {
	registerMetrics.Do(func() {
		// Register inference gauges
		metrics.Registry.MustRegister(inferenceGauges)

		// Register Histograms
		metrics.Registry.MustRegister(requestTPOT)
		metrics.Registry.MustRegister(requestTTFT)
		metrics.Registry.MustRegister(requestPredictedTPOT)
		metrics.Registry.MustRegister(requestPredictedTTFT)
		metrics.Registry.MustRegister(requestTPOTPredictionDuration)
		metrics.Registry.MustRegister(requestTTFTPredictionDuration)

		// Register SLO violation counters
		metrics.Registry.MustRegister(sloViolationCounter)

		// Register other metrics
		metrics.Registry.MustRegister(requestCounter)
		metrics.Registry.MustRegister(requestErrCounter)
		metrics.Registry.MustRegister(requestLatencies)
		metrics.Registry.MustRegister(requestSizes)
		metrics.Registry.MustRegister(responseSizes)
		metrics.Registry.MustRegister(inputTokens)
		metrics.Registry.MustRegister(outputTokens)
		metrics.Registry.MustRegister(promptCachedTokens)
		metrics.Registry.MustRegister(runningRequests)
		metrics.Registry.MustRegister(NormalizedTimePerOutputToken)
		metrics.Registry.MustRegister(inferencePoolAvgKVCache)
		metrics.Registry.MustRegister(inferencePoolAvgQueueSize)
		metrics.Registry.MustRegister(inferencePoolReadyPods)
		metrics.Registry.MustRegister(SchedulerE2ELatency)
		metrics.Registry.MustRegister(SchedulerAttemptsTotal)
		metrics.Registry.MustRegister(PluginProcessingLatencies)
		metrics.Registry.MustRegister(InferenceExtensionInfo)
		metrics.Registry.MustRegister(PrefixCacheSize)
		metrics.Registry.MustRegister(PrefixCacheHitRatio)
		metrics.Registry.MustRegister(PrefixCacheHitLength)
		metrics.Registry.MustRegister(flowControlRequestQueueDuration)
		metrics.Registry.MustRegister(flowControlQueueSize)
		metrics.Registry.MustRegister(inferenceModelRewriteDecisionsTotal)
		for _, collector := range customCollectors {
			metrics.Registry.MustRegister(collector)
		}
	})
}

// Just for integration test
func Reset() {
	// Reset inference gauges
	inferenceGauges.Reset()

	// Reset Histograms
	requestTPOT.Reset()
	requestTTFT.Reset()
	requestPredictedTPOT.Reset()
	requestPredictedTTFT.Reset()
	requestTPOTPredictionDuration.Reset()
	requestTTFTPredictionDuration.Reset()

	// Reset SLO violation counter
	sloViolationCounter.Reset()

	// Reset other metrics
	requestCounter.Reset()
	requestErrCounter.Reset()
	requestLatencies.Reset()
	requestSizes.Reset()
	responseSizes.Reset()
	inputTokens.Reset()
	outputTokens.Reset()
	promptCachedTokens.Reset()
	runningRequests.Reset()
	NormalizedTimePerOutputToken.Reset()
	inferencePoolAvgKVCache.Reset()
	inferencePoolAvgQueueSize.Reset()
	inferencePoolReadyPods.Reset()
	SchedulerE2ELatency.Reset()
	SchedulerAttemptsTotal.Reset()
	PluginProcessingLatencies.Reset()
	InferenceExtensionInfo.Reset()
	PrefixCacheSize.Reset()
	PrefixCacheHitRatio.Reset()
	PrefixCacheHitLength.Reset()
	flowControlRequestQueueDuration.Reset()
	flowControlQueueSize.Reset()
	inferenceModelRewriteDecisionsTotal.Reset()
}

// RecordRequestCounter records the number of requests.
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
	inferenceGauges.WithLabelValues(modelName, targetModelName, TypeTPOT).Set(tpot)
	return true
}

// RecordRequestTPOTWithSLO records TPOT and checks for SLO violation.
// If tpot exceeds the threshold, it records a violation (sets gauge to 1 and increments counter).
// If tpot is within limits, it sets gauge to 0.
func RecordRequestTPOTWithSLO(ctx context.Context, modelName, targetModelName string, tpot float64, sloThreshold float64) bool {
	if tpot < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TPOT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "tpot", tpot)
		return false
	}

	// Check for SLO violation (tpot exceeds threshold)
	if tpot > sloThreshold {
		inferenceGauges.WithLabelValues(modelName, targetModelName, TypeTPOTSLOViolation).Set(1)
		sloViolationCounter.WithLabelValues(modelName, targetModelName, TypeTPOT).Inc()
		log.FromContext(ctx).V(logutil.DEFAULT).Info("TPOT SLO violation detected",
			"modelName", modelName, "targetModelName", targetModelName, "tpot", tpot, "threshold", sloThreshold)
	} else {
		inferenceGauges.WithLabelValues(modelName, targetModelName, TypeTPOTSLOViolation).Set(0)
	}

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
	inferenceGauges.WithLabelValues(modelName, targetModelName, TypePredictedTPOT).Set(predicted_tpot)
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
	inferenceGauges.WithLabelValues(modelName, targetModelName, TypeTPOTPredictionDuration).Set(duration)
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
	inferenceGauges.WithLabelValues(modelName, targetModelName, TypeTTFT).Set(ttft)
	return true
}

// RecordRequestTTFTWithSLO records TTFT and checks for SLO violation.
// If ttft exceeds the threshold, it records a violation (sets gauge to 1 and increments counter).
// If ttft is within limits, it sets gauge to 0.
func RecordRequestTTFTWithSLO(ctx context.Context, modelName, targetModelName string, ttft float64, sloThreshold float64) bool {
	if ttft < 0 {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(nil, "TTFT value must be non-negative",
			"modelName", modelName, "targetModelName", targetModelName, "ttft", ttft)
		return false
	}

	// Check for SLO violation (ttft exceeds threshold)
	if ttft > sloThreshold {
		inferenceGauges.WithLabelValues(modelName, targetModelName, TypeTTFTSLOViolation).Set(1)
		sloViolationCounter.WithLabelValues(modelName, targetModelName, TypeTTFT).Inc()
		log.FromContext(ctx).V(logutil.DEFAULT).Info("TTFT SLO violation detected",
			"modelName", modelName, "targetModelName", targetModelName, "ttft", ttft, "threshold", sloThreshold)
	} else {
		inferenceGauges.WithLabelValues(modelName, targetModelName, TypeTTFTSLOViolation).Set(0)
	}

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
	inferenceGauges.WithLabelValues(modelName, targetModelName, TypePredictedTTFT).Set(predicted_ttft)
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
	inferenceGauges.WithLabelValues(modelName, targetModelName, TypeTTFTPredictionDuration).Set(duration)
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

// RecordPromptCachedTokens records prompt cached tokens count.
func RecordPromptCachedTokens(modelName, targetModelName string, size int) {
	promptCachedTokens.WithLabelValues(modelName, targetModelName).Observe(float64(size))
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

// RecordSchedulerE2ELatency records the end-to-end scheduling latency.
func RecordSchedulerE2ELatency(duration time.Duration) {
	SchedulerE2ELatency.WithLabelValues().Observe(duration.Seconds())
}

// RecordSchedulerAttempt records a scheduling attempt with status.
func RecordSchedulerAttempt(err error) {
	if err != nil {
		SchedulerAttemptsTotal.WithLabelValues(SchedulerStatusFailure).Inc()
	} else {
		SchedulerAttemptsTotal.WithLabelValues(SchedulerStatusSuccess).Inc()
	}
}

const (
	SchedulerStatusSuccess = "success"
	SchedulerStatusFailure = "failure"
)

// RecordPluginProcessingLatency records the processing latency for a plugin.
func RecordPluginProcessingLatency(extensionPoint, pluginType, pluginName string, duration time.Duration) {
	PluginProcessingLatencies.WithLabelValues(extensionPoint, pluginType, pluginName).Observe(duration.Seconds())
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

func RecordInferenceExtensionInfo(commitSha, buildRef string) {
	InferenceExtensionInfo.WithLabelValues(commitSha, buildRef).Set(1)
}

// RecordFlowControlRequestQueueDuration records the duration a request spent in the Flow Control layer.
func RecordFlowControlRequestQueueDuration(fairnessID, priority, outcome string, duration time.Duration) {
	flowControlRequestQueueDuration.WithLabelValues(fairnessID, priority, outcome).Observe(duration.Seconds())
}

// IncFlowControlQueueSize increments the Flow Control queue size gauge.
func IncFlowControlQueueSize(fairnessID, priority string) {
	flowControlQueueSize.WithLabelValues(fairnessID, priority).Inc()
}

// DecFlowControlQueueSize decrements the Flow Control queue size gauge.
func DecFlowControlQueueSize(fairnessID, priority string) {
	flowControlQueueSize.WithLabelValues(fairnessID, priority).Dec()
}

// SetTTFTSLOThreshold sets the TTFT SLO threshold for a model.
// This allows dynamic threshold management and makes the threshold visible in metrics.
func SetTTFTSLOThreshold(modelName, targetModelName string, threshold float64) {
	inferenceGauges.WithLabelValues(modelName, targetModelName, TypeTTFTSLOThreshold).Set(threshold)
}

// SetTPOTSLOThreshold sets the TPOT SLO threshold for a model.
// This allows dynamic threshold management and makes the threshold visible in metrics.
func SetTPOTSLOThreshold(modelName, targetModelName string, threshold float64) {
	inferenceGauges.WithLabelValues(modelName, targetModelName, TypeTPOTSLOThreshold).Set(threshold)
}

// RecordInferenceModelRewriteDecision records the routing decision for InferenceModelRewrite.
func RecordInferenceModelRewriteDecision(modelRewriteName, modelName, targetModel string) {
	inferenceModelRewriteDecisionsTotal.WithLabelValues(modelRewriteName, modelName, targetModel).Inc()
}
