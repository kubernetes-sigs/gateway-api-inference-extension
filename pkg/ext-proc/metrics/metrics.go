package metrics

import (
	"sync"
	"time"

	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	LLMServiceModelComponent = "llmservice_model"
)

var (
	requestCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      LLMServiceModelComponent,
			Name:           "request_total",
			Help:           "Counter of LLM service requests broken out for each model and target model.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"llmservice_name", "model_name", "target_model_name"},
	)

	requestLatencies = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Subsystem: LLMServiceModelComponent,
			Name:      "request_duration_seconds",
			Help:      "LLM service response latency distribution in seconds for each model and target model.",
			Buckets: []float64{0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3,
				4, 5, 6, 8, 10, 15, 20, 30, 45, 60, 120, 180, 240, 300, 360, 480, 600, 900, 1200, 1800, 2700, 3600},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"llmservice_name", "model_name", "target_model_name"},
	)

	requestSizes = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Subsystem: LLMServiceModelComponent,
			Name:      "request_sizes",
			Help:      "LLM service requests size distribution in bytes for each model and target model.",
			// Use buckets ranging from 1000 bytes (1KB) to 10^9 bytes (1GB).
			Buckets:        compbasemetrics.ExponentialBuckets(1000, 10.0, 7),
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"llmservice_name", "model_name", "target_model_name"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(requestCounter)
		legacyregistry.MustRegister(requestLatencies)
		legacyregistry.MustRegister(requestSizes)
	})
}

// MonitorRequest handles monitoring requests.
func MonitorRequest(llmserviceName, modelName, targetModelName string, reqSize int, elapsed time.Duration) {
	elapsedSeconds := elapsed.Seconds()
	requestCounter.WithLabelValues(llmserviceName, modelName, targetModelName).Inc()
	requestLatencies.WithLabelValues(llmserviceName, modelName, targetModelName).Observe(elapsedSeconds)
	requestSizes.WithLabelValues(llmserviceName, modelName, targetModelName).Observe(float64(reqSize))
}
