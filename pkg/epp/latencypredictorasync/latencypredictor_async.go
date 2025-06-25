// Package latencypredictorasync provides a Go client for the Python-based
// latency prediction service with asynchronous batching and cached metrics.
package latencypredictorasync

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
)

// --- Configuration ---

type Config struct {
	// PythonURL is the base URL of the Python latency predictor server.
	PythonURL string
}

func DefaultConfig() *Config {
	return &Config{PythonURL: "http://localhost:8000"}
}

func ConfigFromEnv() *Config {
	cfg := DefaultConfig()
	if url := os.Getenv("LATENCY_SERVER_URL"); url != "" {
		cfg.PythonURL = url
	}
	return cfg
}

// --- Data Models ---

type TrainingEntry struct {
	KVCachePercentage  float64   `json:"kv_cache_percentage"`
	InputTokenLength   int       `json:"input_token_length"`
	NumRequestWaiting  int       `json:"num_request_waiting"`
	NumRequestRunning  int       `json:"num_request_running"`
	NumTokensGenerated int       `json:"num_tokens_generated"`
	ActualTTFT         float64   `json:"actual_ttft_ms"`
	ActualTPOT         float64   `json:"actual_tpot_ms"`
	Timestamp          time.Time `json:"timestamp"`
}

type BulkTrainingRequest struct {
	Entries []TrainingEntry `json:"entries"`
}

type PredictionRequest struct {
	KVCachePercentage  float64 `json:"kv_cache_percentage"`
	InputTokenLength   int     `json:"input_token_length"`
	NumRequestWaiting  int     `json:"num_request_waiting"`
	NumRequestRunning  int     `json:"num_request_running"`
	NumTokensGenerated int     `json:"num_tokens_generated"`
}

type PredictionResponse struct {
	TTFT                 float64    `json:"ttft_ms"`
	TPOT                 float64    `json:"tpot_ms"`
	TTFTUncertainty      float64    `json:"ttft_uncertainty"`
	TPOTUncertainty      float64    `json:"tpot_uncertainty"`
	TTFTPredictionBounds [2]float64 `json:"ttft_prediction_bounds"`
	TPOTPredictionBounds [2]float64 `json:"tpot_prediction_bounds"`
	PredictedAt          time.Time  `json:"predicted_at"`
}

type ModelCoefficients struct {
	TTFTIntercept float64            `json:"ttft_intercept"`
	TTFTCoeffs    map[string]float64 `json:"ttft_coefficients"`
	TPOTIntercept float64            `json:"tpot_intercept"`
	TPOTCoeffs    map[string]float64 `json:"tpot_coefficients"`
}

type BucketCounts struct {
	TTFTBuckets map[int]int `json:"ttft_buckets"`
	TPOTBuckets map[int]int `json:"tpot_buckets"`
}

type MetricsResponse struct {
	Coefficients *ModelCoefficients `json:"coefficients"`
	BucketCounts *BucketCounts      `json:"bucket_counts"`
	RawMetrics   string             `json:"raw_metrics"`
}

// --- Predictor Client ---

type Predictor struct {
	config        *Config
	httpClient    *http.Client
	logger        logr.Logger

	// cached metrics
	metricsMu     sync.RWMutex
	cachedMetrics *MetricsResponse

	// buffer for pending training
	bufferMu      sync.Mutex
	pending       []TrainingEntry

	// shutdown signal
	done          chan struct{}
}

func New(config *Config, logger logr.Logger) *Predictor {
	if config == nil {
		config = ConfigFromEnv()
	}
	p := &Predictor{
		config:     config,
		httpClient: &http.Client{Timeout: 10 * time.Second},
		logger:     logger.WithName("latency-predictor-client"),
		done:       make(chan struct{}),
	}
	go p.backgroundLoop()
	return p
}

// Start is a no-op for the client but is included for API compatibility.
func (p *Predictor) Start() error {
	p.logger.Info("Latency predictor async client started.", "target_url", p.config.PythonURL)
	return nil
}

// Stop flushes remaining data and stops background work.
func (p *Predictor) Stop() {
	// final flush
	p.flushTraining()
	p.refreshMetrics()
	close(p.done)
}

// backgroundLoop runs flush & refresh once per second.
func (p *Predictor) backgroundLoop() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			p.flushTraining()
			p.refreshMetrics()
		case <-p.done:
			return
		}
	}
}

// AddTrainingDataBulk buffers entries for periodic flush.
func (p *Predictor) AddTrainingDataBulk(entries []TrainingEntry) error {
	p.bufferMu.Lock()
	p.pending = append(p.pending, entries...)
	p.bufferMu.Unlock()
	return nil
}

// flushTraining sends buffered entries in one bulk POST.
func (p *Predictor) flushTraining() {
	p.bufferMu.Lock()
	batch := p.pending
	p.pending = nil
	p.bufferMu.Unlock()

	if len(batch) == 0 {
		return
	}

	payload := BulkTrainingRequest{Entries: batch}
	data, err := json.Marshal(payload)
	if err != nil {
		p.logger.Error(err, "marshal bulk payload")
		return
	}

	url := p.config.PythonURL + "/add_training_data_bulk"
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, url, bytes.NewBuffer(data))
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		p.logger.Error(err, "bulk POST failed", "url", url)
		return
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		p.logger.Error(fmt.Errorf("status %d", resp.StatusCode),
			"bulk POST returned non-202", "url", url)
	} else {
		p.logger.V(1).Info("flushed training batch", "count", len(batch))
	}
}

// refreshMetrics GETs /metrics and caches parsed coefficients.
func (p *Predictor) refreshMetrics() {
	url := p.config.PythonURL + "/metrics"
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, url, nil)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		p.logger.Error(err, "metrics GET failed", "url", url)
		return
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		p.logger.Error(fmt.Errorf("status %d", resp.StatusCode),
			"metrics GET returned non-200", "url", url)
		return
	}

	coeffs, buckets, err := p.parsePrometheusMetrics(string(data))
	mr := &MetricsResponse{RawMetrics: string(data)}
	if err == nil {
		mr.Coefficients = coeffs
		mr.BucketCounts  = buckets
	} else {
		p.logger.V(2).Info("failed to parse metrics, caching raw only", "err", err)
	}

	p.metricsMu.Lock()
	p.cachedMetrics = mr
	p.metricsMu.Unlock()
	p.logger.V(1).Info("metrics refreshed")
}

// Predict uses cached coefficients for a local prediction.
func (p *Predictor) Predict(req PredictionRequest) (*PredictionResponse, error) {
	p.metricsMu.RLock()
	mr := p.cachedMetrics
	p.metricsMu.RUnlock()

	if mr == nil || mr.Coefficients == nil {
		return nil, fmt.Errorf("no cached model coefficients available")
	}
	c := mr.Coefficients

	// linear combination
	ttft := c.TTFTIntercept
	ttft += c.TTFTCoeffs["kv_cache_percentage"]  * req.KVCachePercentage
	ttft += c.TTFTCoeffs["input_token_length"]   * float64(req.InputTokenLength)
	ttft += c.TTFTCoeffs["num_request_waiting"]  * float64(req.NumRequestWaiting)
	ttft += c.TTFTCoeffs["num_request_running"]  * float64(req.NumRequestRunning)

	tpot := c.TPOTIntercept
	tpot += c.TPOTCoeffs["kv_cache_percentage"]  * req.KVCachePercentage
	tpot += c.TPOTCoeffs["num_request_waiting"]  * float64(req.NumRequestWaiting)
	tpot += c.TPOTCoeffs["num_request_running"]  * float64(req.NumRequestRunning)
	tpot += c.TPOTCoeffs["num_tokens_generated"]* float64(req.NumTokensGenerated)

	return &PredictionResponse{
		TTFT:    ttft,
		TPOT:    tpot,
		TTFTUncertainty:      0,
		TPOTUncertainty:      0,
		TTFTPredictionBounds: [2]float64{ttft, ttft},
		TPOTPredictionBounds: [2]float64{tpot, tpot},
		PredictedAt:          time.Now(),
	}, nil
}


// GetMetrics fetches metrics from the server and stores them in memory.
func (p *Predictor) GetMetrics() (*MetricsResponse, error) {
	url := p.config.PythonURL + "/metrics"
	req, err := http.NewRequestWithContext(context.Background(), "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create metrics request: %w", err)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call Python /metrics endpoint: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server returned non-200 status: %d %s, body: %s", resp.StatusCode, resp.Status, string(body))
	}

	rawMetrics, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read metrics response: %w", err)
	}

	metricsResponse := &MetricsResponse{
		RawMetrics: string(rawMetrics),
	}

	coeffs, buckets, err := p.parsePrometheusMetrics(metricsResponse.RawMetrics)
	if err != nil {
		p.logger.V(1).Info("Failed to parse metrics, caching raw only", "error", err)
	} else {
		metricsResponse.Coefficients = coeffs
		metricsResponse.BucketCounts = buckets
	}

	// cache it
	p.metricsMu.Lock()
	p.cachedMetrics = metricsResponse
	p.metricsMu.Unlock()

	p.logger.V(1).Info("Successfully retrieved and cached metrics.")
	return metricsResponse, nil
}


// parsePrometheusMetrics parses the Prometheus-format metrics into structured data.
func (p *Predictor) parsePrometheusMetrics(rawMetrics string) (*ModelCoefficients, *BucketCounts, error) {
	lines := strings.Split(rawMetrics, "\n")
	
	coefficients := &ModelCoefficients{
		TTFTCoeffs: make(map[string]float64),
		TPOTCoeffs: make(map[string]float64),
	}
	
	bucketCounts := &BucketCounts{
		TTFTBuckets: make(map[int]int),
		TPOTBuckets: make(map[int]int),
	}

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Parse metric lines
		if err := p.parseMetricLine(line, coefficients, bucketCounts); err != nil {
			p.logger.V(2).Info("Failed to parse metric line", "line", line, "error", err)
			// Continue parsing other lines instead of failing completely
		}
	}

	return coefficients, bucketCounts, nil
}

// parseMetricLine parses a single Prometheus metric line.
func (p *Predictor) parseMetricLine(line string, coefficients *ModelCoefficients, bucketCounts *BucketCounts) error {
	parts := strings.Fields(line)
	if len(parts) != 2 {
		return fmt.Errorf("invalid metric line format: %s", line)
	}

	metricName := parts[0]
	valueStr := parts[1]

	value, err := strconv.ParseFloat(valueStr, 64)
	if err != nil {
		return fmt.Errorf("failed to parse metric value '%s': %w", valueStr, err)
	}

	// Parse different metric types
	switch {
	case metricName == "ttft_intercept":
		coefficients.TTFTIntercept = value

	case metricName == "tpot_intercept":
		coefficients.TPOTIntercept = value

	case strings.HasPrefix(metricName, "ttft_coef{feature=\""):
		feature := p.extractFeatureName(metricName)
		if feature != "" {
			coefficients.TTFTCoeffs[feature] = value
		}

	case strings.HasPrefix(metricName, "tpot_coef{feature=\""):
		feature := p.extractFeatureName(metricName)
		if feature != "" {
			coefficients.TPOTCoeffs[feature] = value
		}

	case strings.HasPrefix(metricName, "ttft_bucket_count{bucket=\""):
		bucket := p.extractBucketNumber(metricName)
		if bucket >= 0 {
			bucketCounts.TTFTBuckets[bucket] = int(value)
		}

	case strings.HasPrefix(metricName, "tpot_bucket_count{bucket=\""):
		bucket := p.extractBucketNumber(metricName)
		if bucket >= 0 {
			bucketCounts.TPOTBuckets[bucket] = int(value)
		}
	}

	return nil
}

// extractFeatureName extracts the feature name from a coefficient metric.
// Example: ttft_coef{feature="kv_cache_percentage"} -> "kv_cache_percentage"
func (p *Predictor) extractFeatureName(metricName string) string {
	start := strings.Index(metricName, "feature=\"")
	if start == -1 {
		return ""
	}
	start += len("feature=\"")
	end := strings.Index(metricName[start:], "\"")
	if end == -1 {
		return ""
	}
	return metricName[start : start+end]
}

// extractBucketNumber extracts the bucket number from a bucket count metric.
// Example: ttft_bucket_count{bucket="5"} -> 5
func (p *Predictor) extractBucketNumber(metricName string) int {
	start := strings.Index(metricName, "bucket=\"")
	if start == -1 {
		return -1
	}
	start += len("bucket=\"")
	end := strings.Index(metricName[start:], "\"")
	if end == -1 {
		return -1
	}
	bucketStr := metricName[start : start+end]
	bucket, err := strconv.Atoi(bucketStr)
	if err != nil {
		return -1
	}
	return bucket
}

// GetModelCoefficients is a convenience method that returns just the model coefficients.
func (p *Predictor) GetModelCoefficients() (*ModelCoefficients, error) {
	metrics, err := p.GetMetrics()
	if err != nil {
		return nil, err
	}
	return metrics.Coefficients, nil
}

// GetBucketCounts is a convenience method that returns just the bucket counts.
func (p *Predictor) GetBucketCounts() (*BucketCounts, error) {
	metrics, err := p.GetMetrics()
	if err != nil {
		return nil, err
	}
	return metrics.BucketCounts, nil
}


// GetCachedMetrics returns the last metrics fetched by GetMetrics (if any).
// The bool indicates whether we have a cached value.
func (p *Predictor) GetCachedMetrics() (*MetricsResponse, bool) {
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()
	if p.cachedMetrics == nil {
		return nil, false
	}
	return p.cachedMetrics, true
}