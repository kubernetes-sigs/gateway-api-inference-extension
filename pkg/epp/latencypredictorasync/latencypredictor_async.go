// Package latencypredictorasync provides a Go client for the Python-based
// latency prediction service with asynchronous batching and cached metrics.
package latencypredictorasync

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
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
	PythonURL     string
	// MaxSampleSize is the maximum number of training entries to send in each flush.
	// If the buffer contains more entries, they will be randomly sampled.
	MaxSampleSize int
	// FlushInterval determines how often to flush training & refresh metrics.
	FlushInterval time.Duration
}

func DefaultConfig() *Config {
	return &Config{
		PythonURL:     "http://localhost:8000",
		MaxSampleSize: 1000,
		FlushInterval: 1 * time.Second,
	}
}

func ConfigFromEnv() *Config {
	cfg := DefaultConfig()
	if url := os.Getenv("LATENCY_SERVER_URL"); url != "" {
		cfg.PythonURL = url
	}
	if sizeStr := os.Getenv("LATENCY_MAX_SAMPLE_SIZE"); sizeStr != "" {
		if size, err := strconv.Atoi(sizeStr); err == nil && size > 0 {
			cfg.MaxSampleSize = size
		}
	}
	if intervalStr := os.Getenv("LATENCY_FLUSH_INTERVAL_SEC"); intervalStr != "" {
		if sec, err := strconv.Atoi(intervalStr); err == nil && sec > 0 {
			cfg.FlushInterval = time.Duration(sec) * time.Second
		}
	}
	return cfg
}

// Predictor defines the interface for latency prediction and training.
type PredictorInterface interface {
    Predict(ctx context.Context, req PredictionRequest) (*PredictionResponse, error)
    AddTrainingDataBulk(entry []TrainingEntry) error
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
	rng           *rand.Rand

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
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
		done:       make(chan struct{}),
	}
	go p.backgroundLoop()
	return p
}

// Start is a no-op for API compatibility.
func (p *Predictor) Start(ctx context.Context) error {
	p.logger.Info("Latency predictor async client started.",
		"target_url", p.config.PythonURL,
		"max_sample_size", p.config.MaxSampleSize,
		"flush_interval", p.config.FlushInterval)
	return nil
}

// Stop stops background work, then does a final flush/refresh.
func (p *Predictor) Stop() {
	close(p.done)
	// final flush & refresh
	p.flushTraining()
	p.refreshMetrics()
}

// backgroundLoop runs flush & refresh at configured intervals.
func (p *Predictor) backgroundLoop() {
	ticker := time.NewTicker(p.config.FlushInterval)
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

// randomSample returns up to maxSize entries via partial Fisher-Yates shuffle.
func (p *Predictor) randomSample(entries []TrainingEntry, maxSize int) []TrainingEntry {
	if len(entries) <= maxSize {
		return entries
	}

	sample := make([]TrainingEntry, len(entries))
	copy(sample, entries)
	for i := 0; i < maxSize; i++ {
		j := p.rng.Intn(len(sample)-i) + i
		sample[i], sample[j] = sample[j], sample[i]
	}
	return sample[:maxSize]
}

// flushTraining sends buffered entries in one bulk POST, with error handling.
func (p *Predictor) flushTraining() {
	p.bufferMu.Lock()
	batch := p.pending
	p.pending = nil
	p.bufferMu.Unlock()

	if len(batch) == 0 {
		return
	}

	originalSize := len(batch)
	if len(batch) > p.config.MaxSampleSize {
		batch = p.randomSample(batch, p.config.MaxSampleSize)
		p.logger.V(1).Info("sampled training entries for flush",
			"original_size", originalSize,
			"sampled_size", len(batch),
			"max_sample_size", p.config.MaxSampleSize)
	}

	payload := BulkTrainingRequest{Entries: batch}
	data, err := json.Marshal(payload)
	if err != nil {
		p.logger.Error(err, "marshal bulk payload")
		return
	}

	url := p.config.PythonURL + "/add_training_data_bulk"
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, bytes.NewBuffer(data))
	if err != nil {
		p.logger.Error(err, "creating bulk POST request", "url", url)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		p.logger.Error(err, "bulk POST failed", "url", url)
		return
	}
	defer resp.Body.Close()

	io.Copy(io.Discard, resp.Body)
	if resp.StatusCode != http.StatusAccepted {
		p.logger.Error(fmt.Errorf("status %d", resp.StatusCode),
			"bulk POST returned non-202", "url", url)
	} else {
		if originalSize > len(batch) {
			p.logger.V(1).Info("flushed sampled training batch",
				"sent_count", len(batch),
				"original_count", originalSize,
				"sample_rate", float64(len(batch))/float64(originalSize))
		} else {
			p.logger.V(1).Info("flushed training batch", "count", len(batch))
		}
	}
}

// refreshMetrics GETs /metrics and caches parsed coefficients.
func (p *Predictor) refreshMetrics() {
	_, _ = p.GetMetrics(context.Background())
}

// Predict uses cached coefficients for a local prediction.
func (p *Predictor) Predict(ctx context.Context, req PredictionRequest) (*PredictionResponse, error) {
	p.metricsMu.RLock()
	mr := p.cachedMetrics
	p.metricsMu.RUnlock()

	if mr == nil || mr.Coefficients == nil {
		return nil, fmt.Errorf("no cached model coefficients available")
	}
	c := mr.Coefficients

	// linear combination
	ttft := c.TTFTIntercept +
		c.TTFTCoeffs["kv_cache_percentage"]*req.KVCachePercentage +
		c.TTFTCoeffs["input_token_length"]*float64(req.InputTokenLength) +
		c.TTFTCoeffs["num_request_waiting"]*float64(req.NumRequestWaiting) +
		c.TTFTCoeffs["num_request_running"]*float64(req.NumRequestRunning)

	tpot := c.TPOTIntercept +
		c.TPOTCoeffs["kv_cache_percentage"]*req.KVCachePercentage +
		c.TPOTCoeffs["num_request_waiting"]*float64(req.NumRequestWaiting) +
		c.TPOTCoeffs["num_request_running"]*float64(req.NumRequestRunning) +
		c.TPOTCoeffs["num_tokens_generated"]*float64(req.NumTokensGenerated) + 
		c.TPOTCoeffs["input_token_length"]*float64(req.InputTokenLength)

	return &PredictionResponse{
		TTFT:                 ttft,
		TPOT:                 tpot,
		TTFTUncertainty:      0,
		TPOTUncertainty:      0,
		TTFTPredictionBounds: [2]float64{ttft, ttft},
		TPOTPredictionBounds: [2]float64{tpot, tpot},
		PredictedAt:          time.Now(),
	}, nil
}

// GetMetrics fetches & parses metrics from the server.
func (p *Predictor) GetMetrics(ctx context.Context) (*MetricsResponse, error) {
	url := p.config.PythonURL + "/metrics"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
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

	rawMetricsBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read metrics response: %w", err)
	}

	metricsResponse := &MetricsResponse{RawMetrics: string(rawMetricsBytes)}
	coeffs, buckets, err := p.parsePrometheusMetrics(metricsResponse.RawMetrics)
	if err != nil {
		p.logger.V(1).Info("Failed to parse metrics, caching raw only", "error", err)
	} else {
		metricsResponse.Coefficients = coeffs
		metricsResponse.BucketCounts = buckets
	}

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

func (p *Predictor) parseMetricLine(line string, coefficients *ModelCoefficients, bucketCounts *BucketCounts) error {
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return fmt.Errorf("invalid metric line format: %s", line)
	}

	// Handle both formats:
	// "metric_name value" (2 parts)
	// "metric_name {} value" (3 parts)
	var metricName, valueStr string
	if len(parts) == 2 {
		metricName = parts[0]
		valueStr = parts[1]
	} else if len(parts) == 3 && parts[1] == "{}" {
		metricName = parts[0]
		valueStr = parts[2]
	} else {
		return fmt.Errorf("invalid metric line format: %s", line)
	}

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

	// Optional: Add cases for the other metrics if you want to capture them
	case metricName == "ttft_test_data_count":
		// Store if needed - you could add these to your structs if useful
	case metricName == "tpot_test_data_count":
		// Store if needed
	case metricName == "ttft_train_data_count":
		// Store if needed
	case metricName == "tpot_train_data_count":
		// Store if needed
	case metricName == "test_train_ratio":
		// Store if needed
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

func (p *Predictor) GetModelCoefficients(ctx context.Context) (*ModelCoefficients, error) {
    metrics, err := p.GetMetrics(ctx)
    if err != nil {
        return nil, err
    }
    return metrics.Coefficients, nil
}

func (p *Predictor) GetBucketCounts(ctx context.Context) (*BucketCounts, error) {
    metrics, err := p.GetMetrics(ctx)
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