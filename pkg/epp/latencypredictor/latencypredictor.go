// Package latencypredictor provides a Go client for the Python-based
// latency prediction service.
package latencypredictor

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

// Config holds the configuration for the predictor client.
type Config struct {
	// PythonURL is the base URL of the Python latency predictor server.
	PythonURL string
}

// DefaultConfig returns a default configuration pointing to localhost.
func DefaultConfig() *Config {
	return &Config{
		PythonURL: "http://localhost:8000",
	}
}

// ConfigFromEnv returns a configuration, overriding defaults with the
// LATENCY_SERVER_URL environment variable if it is set.
func ConfigFromEnv() *Config {
	cfg := DefaultConfig()
	if url := os.Getenv("LATENCY_SERVER_URL"); url != "" {
		cfg.PythonURL = url
	}
	return cfg
}

// --- Data Models ---
// These structs correspond to the Pydantic models in the Python server.
// The `json` tags are crucial for correct serialization and deserialization.

// TrainingEntry captures a single labeled sample to be sent to the server.
type TrainingEntry struct {
	KVCachePercentage float64   `json:"kv_cache_percentage"`
	InputTokenLength  int       `json:"input_token_length"`
	NumRequestWaiting int       `json:"num_request_waiting"`
	NumRequestRunning int       `json:"num_request_running"`
	NumTokensGenerated int       `json:"num_tokens_generated"`
	ActualTTFT        float64   `json:"actual_ttft_ms"`
	ActualTPOT        float64   `json:"actual_tpot_ms"`
	Timestamp         time.Time `json:"timestamp"`
}

type BulkTrainingRequest struct {
	Entries []TrainingEntry `json:"entries"`
}

// PredictionRequest defines the input features for a prediction request.
type PredictionRequest struct {
	KVCachePercentage float64 `json:"kv_cache_percentage"`
	InputTokenLength  int     `json:"input_token_length"`
	NumRequestWaiting int     `json:"num_request_waiting"`
	NumRequestRunning int     `json:"num_request_running"`
	NumTokensGenerated int     `json:"num_tokens_generated"`
}

// PredictionResponse contains the latency predictions and metadata from the server.
type PredictionResponse struct {
	TTFT                 float64   `json:"ttft_ms"`
	TPOT                 float64   `json:"tpot_ms"`
	TTFTUncertainty      float64   `json:"ttft_uncertainty"`
	TPOTUncertainty      float64   `json:"tpot_uncertainty"`
	TTFTPredictionBounds [2]float64 `json:"ttft_prediction_bounds"`
	TPOTPredictionBounds [2]float64 `json:"tpot_prediction_bounds"`
	PredictedAt          time.Time `json:"predicted_at"`
}

// ModelCoefficients represents the model coefficients for TTFT and TPOT models.
type ModelCoefficients struct {
	TTFTIntercept float64            `json:"ttft_intercept"`
	TTFTCoeffs    map[string]float64 `json:"ttft_coefficients"`
	TPOTIntercept float64            `json:"tpot_intercept"`
	TPOTCoeffs    map[string]float64 `json:"tpot_coefficients"`
}

// BucketCounts represents the training data distribution across buckets.
type BucketCounts struct {
	TTFTBuckets map[int]int `json:"ttft_buckets"`
	TPOTBuckets map[int]int `json:"tpot_buckets"`
}

// MetricsResponse contains the parsed metrics from the server.
type MetricsResponse struct {
	Coefficients *ModelCoefficients `json:"coefficients"`
	BucketCounts *BucketCounts      `json:"bucket_counts"`
	RawMetrics   string             `json:"raw_metrics"`
}

// --- Predictor Client ---

// Predictor is the client that interacts with the Python latency prediction service.
type Predictor struct {
	config     *Config
	httpClient *http.Client
	logger     logr.Logger

	// new fields for in‐memory caching
	metricsMu     sync.RWMutex
	cachedMetrics *MetricsResponse
}

// New creates a new client for the latency predictor service.
func New(config *Config, logger logr.Logger) *Predictor {
	if config == nil {
		config = ConfigFromEnv()
	}
	return &Predictor{
		config: config,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
		logger: logger.WithName("latency-predictor-client"),
	}
}

// Start is a no-op for the client but is included for API compatibility.
func (p *Predictor) Start() error {
	p.logger.Info("Latency predictor client started.", "target_url", p.config.PythonURL)
	return nil
}

// Stop is a no-op for the client but is included for API compatibility.
func (p *Predictor) Stop() error {
	p.logger.Info("Latency predictor client stopped.")
	return nil
}

// AddTrainingDataBulk sends one or more training entries in a single POST.
func (p *Predictor) AddTrainingDataBulk(entries []TrainingEntry) error {
	payload := BulkTrainingRequest{Entries: entries}
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal bulk training payload: %w", err)
	}

	url := p.config.PythonURL + "/add_training_data_bulk"
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("create bulk request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("POST %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("bulk endpoint returned %d: %s", resp.StatusCode, string(body))
	}

	p.logger.V(1).Info("Successfully added bulk training data", "count", len(entries))
	return nil
}

// Predict sends a request for a latency prediction to the Python server.
func (p *Predictor) Predict(request PredictionRequest) (*PredictionResponse, error) {
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal prediction request: %w", err)
	}

	url := p.config.PythonURL + "/predict"
	req, err := http.NewRequestWithContext(context.Background(), "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call Python /predict endpoint: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server returned non-200 status: %d %s, body: %s", resp.StatusCode, resp.Status, string(body))
	}

	var predictionResp PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&predictionResp); err != nil {
		return nil, fmt.Errorf("failed to decode prediction response: %w", err)
	}

	p.logger.V(1).Info("Successfully received prediction.")
	return &predictionResp, nil
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