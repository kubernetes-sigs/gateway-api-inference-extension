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
	// TrainingURL is the base URL of the Python training server.
	TrainingURL string
	// PredictionURLs is a list of prediction server URLs for load balancing.
	PredictionURLs []string
	// MaxSampleSize is the maximum number of training entries to send in each flush.
	// If the buffer contains more entries, they will be randomly sampled.
	MaxSampleSize int
	// FlushInterval determines how often to flush training & refresh metrics.
	FlushInterval time.Duration
	// UseNativeXGBoost when true, attempts to use local XGBoost models for prediction.
	// When false, falls back to HTTP calls to the Python server for XGBoost predictions.
	UseNativeXGBoost bool
	// HTTPTimeout is the timeout for HTTP requests to the Python server.
	HTTPTimeout time.Duration

	MetricsRefreshInterval time.Duration
}

func DefaultConfig() *Config {
	return &Config{
		TrainingURL:            "http://localhost:8000",
		PredictionURLs:         []string{"http://localhost:8001"},
		MaxSampleSize:          1000,
		FlushInterval:          1 * time.Second,
		MetricsRefreshInterval: 60 * time.Second,
		UseNativeXGBoost:       true,
		HTTPTimeout:            10 * time.Second,
	}
}

func ConfigFromEnv() *Config {
	cfg := DefaultConfig()

	// Training URL (single URL for training data submission)
	if url := os.Getenv("TRAINING_SERVER_URL"); url != "" {
		cfg.TrainingURL = url
	}

	// Prediction URLs (comma-separated list for load balancing)
	if urls := os.Getenv("PREDICTION_SERVER_URL"); urls != "" {
		predictionURLs := strings.Split(urls, ",")
		for i, url := range predictionURLs {
			predictionURLs[i] = strings.TrimSpace(url)
		}
		cfg.PredictionURLs = predictionURLs
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
	if nativeStr := os.Getenv("LATENCY_USE_NATIVE_XGBOOST"); nativeStr != "" {
		cfg.UseNativeXGBoost = strings.ToLower(nativeStr) == "true"
	}
	if timeoutStr := os.Getenv("LATENCY_HTTP_TIMEOUT_SEC"); timeoutStr != "" {
		if sec, err := strconv.Atoi(timeoutStr); err == nil && sec > 0 {
			cfg.HTTPTimeout = time.Duration(sec) * time.Second
		}
	}

	if s := os.Getenv("LATENCY_METRICS_INTERVAL_SEC"); s != "" {
		if sec, err := strconv.Atoi(s); err == nil && sec > 0 {
			cfg.MetricsRefreshInterval = time.Duration(sec) * time.Second
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
	PrefixCacheScore   float64   `json:"prefix_cache_score"` // Added prefix cache score
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
	PrefixCacheScore   float64 `json:"prefix_cache_score"` // Added prefix cache score
}

type PredictionResponse struct {
	TTFT                 float64    `json:"ttft_ms"`
	TPOT                 float64    `json:"tpot_ms"`
	TTFTUncertainty      float64    `json:"ttft_uncertainty"`
	TPOTUncertainty      float64    `json:"tpot_uncertainty"`
	TTFTPredictionBounds [2]float64 `json:"ttft_prediction_bounds"`
	TPOTPredictionBounds [2]float64 `json:"tpot_prediction_bounds"`
	PredictedAt          time.Time  `json:"predicted_at"`
	ModelType            string     `json:"model_type"`
	Quantile             float64    `json:"quantile"`        // Add this field
	LastModelLoad        *time.Time `json:"last_model_load"` // Add this field
}

type ModelCoefficients struct {
	TTFTIntercept float64            `json:"ttft_intercept"`
	TTFTCoeffs    map[string]float64 `json:"ttft_coefficients"`
	TPOTIntercept float64            `json:"tpot_intercept"`
	TPOTCoeffs    map[string]float64 `json:"tpot_coefficients"`
}

type XGBoostTrees struct {
	TTFTTrees []interface{} `json:"ttft_trees"`
	TPOTTrees []interface{} `json:"tpot_trees"`
}

type BucketCounts struct {
	TTFTBuckets map[int]int `json:"ttft_buckets"`
	TPOTBuckets map[int]int `json:"tpot_buckets"`
}

type ModelInfo struct {
	ModelType   string          `json:"model_type"`
	ModelStatus map[string]bool `json:"model_status"`
}

type MetricsResponse struct {
	ModelType    string             `json:"model_type"`
	Coefficients *ModelCoefficients `json:"coefficients"`
	XGBoostTrees *XGBoostTrees      `json:"xgboost_trees"`
	BucketCounts *BucketCounts      `json:"bucket_counts"`
	RawMetrics   string             `json:"raw_metrics"`
}

// --- Predictor Client ---

type Predictor struct {
	config     *Config
	httpClient *http.Client
	logger     logr.Logger
	rng        *rand.Rand

	metricsMu     sync.RWMutex
	cachedMetrics *MetricsResponse
	modelInfo     *ModelInfo

	xgboostMu sync.RWMutex

	bufferMu sync.Mutex
	pending  []TrainingEntry

	wg   sync.WaitGroup
	done chan struct{}
}

func New(config *Config, logger logr.Logger) *Predictor {
	if config == nil {
		config = ConfigFromEnv()
	}
	p := &Predictor{
		config:     config,
		httpClient: &http.Client{Timeout: config.HTTPTimeout},
		logger:     logger.WithName("latency-predictor-client"),
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
		done:       make(chan struct{}),
	}
	p.wg.Add(1)
	go p.backgroundLoop()
	return p
}

// getRandomPredictionURL returns a randomly selected prediction URL for load balancing
func (p *Predictor) getRandomPredictionURL() string {
	if len(p.config.PredictionURLs) == 0 {
		return p.config.TrainingURL // Fallback to training URL
	}
	if len(p.config.PredictionURLs) == 1 {
		return p.config.PredictionURLs[0]
	}
	index := p.rng.Intn(len(p.config.PredictionURLs))
	return p.config.PredictionURLs[index]
}

// Start is a no-op for API compatibility.
func (p *Predictor) Start(ctx context.Context) error {
	// Get initial model info
	if err := p.refreshModelInfo(ctx); err != nil {
		p.logger.Error(err, "Failed to get initial model info")
	}

	p.logger.Info("Latency predictor async client started.",
		"training_url", p.config.TrainingURL,
		"prediction_urls", p.config.PredictionURLs,
		"max_sample_size", p.config.MaxSampleSize,
		"flush_interval", p.config.FlushInterval,
		"use_native_xgboost", p.config.UseNativeXGBoost)
	return nil
}

// Stop stops background work, then does a final flush/refresh.
func (p *Predictor) Stop() {
	close(p.done)
	p.wg.Wait() // Wait for the background loop to finish
	// final flush & refresh
	p.flushTraining()
	p.refreshMetrics()
	p.logger.Info("Latency predictor async client stopped.")
}

// backgroundLoop runs flush & refresh at configured intervals.
func (p *Predictor) backgroundLoop() {
	defer p.wg.Done()
	flushTicker := time.NewTicker(p.config.FlushInterval)
	metricsTicker := time.NewTicker(p.config.MetricsRefreshInterval)
	defer flushTicker.Stop()
	defer metricsTicker.Stop()

	for {
		select {
		case <-flushTicker.C:
			p.flushTraining()
		case <-metricsTicker.C:
			p.refreshMetrics()
		case <-p.done:
			return
		}
	}
}

// refreshModelInfo gets current model type and readiness info from training server
func (p *Predictor) refreshModelInfo(ctx context.Context) error {
	url := p.config.TrainingURL + "/model/download/info"
	p.logger.V(1).Info("Fetching model info", "url", url)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create model info request: %w", err)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call /model/download/info endpoint: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server %s returned non-200 status: %d %s, body: %s", url, resp.StatusCode, resp.Status, string(body))
	}

	var modelInfo ModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&modelInfo); err != nil {
		return fmt.Errorf("failed to decode model info response: %w", err)
	}

	p.metricsMu.Lock()
	p.modelInfo = &modelInfo
	p.metricsMu.Unlock()

	p.logger.V(1).Info("Retrieved model info", "model_type", modelInfo.ModelType, "model_status", modelInfo.ModelStatus)
	return nil
}

// getXGBoostTrees fetches tree JSON from the training server
func (p *Predictor) getXGBoostTrees(ctx context.Context) (*XGBoostTrees, error) {
	trees := &XGBoostTrees{}

	// Fetch TTFT trees from training server
	ttftURL := p.config.TrainingURL + "/model/ttft/xgb/json"
	ttftReq, err := http.NewRequestWithContext(ctx, http.MethodGet, ttftURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create TTFT trees request: %w", err)
	}

	ttftResp, err := p.httpClient.Do(ttftReq)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch TTFT trees: %w", err)
	}
	defer ttftResp.Body.Close()

	if ttftResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(ttftResp.Body)
		return nil, fmt.Errorf("TTFT trees request failed: %d %s, body: %s", ttftResp.StatusCode, ttftResp.Status, string(body))
	}

	if err := json.NewDecoder(ttftResp.Body).Decode(&trees.TTFTTrees); err != nil {
		return nil, fmt.Errorf("failed to decode TTFT trees: %w", err)
	}

	// Fetch TPOT trees from training server
	tpotURL := p.config.TrainingURL + "/model/tpot/xgb/json"
	tpotReq, err := http.NewRequestWithContext(ctx, http.MethodGet, tpotURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create TPOT trees request: %w", err)
	}

	tpotResp, err := p.httpClient.Do(tpotReq)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch TPOT trees: %w", err)
	}
	defer tpotResp.Body.Close()

	if tpotResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(tpotResp.Body)
		return nil, fmt.Errorf("TPOT trees request failed: %d %s, body: %s", tpotResp.StatusCode, tpotResp.Status, string(body))
	}

	if err := json.NewDecoder(tpotResp.Body).Decode(&trees.TPOTTrees); err != nil {
		return nil, fmt.Errorf("failed to decode TPOT trees: %w", err)
	}

	return trees, nil
}

// AddTrainingDataBulk buffers entries for periodic flush.
func (p *Predictor) AddTrainingDataBulk(entries []TrainingEntry) error {
	p.bufferMu.Lock()
	p.pending = append(p.pending, entries...)
	p.bufferMu.Unlock()
	return nil
}

// randomSample returns up to maxSize entries via stratified sampling to preserve
// the ratio of TTFT entries (ActualTTFT > 0) and TPOT entries (ActualTPOT > 0).
func (p *Predictor) randomSample(entries []TrainingEntry, maxSize int) []TrainingEntry {
	if len(entries) <= maxSize {
		return entries
	}

	// Separate entries into three groups
	var ttftEntries []TrainingEntry
	var tpotEntries []TrainingEntry
	var otherEntries []TrainingEntry

	for _, entry := range entries {
		hasTTFT := entry.ActualTTFT > 0
		hasTPOT := entry.ActualTPOT > 0

		if hasTTFT && hasTPOT {
			// Entry has both - we'll categorize it as TTFT for simplicity
			ttftEntries = append(ttftEntries, entry)
		} else if hasTTFT {
			ttftEntries = append(ttftEntries, entry)
		} else if hasTPOT {
			tpotEntries = append(tpotEntries, entry)
		} else {
			otherEntries = append(otherEntries, entry)
		}
	}

	totalEntries := len(entries)
	if totalEntries == 0 {
		return entries
	}

	// Calculate proportional sample sizes
	ttftSampleSize := int(float64(len(ttftEntries)) / float64(totalEntries) * float64(maxSize))
	tpotSampleSize := int(float64(len(tpotEntries)) / float64(totalEntries) * float64(maxSize))
	otherSampleSize := int(float64(len(otherEntries)) / float64(totalEntries) * float64(maxSize))

	// Adjust for rounding errors to ensure we reach exactly maxSize
	totalSampled := ttftSampleSize + tpotSampleSize + otherSampleSize
	if totalSampled < maxSize {
		remaining := maxSize - totalSampled
		// Distribute remaining samples proportionally to the largest groups
		if len(ttftEntries) >= len(tpotEntries) && len(ttftEntries) >= len(otherEntries) {
			ttftSampleSize += remaining
		} else if len(tpotEntries) >= len(otherEntries) {
			tpotSampleSize += remaining
		} else {
			otherSampleSize += remaining
		}
	} else if totalSampled > maxSize {
		// Reduce from the largest group
		excess := totalSampled - maxSize
		if ttftSampleSize >= tpotSampleSize && ttftSampleSize >= otherSampleSize {
			ttftSampleSize -= excess
		} else if tpotSampleSize >= otherSampleSize {
			tpotSampleSize -= excess
		} else {
			otherSampleSize -= excess
		}
	}

	var result []TrainingEntry

	// Sample from each group
	if ttftSampleSize > 0 && len(ttftEntries) > 0 {
		ttftSample := p.sampleFromSlice(ttftEntries, min(ttftSampleSize, len(ttftEntries)))
		result = append(result, ttftSample...)
	}

	if tpotSampleSize > 0 && len(tpotEntries) > 0 {
		tpotSample := p.sampleFromSlice(tpotEntries, min(tpotSampleSize, len(tpotEntries)))
		result = append(result, tpotSample...)
	}

	if otherSampleSize > 0 && len(otherEntries) > 0 {
		otherSample := p.sampleFromSlice(otherEntries, min(otherSampleSize, len(otherEntries)))
		result = append(result, otherSample...)
	}

	return result
}

// Helper function to sample from a slice
func (p *Predictor) sampleFromSlice(entries []TrainingEntry, sampleSize int) []TrainingEntry {
	if len(entries) <= sampleSize {
		return entries
	}

	// Create a copy and shuffle
	sample := make([]TrainingEntry, len(entries))
	copy(sample, entries)
	p.rng.Shuffle(len(sample), func(i, j int) {
		sample[i], sample[j] = sample[j], sample[i]
	})

	return sample[:sampleSize]
}

// Helper function to get minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// flushTraining sends buffered entries to training server in one bulk POST, with error handling.
func (p *Predictor) flushTraining() {
	p.bufferMu.Lock()
	if len(p.pending) == 0 {
		p.bufferMu.Unlock()
		return
	}
	batch := p.pending
	p.pending = nil
	p.bufferMu.Unlock()

	originalSize := len(batch)
	if originalSize > p.config.MaxSampleSize {
		batch = p.randomSample(batch, p.config.MaxSampleSize)
		p.logger.V(1).Info("Sampled training entries for flush",
			"original_size", originalSize,
			"sampled_size", len(batch))
	}

	payload := BulkTrainingRequest{Entries: batch}
	data, err := json.Marshal(payload)
	if err != nil {
		p.logger.Error(err, "Failed to marshal bulk payload")
		return // Cannot send if marshalling fails
	}

	// Send training data to training server
	url := p.config.TrainingURL + "/add_training_data_bulk"
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, bytes.NewBuffer(data))
	if err != nil {
		p.logger.Error(err, "Failed to create bulk POST request", "url", url)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		p.logger.Error(err, "Bulk POST failed", "url", url)
		return
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body) // Ensure body is read and closed

	if resp.StatusCode != http.StatusAccepted {
		p.logger.Error(fmt.Errorf("status %d", resp.StatusCode),
			"Bulk POST returned non-202 status", "url", url)
	} else {
		p.logger.V(1).Info("Flushed training batch", "sent_count", len(batch), "original_count", originalSize)
	}
}

// refreshMetrics GETs /metrics from training server and caches parsed coefficients or fetches XGBoost trees.
func (p *Predictor) refreshMetrics() {
	ctx, cancel := context.WithTimeout(context.Background(), p.config.HTTPTimeout)
	defer cancel()

	// Refresh model info first
	if err := p.refreshModelInfo(ctx); err != nil {
		p.logger.Error(err, "Failed to refresh model info during periodic refresh")
		return
	}

	p.metricsMu.RLock()
	modelType := ""
	if p.modelInfo != nil {
		modelType = p.modelInfo.ModelType
	}
	p.metricsMu.RUnlock()

	if modelType == "" {
		p.logger.V(1).Info("Cannot refresh metrics: model type is unknown")
		return
	}

	switch modelType {
	case "bayesian_ridge":
		if _, err := p.GetMetrics(ctx); err != nil {
			p.logger.Error(err, "Failed to refresh Bayesian Ridge metrics")
		}
	case "xgboost":
		trees, err := p.getXGBoostTrees(ctx)
		if err != nil {
			p.logger.Error(err, "Failed to fetch XGBoost trees")
			return
		}

		p.metricsMu.Lock()
		if p.cachedMetrics == nil {
			p.cachedMetrics = &MetricsResponse{}
		}
		p.cachedMetrics.ModelType = modelType
		p.cachedMetrics.XGBoostTrees = trees
		p.metricsMu.Unlock()

		if p.IsXGBoostReady() {
			p.logger.V(1).Info("Successfully refreshed XGBoost models")
		} else {
			p.logger.V(1).Info("XGBoost models not ready, will use HTTP fallback")
		}
	default:
		p.logger.Info("Unknown model type, cannot refresh metrics", "model_type", modelType)
	}
}

// Predict uses cached coefficients (Bayesian Ridge) or XGBoost models for local prediction.
func (p *Predictor) Predict(ctx context.Context, req PredictionRequest) (*PredictionResponse, error) {
	p.metricsMu.RLock()
	mr := p.cachedMetrics
	modelInfo := p.modelInfo
	p.metricsMu.RUnlock()

	if modelInfo == nil {
		return nil, fmt.Errorf("model info not yet available")
	}

	switch modelInfo.ModelType {
	case "bayesian_ridge":
		return p.predictBayesianRidge(req, mr)
	case "xgboost":
		return p.predictXGBoostHTTP(ctx, req)
	default:
		return nil, fmt.Errorf("unsupported or unknown model type: %s", modelInfo.ModelType)
	}
}

// predictBayesianRidge uses cached coefficients for linear prediction
func (p *Predictor) predictBayesianRidge(req PredictionRequest, mr *MetricsResponse) (*PredictionResponse, error) {
	if mr == nil || mr.Coefficients == nil {
		return nil, fmt.Errorf("no cached Bayesian Ridge coefficients available for prediction")
	}
	c := mr.Coefficients

	// Updated linear combination for TTFT to include prefix_cache_score
	ttft := c.TTFTIntercept +
		c.TTFTCoeffs["kv_cache_percentage"]*req.KVCachePercentage +
		c.TTFTCoeffs["input_token_length"]*float64(req.InputTokenLength) +
		c.TTFTCoeffs["num_request_waiting"]*float64(req.NumRequestWaiting) +
		c.TTFTCoeffs["num_request_running"]*float64(req.NumRequestRunning) +
		c.TTFTCoeffs["prefix_cache_score"]*req.PrefixCacheScore // Added prefix cache score

	// Linear combination for TPOT (remains unchanged - no prefix cache effect)
	tpot := c.TPOTIntercept +
		c.TPOTCoeffs["kv_cache_percentage"]*req.KVCachePercentage +
		c.TPOTCoeffs["input_token_length"]*float64(req.InputTokenLength) +
		c.TPOTCoeffs["num_request_waiting"]*float64(req.NumRequestWaiting) +
		c.TPOTCoeffs["num_request_running"]*float64(req.NumRequestRunning) +
		c.TPOTCoeffs["num_tokens_generated"]*float64(req.NumTokensGenerated)

	return &PredictionResponse{
		TTFT:        ttft,
		TPOT:        tpot,
		PredictedAt: time.Now(),
		ModelType:   "bayesian_ridge",
		Quantile:    0.9,
	}, nil
}

// predictXGBoostHTTP makes an HTTP call to a randomly selected prediction server for XGBoost predictions
func (p *Predictor) predictXGBoostHTTP(ctx context.Context, req PredictionRequest) (*PredictionResponse, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal prediction request: %w", err)
	}

	// Get random prediction URL for load balancing
	predictionURL := p.getRandomPredictionURL()
	url := predictionURL + "/predict"

	p.logger.V(2).Info("Making prediction request", "url", url)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to call prediction endpoint %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("prediction server returned non-200 status: %d %s, body: %s", resp.StatusCode, resp.Status, string(body))
	}

	var predResp PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&predResp); err != nil {
		return nil, fmt.Errorf("failed to decode prediction response: %w", err)
	}

	return &predResp, nil
}

// GetMetrics fetches & parses metrics from the training server (for Bayesian Ridge).
func (p *Predictor) GetMetrics(ctx context.Context) (*MetricsResponse, error) {
	url := p.config.TrainingURL + "/metrics"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create metrics request: %w", err)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call training server /metrics endpoint: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("training server returned non-200 status: %d %s, body: %s", resp.StatusCode, resp.Status, string(body))
	}

	rawMetricsBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read metrics response body: %w", err)
	}
	rawMetrics := string(rawMetricsBytes)

	metricsResponse := &MetricsResponse{
		RawMetrics: rawMetrics,
		ModelType:  "bayesian_ridge", // Assume Bayesian Ridge when calling /metrics
	}

	coeffs, buckets, err := p.parsePrometheusMetrics(rawMetrics)
	if err != nil {
		p.logger.Error(err, "Failed to parse Prometheus metrics, caching raw only")
	} else {
		metricsResponse.Coefficients = coeffs
		metricsResponse.BucketCounts = buckets
	}

	p.metricsMu.Lock()
	p.cachedMetrics = metricsResponse
	p.metricsMu.Unlock()

	p.logger.V(1).Info("Successfully retrieved and cached Bayesian Ridge metrics.")
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
	var firstErr error

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		if err := p.parseMetricLine(line, coefficients, bucketCounts); err != nil {
			if firstErr == nil {
				firstErr = err // Save first error to return
			}
			p.logger.V(2).Info("Skipping unparseable metric line", "line", line, "error", err)
		}
	}
	return coefficients, bucketCounts, firstErr
}

// parseMetricLine parses a single line of Prometheus-formatted text.
func (p *Predictor) parseMetricLine(line string, coefficients *ModelCoefficients, bucketCounts *BucketCounts) error {
	lastSpaceIdx := strings.LastIndexAny(line, " \t")
	if lastSpaceIdx == -1 {
		return fmt.Errorf("invalid metric format: no space found")
	}

	metricPart := strings.TrimSpace(line[:lastSpaceIdx])
	valueStr := strings.TrimSpace(line[lastSpaceIdx+1:])

	value, err := strconv.ParseFloat(valueStr, 64)
	if err != nil {
		return fmt.Errorf("could not parse value '%s': %w", valueStr, err)
	}

	metricName := metricPart
	if openBrace := strings.Index(metricPart, "{"); openBrace != -1 {
		metricName = metricPart[:openBrace]
	}

	switch metricName {
	case "ttft_intercept":
		coefficients.TTFTIntercept = value
	case "tpot_intercept":
		coefficients.TPOTIntercept = value
	case "ttft_coef":
		if feature := p.extractLabel(metricPart, "feature"); feature != "" {
			coefficients.TTFTCoeffs[feature] = value
		}
	case "tpot_coef":
		if feature := p.extractLabel(metricPart, "feature"); feature != "" {
			coefficients.TPOTCoeffs[feature] = value
		}
	case "training_samples_count":
		model := p.extractLabel(metricPart, "model")
		bucketStr := p.extractLabel(metricPart, "bucket")
		if bucket, err := strconv.Atoi(bucketStr); err == nil {
			switch model {
			case "ttft":
				bucketCounts.TTFTBuckets[bucket] = int(value)
			case "tpot":
				bucketCounts.TPOTBuckets[bucket] = int(value)
			}
		}
	}
	return nil
}

// extractLabel extracts a label value from a Prometheus metric string.
// Example: `metric{key="value"}`, `key` -> `"value"`
func (p *Predictor) extractLabel(metricPart, labelName string) string {
	searchStr := labelName + `="`
	start := strings.Index(metricPart, searchStr)
	if start == -1 {
		return ""
	}
	start += len(searchStr)
	end := strings.Index(metricPart[start:], `"`)
	if end == -1 {
		return ""
	}
	return metricPart[start : start+end]
}

// GetModelCoefficients fetches the latest metrics and returns the parsed coefficients.
func (p *Predictor) GetModelCoefficients(ctx context.Context) (*ModelCoefficients, error) {
	metrics, err := p.GetMetrics(ctx)
	if err != nil {
		return nil, err
	}
	if metrics.Coefficients == nil {
		return nil, fmt.Errorf("coefficients not available in fetched metrics")
	}
	return metrics.Coefficients, nil
}

// GetBucketCounts fetches the latest metrics and returns the parsed bucket counts.
func (p *Predictor) GetBucketCounts(ctx context.Context) (*BucketCounts, error) {
	metrics, err := p.GetMetrics(ctx)
	if err != nil {
		return nil, err
	}
	if metrics.BucketCounts == nil {
		return nil, fmt.Errorf("bucket counts not available in fetched metrics")
	}
	return metrics.BucketCounts, nil
}

// GetXGBoostTrees returns the cached XGBoost tree data. It does not fetch new data.
func (p *Predictor) GetXGBoostTrees(ctx context.Context) (*XGBoostTrees, error) {
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()
	if p.cachedMetrics == nil || p.cachedMetrics.XGBoostTrees == nil {
		return nil, fmt.Errorf("no cached XGBoost trees available")
	}
	return p.cachedMetrics.XGBoostTrees, nil
}

// GetModelInfo fetches the latest model info from the training server.
func (p *Predictor) GetModelInfo(ctx context.Context) (*ModelInfo, error) {
	if err := p.refreshModelInfo(ctx); err != nil {
		return nil, err
	}
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()

	return p.modelInfo, nil
}

// GetCachedMetrics returns the last metrics fetched. The bool indicates if a value is cached.
func (p *Predictor) GetCachedMetrics() (*MetricsResponse, bool) {
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()
	if p.cachedMetrics == nil {
		return nil, false
	}
	return p.cachedMetrics, true
}

// IsXGBoostReady returns true if native XGBoost models are loaded and ready.
func (p *Predictor) IsXGBoostReady() bool {
	p.xgboostMu.RLock()
	defer p.xgboostMu.RUnlock()
	return p.modelInfo != nil && p.modelInfo.ModelType == "xgboost"
}

// IsBayesianRidgeReady returns true if Bayesian Ridge coefficients are cached.
func (p *Predictor) IsBayesianRidgeReady() bool {
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()
	return p.cachedMetrics != nil && p.cachedMetrics.Coefficients != nil
}

// GetCurrentModelType returns the current model type from cached model info.
func (p *Predictor) GetCurrentModelType() string {
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()
	if p.modelInfo == nil {
		return ""
	}
	return p.modelInfo.ModelType
}

// IsReady returns true if a prediction method is ready based on the current model type.
func (p *Predictor) IsReady() bool {
	switch p.GetCurrentModelType() {
	case "bayesian_ridge":
		return p.IsBayesianRidgeReady()
	case "xgboost":
		// Ready if native models are loaded OR we have prediction URLs for HTTP fallback.
		return p.IsXGBoostReady() || len(p.config.PredictionURLs) > 0
	default:
		return false
	}
}

// GetPredictionURLs returns the list of configured prediction URLs for debugging/monitoring.
func (p *Predictor) GetPredictionURLs() []string {
	return p.config.PredictionURLs
}

// GetTrainingURL returns the configured training URL for debugging/monitoring.
func (p *Predictor) GetTrainingURL() string {
	return p.config.TrainingURL
}

// ValidatePredictionRequest validates that a prediction request has all required fields
// with valid values, including the new prefix_cache_score field.
func (p *Predictor) ValidatePredictionRequest(req PredictionRequest) error {
	if req.KVCachePercentage < 0.0 || req.KVCachePercentage > 1.0 {
		return fmt.Errorf("kv_cache_percentage must be between 0.0 and 1.0, got %f", req.KVCachePercentage)
	}
	if req.InputTokenLength < 0 {
		return fmt.Errorf("input_token_length must be non-negative, got %d", req.InputTokenLength)
	}
	if req.NumRequestWaiting < 0 {
		return fmt.Errorf("num_request_waiting must be non-negative, got %d", req.NumRequestWaiting)
	}
	if req.NumRequestRunning < 0 {
		return fmt.Errorf("num_request_running must be non-negative, got %d", req.NumRequestRunning)
	}
	if req.NumTokensGenerated < 0 {
		return fmt.Errorf("num_tokens_generated must be non-negative, got %d", req.NumTokensGenerated)
	}
	if req.PrefixCacheScore < 0.0 || req.PrefixCacheScore > 1.0 {
		return fmt.Errorf("prefix_cache_score must be between 0.0 and 1.0, got %f", req.PrefixCacheScore)
	}
	return nil
}

// ValidateTrainingEntry validates that a training entry has all required fields
// with valid values, including the new prefix_cache_score field.
func (p *Predictor) ValidateTrainingEntry(entry TrainingEntry) error {
	if entry.KVCachePercentage < 0.0 || entry.KVCachePercentage > 1.0 {
		return fmt.Errorf("kv_cache_percentage must be between 0.0 and 1.0, got %f", entry.KVCachePercentage)
	}
	if entry.InputTokenLength < 0 {
		return fmt.Errorf("input_token_length must be non-negative, got %d", entry.InputTokenLength)
	}
	if entry.NumRequestWaiting < 0 {
		return fmt.Errorf("num_request_waiting must be non-negative, got %d", entry.NumRequestWaiting)
	}
	if entry.NumRequestRunning < 0 {
		return fmt.Errorf("num_request_running must be non-negative, got %d", entry.NumRequestRunning)
	}
	if entry.NumTokensGenerated < 0 {
		return fmt.Errorf("num_tokens_generated must be non-negative, got %d", entry.NumTokensGenerated)
	}
	if entry.ActualTTFT < 0.0 {
		return fmt.Errorf("actual_ttft_ms must be non-negative, got %f", entry.ActualTTFT)
	}
	if entry.ActualTPOT < 0.0 {
		return fmt.Errorf("actual_tpot_ms must be non-negative, got %f", entry.ActualTPOT)
	}
	if entry.PrefixCacheScore < 0.0 || entry.PrefixCacheScore > 1.0 {
		return fmt.Errorf("prefix_cache_score must be between 0.0 and 1.0, got %f", entry.PrefixCacheScore)
	}
	return nil
}

// NewTrainingEntry is a helper function to create a new TrainingEntry with proper validation.
func NewTrainingEntry(
	kvCachePercentage float64,
	inputTokenLength int,
	numRequestWaiting int,
	numRequestRunning int,
	numTokensGenerated int,
	actualTTFT float64,
	actualTPOT float64,
	prefixCacheScore float64,
) (TrainingEntry, error) {
	entry := TrainingEntry{
		KVCachePercentage:  kvCachePercentage,
		InputTokenLength:   inputTokenLength,
		NumRequestWaiting:  numRequestWaiting,
		NumRequestRunning:  numRequestRunning,
		NumTokensGenerated: numTokensGenerated,
		ActualTTFT:         actualTTFT,
		ActualTPOT:         actualTPOT,
		PrefixCacheScore:   prefixCacheScore,
		Timestamp:          time.Now(),
	}

	// Create a temporary predictor for validation (could be optimized)
	p := &Predictor{}
	if err := p.ValidateTrainingEntry(entry); err != nil {
		return TrainingEntry{}, err
	}

	return entry, nil
}

// NewPredictionRequest is a helper function to create a new PredictionRequest with proper validation.
func NewPredictionRequest(
	kvCachePercentage float64,
	inputTokenLength int,
	numRequestWaiting int,
	numRequestRunning int,
	numTokensGenerated int,
	prefixCacheScore float64,
) (PredictionRequest, error) {
	req := PredictionRequest{
		KVCachePercentage:  kvCachePercentage,
		InputTokenLength:   inputTokenLength,
		NumRequestWaiting:  numRequestWaiting,
		NumRequestRunning:  numRequestRunning,
		NumTokensGenerated: numTokensGenerated,
		PrefixCacheScore:   prefixCacheScore,
	}

	// Create a temporary predictor for validation (could be optimized)
	p := &Predictor{}
	if err := p.ValidatePredictionRequest(req); err != nil {
		return PredictionRequest{}, err
	}

	return req, nil
}
