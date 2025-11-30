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

package latencypredictorasync

import (
	"context"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	gbmModelType           = "lightgbm"
	bayesianRidgeModelType = "bayesian_ridge"
	xgBoostModelType       = "xgboost"
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
	// MetricsRefreshInterval determines how often to refresh cached metrics.
	MetricsRefreshInterval time.Duration
	// MaxBulkSize is the maximum number of predictions to send in a single bulk request.
	MaxBulkSize int
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
		MaxBulkSize:            100,
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
	if bulkStr := os.Getenv("LATENCY_MAX_BULK_SIZE"); bulkStr != "" {
		if size, err := strconv.Atoi(bulkStr); err == nil && size > 0 && size <= 100 {
			cfg.MaxBulkSize = size
		}
	}
	return cfg
}

// Predictor defines the interface for latency prediction and training.
type PredictorInterface interface {
	Predict(ctx context.Context, req PredictionRequest) (*PredictionResponse, error)
	PredictBulk(ctx context.Context, requests []PredictionRequest) (*BulkPredictionResponse, error)
	PredictBulkStrict(ctx context.Context, requests []PredictionRequest) (*BulkPredictionResponse, error)
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
	PrefixCacheScore   float64   `json:"prefix_cache_score"`
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
	PrefixCacheScore   float64 `json:"prefix_cache_score"`
}

type PredictionResponse struct {
	TTFT                 float64    `json:"ttft_ms"`
	TPOT                 float64    `json:"tpot_ms"`
	TTFTUncertainty      float64    `json:"ttft_uncertainty,omitempty"`
	TPOTUncertainty      float64    `json:"tpot_uncertainty,omitempty"`
	TTFTPredictionBounds [2]float64 `json:"ttft_prediction_bounds,omitempty"`
	TPOTPredictionBounds [2]float64 `json:"tpot_prediction_bounds,omitempty"`
	PredictedAt          time.Time  `json:"predicted_at"`
	ModelType            string     `json:"model_type"`
	Quantile             float64    `json:"quantile"`
	LastModelLoad        *time.Time `json:"last_model_load"`
}

// New data models for bulk predictions
type BulkPredictionRequest struct {
	Requests []PredictionRequest `json:"requests"`
}

type BulkPredictionResponse struct {
	Predictions           []PredictionResponse `json:"predictions"`
	TotalRequests         int                  `json:"total_requests"`
	SuccessfulPredictions int                  `json:"successful_predictions"`
	FailedPredictions     int                  `json:"failed_predictions"`
	ProcessingTimeMs      float64              `json:"processing_time_ms"`
}

type BulkPredictionError struct {
	Index   int               `json:"index"`
	Error   string            `json:"error"`
	Request PredictionRequest `json:"request"`
}

type BulkPredictionResponseWithErrors struct {
	Predictions           []*PredictionResponse `json:"predictions"`
	Errors                []BulkPredictionError `json:"errors"`
	TotalRequests         int                   `json:"total_requests"`
	SuccessfulPredictions int                   `json:"successful_predictions"`
	FailedPredictions     int                   `json:"failed_predictions"`
	ProcessingTimeMs      float64               `json:"processing_time_ms"`
}

// Server status response
type ServerStatusResponse struct {
	IsReady           bool            `json:"is_ready"`
	ModelType         string          `json:"model_type"`
	Quantile          float64         `json:"quantile"`
	LastModelLoad     *time.Time      `json:"last_model_load"`
	TrainingServerURL string          `json:"training_server_url"`
	ModelsExist       map[string]bool `json:"models_exist"`
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
	Quantile    float64         `json:"quantile"`
}

type MetricsResponse struct {
	ModelType    string             `json:"model_type"`
	Coefficients *ModelCoefficients `json:"coefficients"`
	XGBoostTrees *XGBoostTrees      `json:"xgboost_trees"`
	BucketCounts *BucketCounts      `json:"bucket_counts"`
	RawMetrics   string             `json:"raw_metrics"`
}
