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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// Predict uses cached coefficients (Bayesian Ridge) or HTTP calls (XGBoost/LightGBM) for prediction.
func (p *Predictor) Predict(ctx context.Context, req PredictionRequest) (*PredictionResponse, error) {
	// Get current model type from server status first, fall back to model info
	p.metricsMu.RLock()
	modelType := ""
	quantile := 0.9 // default

	if p.serverStatus != nil {
		modelType = p.serverStatus.ModelType
		quantile = p.serverStatus.Quantile
	} else if p.modelInfo != nil {
		modelType = p.modelInfo.ModelType
		if p.modelInfo.Quantile > 0 {
			quantile = p.modelInfo.Quantile
		}
	}

	mr := p.cachedMetrics
	p.metricsMu.RUnlock()

	if modelType == "" {
		return nil, errors.New("model type not yet available from server")
	}

	switch modelType {
	case bayesianRidgeModelType:
		return p.predictBayesianRidge(req, mr, quantile)
	case xgBoostModelType, gbmModelType:
		return p.predictHTTP(ctx, req)
	default:
		return nil, fmt.Errorf("unsupported or unknown model type: %s", modelType)
	}
}

// PredictBulk makes bulk predictions with error handling (allows partial failures)
func (p *Predictor) PredictBulk(ctx context.Context, requests []PredictionRequest) (*BulkPredictionResponse, error) {
	if len(requests) == 0 {
		return nil, errors.New("no prediction requests provided")
	}

	if len(requests) > p.config.MaxBulkSize {
		return nil, fmt.Errorf("too many requests: %d (max: %d)", len(requests), p.config.MaxBulkSize)
	}

	// Validate all requests first
	for i, req := range requests {
		if err := p.ValidatePredictionRequest(req); err != nil {
			return nil, fmt.Errorf("validation failed for request %d: %w", i, err)
		}
	}

	payload := BulkPredictionRequest{Requests: requests}
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal bulk prediction request: %w", err)
	}

	predictionURL := p.getRandomPredictionURL()
	url := predictionURL + "/predict/bulk"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create bulk prediction request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call bulk prediction endpoint %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bulk prediction server returned non-200 status: %d %s, body: %s", resp.StatusCode, resp.Status, string(body))
	}

	var bulkResp BulkPredictionResponseWithErrors
	if err := json.NewDecoder(resp.Body).Decode(&bulkResp); err != nil {
		return nil, fmt.Errorf("failed to decode bulk prediction response: %w", err)
	}

	// Convert to standard bulk response format
	var predictions []PredictionResponse
	for _, pred := range bulkResp.Predictions {
		if pred != nil {
			predictions = append(predictions, *pred)
		}
	}

	return &BulkPredictionResponse{
		Predictions:           predictions,
		TotalRequests:         bulkResp.TotalRequests,
		SuccessfulPredictions: bulkResp.SuccessfulPredictions,
		FailedPredictions:     bulkResp.FailedPredictions,
		ProcessingTimeMs:      bulkResp.ProcessingTimeMs,
	}, nil
}

// PredictBulkStrict makes bulk predictions that fail if any single prediction fails
func (p *Predictor) PredictBulkStrict(ctx context.Context, requests []PredictionRequest) (*BulkPredictionResponse, error) {
	if len(requests) == 0 {
		return nil, errors.New("no prediction requests provided")
	}

	if len(requests) > p.config.MaxBulkSize {
		return nil, fmt.Errorf("too many requests: %d (max: %d)", len(requests), p.config.MaxBulkSize)
	}

	// Validate all requests first
	for i, req := range requests {
		if err := p.ValidatePredictionRequest(req); err != nil {
			return nil, fmt.Errorf("validation failed for request %d: %w", i, err)
		}
	}

	payload := BulkPredictionRequest{Requests: requests}
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal bulk prediction request: %w", err)
	}

	predictionURL := p.getRandomPredictionURL()
	url := predictionURL + "/predict/bulk/strict"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create bulk prediction request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call bulk prediction endpoint %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bulk prediction server returned non-200 status: %d %s, body: %s", resp.StatusCode, resp.Status, string(body))
	}

	var bulkResp BulkPredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&bulkResp); err != nil {
		return nil, fmt.Errorf("failed to decode bulk prediction response: %w", err)
	}

	return &bulkResp, nil
}

// predictBayesianRidge uses cached coefficients for linear prediction
func (p *Predictor) predictBayesianRidge(req PredictionRequest, mr *MetricsResponse, quantile float64) (*PredictionResponse, error) {
	if mr == nil || mr.Coefficients == nil {
		return nil, errors.New("no cached Bayesian Ridge coefficients available for prediction")
	}
	c := mr.Coefficients

	// Updated linear combination for TTFT to include prefix_cache_score
	ttft := c.TTFTIntercept +
		c.TTFTCoeffs["kv_cache_percentage"]*req.KVCachePercentage +
		c.TTFTCoeffs["input_token_length"]*float64(req.InputTokenLength) +
		c.TTFTCoeffs["num_request_waiting"]*float64(req.NumRequestWaiting) +
		c.TTFTCoeffs["num_request_running"]*float64(req.NumRequestRunning) +
		c.TTFTCoeffs["prefix_cache_score"]*req.PrefixCacheScore

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
		Quantile:    quantile,
	}, nil
}

// predictHTTP makes an HTTP call to a randomly selected prediction server for XGBoost/LightGBM predictions
func (p *Predictor) predictHTTP(ctx context.Context, req PredictionRequest) (*PredictionResponse, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal prediction request: %w", err)
	}

	// Get random prediction URL for load balancing
	predictionURL := p.getRandomPredictionURL()
	url := predictionURL + "/predict"

	p.logger.V(logutil.TRACE).Info("Making prediction request", "url", url)

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

// refreshServerStatus gets current server status from a prediction server
func (p *Predictor) refreshServerStatus(ctx context.Context) error {
	predictionURL := p.getRandomPredictionURL()
	url := predictionURL + "/status"

	p.logger.V(logutil.DEBUG).Info("Fetching server status", "url", url)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create server status request: %w", err)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call /status endpoint: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server %s returned non-200 status: %d %s, body: %s", url, resp.StatusCode, resp.Status, string(body))
	}

	var status ServerStatusResponse
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return fmt.Errorf("failed to decode server status response: %w", err)
	}

	p.metricsMu.Lock()
	p.serverStatus = &status
	p.metricsMu.Unlock()

	p.logger.V(logutil.DEBUG).Info("Retrieved server status",
		"model_type", status.ModelType,
		"quantile", status.Quantile,
		"is_ready", status.IsReady)
	return nil
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
