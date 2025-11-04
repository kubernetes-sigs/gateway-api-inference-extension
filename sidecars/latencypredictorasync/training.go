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
	"fmt"
	"io"
	"net/http"
	"time"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

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

		switch {
		case hasTTFT:
			ttftEntries = append(ttftEntries, entry)
		case hasTPOT:
			tpotEntries = append(tpotEntries, entry)
		default:
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

		// Distribute remaining samples to the largest *original* group
		switch {
		case len(ttftEntries) >= len(tpotEntries) && len(ttftEntries) >= len(otherEntries):
			ttftSampleSize += remaining
		case len(tpotEntries) >= len(otherEntries):
			tpotSampleSize += remaining
		default:
			otherSampleSize += remaining
		}

	} else if totalSampled > maxSize {
		excess := totalSampled - maxSize

		// Reduce from the largest *sampled* group
		switch {
		case ttftSampleSize >= tpotSampleSize && ttftSampleSize >= otherSampleSize:
			ttftSampleSize -= excess
		case tpotSampleSize >= otherSampleSize:
			tpotSampleSize -= excess
		default:
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
		p.logger.V(logutil.DEBUG).Info("Sampled training entries for flush",
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

	// Ensure body is read and closed
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		p.logger.Error(err, "failed to read response body", "url", url)
	}

	if resp.StatusCode != http.StatusAccepted {
		p.logger.Error(fmt.Errorf("status %d", resp.StatusCode),
			"Bulk POST returned non-202 status", "url", url)
	} else {
		p.logger.V(logutil.DEBUG).Info("Flushed training batch", "sent_count", len(batch), "original_count", originalSize)
	}
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

// refreshModelInfo gets current model type and readiness info from training server
func (p *Predictor) refreshModelInfo(ctx context.Context) error {
	url := p.config.TrainingURL + "/model/download/info"
	p.logger.V(logutil.DEBUG).Info("Fetching model info", "url", url)
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

	p.logger.V(logutil.DEBUG).Info("Retrieved model info", "model_type", modelInfo.ModelType, "model_status", modelInfo.ModelStatus)
	return nil
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
		p.logger.V(logutil.DEBUG).Info("Cannot refresh metrics: model type is unknown")
		return
	}

	switch modelType {
	case bayesianRidgeModelType:
		if _, err := p.GetMetrics(ctx); err != nil {
			p.logger.Error(err, "Failed to refresh Bayesian Ridge metrics")
		}
	case xgBoostModelType:
		if p.config.UseNativeXGBoost {
			// Fetch XGBoost trees for native predictions
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

			p.logger.V(logutil.DEBUG).Info("Updated XGBoost trees for native predictions")
		} else {
			// Just update model type for HTTP-based predictions
			p.metricsMu.Lock()
			if p.cachedMetrics == nil {
				p.cachedMetrics = &MetricsResponse{}
			}
			p.cachedMetrics.ModelType = modelType
			p.metricsMu.Unlock()

			p.logger.V(logutil.DEBUG).Info("Updated model type for HTTP-based predictions", "model_type", modelType)
		}
	case gbmModelType:
		// LightGBM only supports HTTP calls, no native tree caching needed
		p.metricsMu.Lock()
		if p.cachedMetrics == nil {
			p.cachedMetrics = &MetricsResponse{}
		}
		p.cachedMetrics.ModelType = modelType
		p.metricsMu.Unlock()

		p.logger.V(logutil.DEBUG).Info("Updated model type for HTTP-based predictions", "model_type", modelType)
	default:
		p.logger.Info("Unknown model type, cannot refresh metrics", "model_type", modelType)
	}
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
