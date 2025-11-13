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
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/go-logr/zapr"
	"go.uber.org/zap"
)

func TestLatencyPredictorIntegration(t *testing.T) {
	// Setup logger
	zapLog, err := zap.NewDevelopment()
	if err != nil {
		t.Fatalf("Failed to create logger: %v", err)
	}
	logger := zapr.NewLogger(zapLog)

	// Check if server URLs are set
	predictionURLs := os.Getenv("PREDICTION_SERVER_URL")
	trainingURL := os.Getenv("TRAINING_SERVER_URL")

	if predictionURLs == "" {
		t.Skip("PREDICTION_SERVER_URL not set, skipping integration test")
	}
	if trainingURL == "" {
		// Fallback to first prediction URL for training if not set
		urls := strings.Split(predictionURLs, ",")
		if len(urls) > 0 {
			trainingURL = strings.TrimSpace(urls[0])
		} else {
			t.Skip("No valid URLs available for testing")
		}
	}

	// Parse prediction URLs
	urls := strings.Split(predictionURLs, ",")
	parsedPredictionURLs := make([]string, 0, len(urls))
	for _, url := range urls {
		parsedPredictionURLs = append(parsedPredictionURLs, strings.TrimSpace(url))
	}

	// Create config with the actual server URLs
	config := &Config{
		TrainingURL:            trainingURL,
		PredictionURLs:         parsedPredictionURLs,
		MaxSampleSize:          1000,
		FlushInterval:          500 * time.Millisecond, // Shorter for testing
		MetricsRefreshInterval: 1 * time.Second,        // Longer for metrics
		UseNativeXGBoost:       true,
		HTTPTimeout:            30 * time.Second, // Longer timeout for tests
		MaxBulkSize:            50,               // Test bulk size
	}

	// Create predictor
	predictor := New(config, logger)
	defer predictor.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Start the predictor
	err = predictor.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start predictor: %v", err)
	}

	t.Run("TestModelInfo", func(t *testing.T) {
		testModelInfo(t, ctx, predictor)
	})

	t.Run("TestBulkTrainingData", func(t *testing.T) {
		testBulkTrainingData(t, predictor)
	})

	t.Run("TestPrediction", func(t *testing.T) {
		testPrediction(t, ctx, predictor)
	})

	t.Run("TestBulkPredictions", func(t *testing.T) {
		testBulkPredictions(t, ctx, predictor)
	})

	t.Run("TestBulkPredictionsStrict", func(t *testing.T) {
		testBulkPredictionsStrict(t, ctx, predictor)
	})

	t.Run("TestPredictionWithPrefixCache", func(t *testing.T) {
		testPredictionWithPrefixCache(t, ctx, predictor)
	})

	t.Run("TestHTTPFallbackPrediction", func(t *testing.T) {
		testHTTPFallbackPrediction(t, ctx, predictor)
	})

	t.Run("TestLightGBMSupport", func(t *testing.T) {
		testLightGBMSupport(t, ctx, predictor)
	})

	t.Run("TestPredictionPerformance", func(t *testing.T) {
		testPredictionPerformance(t, ctx, predictor)
	})

	t.Run("TestBulkPredictionPerformance", func(t *testing.T) {
		testBulkPredictionPerformance(t, ctx, predictor)
	})

	t.Run("TestHTTPOnlyPerformance", func(t *testing.T) {
		testHTTPOnlyPerformance(t, ctx)
	})

	t.Run("TestXGBoostJSONStructure", func(t *testing.T) {
		testXGBoostJSONStructure(t, ctx, predictor)
	})

	t.Run("TestHTTPOnlyPrediction", func(t *testing.T) {
		testHTTPOnlyPrediction(t, ctx)
	})

	t.Run("TestMetricsRetrieval", func(t *testing.T) {
		testMetricsRetrieval(t, ctx, predictor)
	})

	t.Run("TestLoadBalancing", func(t *testing.T) {
		testLoadBalancing(t, ctx, predictor)
	})

	t.Run("TestPrefixCacheValidation", func(t *testing.T) {
		testPrefixCacheValidation(t, predictor)
	})

	t.Run("TestPredictionConstructors", func(t *testing.T) {
		testPredictionConstructors(t)
	})
}

func testModelInfo(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing model info retrieval...")

	modelInfo, err := predictor.GetModelInfo(ctx)
	if err != nil {
		t.Fatalf("Failed to get model info: %v", err)
	}

	t.Logf("Model Info - Type: %s, Model Status: %v, Quantile: %.2f",
		modelInfo.ModelType, modelInfo.ModelStatus, modelInfo.Quantile)

	if modelInfo.ModelType == "" {
		t.Error("Model type should not be empty")
	}

	// Store model type for other tests
	currentModelType := predictor.GetCurrentModelType()
	t.Logf("Current model type from predictor: %s", currentModelType)

	// Log URLs being used
	t.Logf("Training URL: %s", predictor.GetTrainingURL())
	t.Logf("Prediction URLs: %v", predictor.GetPredictionURLs())
}

func testBulkTrainingData(t *testing.T, predictor *Predictor) {
	t.Log("Testing bulk training data submission with prefix cache score...")

	// Generate 1000 random training entries including prefix cache scores
	entries := generateTrainingEntries(1000)

	err := predictor.AddTrainingDataBulk(entries)
	if err != nil {
		t.Fatalf("Failed to add bulk training data: %v", err)
	}

	t.Logf("Successfully added %d training entries to buffer (with prefix cache scores)", len(entries))

	// Wait a bit for the background flush to occur
	time.Sleep(2 * time.Second)

	t.Log("Training data should have been flushed to training server")
}

func testPrediction(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing prediction functionality...")

	// Log current predictor state
	t.Logf("Predictor state:")
	t.Logf("  Current model type: %s", predictor.GetCurrentModelType())
	t.Logf("  Current quantile: %.2f", predictor.GetCurrentQuantile())
	t.Logf("  Overall ready: %t", predictor.IsReady())
	t.Logf("  XGBoost ready: %t", predictor.IsXGBoostReady())
	t.Logf("  LightGBM ready: %t", predictor.IsLightGBMReady())
	t.Logf("  Bayesian Ridge ready: %t", predictor.IsBayesianRidgeReady())

	// Wait for models to be ready
	maxWait := 30 * time.Second
	waitTime := 100 * time.Millisecond
	elapsed := time.Duration(0)

	for elapsed < maxWait {
		if predictor.IsReady() {
			break
		}
		time.Sleep(waitTime)
		elapsed += waitTime
	}

	if !predictor.IsReady() {
		t.Log("Warning: Predictor not ready after waiting, attempting prediction anyway")
	}

	// Create a sample prediction request with prefix cache score
	req := PredictionRequest{
		KVCachePercentage:  0.755, // 75.5% as a fraction
		InputTokenLength:   512,
		NumRequestWaiting:  3,
		NumRequestRunning:  2,
		NumTokensGenerated: 100,
		PrefixCacheScore:   0.8, // 80% prefix cache hit rate
	}

	t.Logf("Making prediction request: %+v", req)

	response, err := predictor.Predict(ctx, req)
	if err != nil {
		t.Fatalf("Failed to make prediction: %v", err)
	}

	t.Logf("Prediction Response:")
	t.Logf("  TTFT: %.2f ms (uncertainty: %.2f)", response.TTFT, response.TTFTUncertainty)
	t.Logf("  TPOT: %.2f ms (uncertainty: %.2f)", response.TPOT, response.TPOTUncertainty)
	t.Logf("  TTFT Bounds: [%.2f, %.2f]", response.TTFTPredictionBounds[0], response.TTFTPredictionBounds[1])
	t.Logf("  TPOT Bounds: [%.2f, %.2f]", response.TPOTPredictionBounds[0], response.TPOTPredictionBounds[1])
	t.Logf("  Model Type: %s", response.ModelType)
	t.Logf("  Quantile: %.2f", response.Quantile)
	t.Logf("  Predicted At: %s", response.PredictedAt.Format(time.RFC3339))

	// Validate response
	if response.TTFT <= 0 {
		t.Error("TTFT should be positive")
	}
	if response.TPOT <= 0 {
		t.Error("TPOT should be positive")
	}
	if response.ModelType == "" {
		t.Error("Model type should not be empty")
	}
	if response.Quantile <= 0 || response.Quantile >= 1 {
		t.Errorf("Quantile should be between 0 and 1, got: %.2f", response.Quantile)
	}

	// Test multiple predictions to ensure consistency
	t.Log("Testing multiple predictions with varying prefix cache scores...")
	for i := 0; i < 5; i++ {
		testReq := PredictionRequest{
			KVCachePercentage:  float64(50+i*10) / 100.0, // Convert percentage to fraction
			InputTokenLength:   256 + i*128,
			NumRequestWaiting:  i,
			NumRequestRunning:  1 + i,
			NumTokensGenerated: 50 + i*25,
			PrefixCacheScore:   float64(i*20) / 100.0, // Vary prefix cache from 0% to 80%
		}

		resp, err := predictor.Predict(ctx, testReq)
		if err != nil {
			t.Errorf("Prediction %d failed: %v", i+1, err)
			continue
		}

		t.Logf("Prediction %d: TTFT=%.2f, TPOT=%.2f (prefix_cache=%.1f%%, quantile=%.2f)",
			i+1, resp.TTFT, resp.TPOT, testReq.PrefixCacheScore*100, resp.Quantile)
	}
}

func testBulkPredictions(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing bulk predictions with error tolerance...")

	if !predictor.IsReady() {
		t.Skip("Predictor not ready for bulk prediction testing")
	}

	// Create multiple prediction requests
	requests := make([]PredictionRequest, 10)
	for i := 0; i < 10; i++ {
		requests[i] = PredictionRequest{
			KVCachePercentage:  float64(40+i*5) / 100.0, // 40% to 85%
			InputTokenLength:   200 + i*50,              // 200 to 650
			NumRequestWaiting:  i % 5,                   // 0 to 4
			NumRequestRunning:  (i % 3) + 1,             // 1 to 3
			NumTokensGenerated: 25 + i*10,               // 25 to 115
			PrefixCacheScore:   float64(i) / 9.0,        // 0.0 to 1.0
		}
	}

	t.Logf("Making bulk prediction request with %d requests", len(requests))

	bulkResponse, err := predictor.PredictBulk(ctx, requests)
	if err != nil {
		t.Fatalf("Bulk prediction failed: %v", err)
	}

	t.Logf("Bulk Prediction Response:")
	t.Logf("  Total Requests: %d", bulkResponse.TotalRequests)
	t.Logf("  Successful: %d", bulkResponse.SuccessfulPredictions)
	t.Logf("  Failed: %d", bulkResponse.FailedPredictions)
	t.Logf("  Processing Time: %.2f ms", bulkResponse.ProcessingTimeMs)

	if bulkResponse.TotalRequests != len(requests) {
		t.Errorf("Expected %d total requests, got %d", len(requests), bulkResponse.TotalRequests)
	}

	if bulkResponse.SuccessfulPredictions != len(bulkResponse.Predictions) {
		t.Errorf("Successful count (%d) doesn't match predictions length (%d)",
			bulkResponse.SuccessfulPredictions, len(bulkResponse.Predictions))
	}

	// Validate each prediction in the response
	for i, prediction := range bulkResponse.Predictions {
		if prediction.TTFT <= 0 {
			t.Errorf("Prediction %d: TTFT should be positive, got %.2f", i, prediction.TTFT)
		}
		if prediction.TPOT <= 0 {
			t.Errorf("Prediction %d: TPOT should be positive, got %.2f", i, prediction.TPOT)
		}
		if prediction.ModelType == "" {
			t.Errorf("Prediction %d: Model type should not be empty", i)
		}

		t.Logf("  Prediction %d: TTFT=%.2f, TPOT=%.2f, quantile=%.2f",
			i+1, prediction.TTFT, prediction.TPOT, prediction.Quantile)
	}

	// Test performance expectation
	avgTimePerPrediction := bulkResponse.ProcessingTimeMs / float64(bulkResponse.SuccessfulPredictions)
	t.Logf("Average time per prediction: %.2f ms", avgTimePerPrediction)

	if avgTimePerPrediction > 100 { // Bulk should be more efficient
		t.Logf("Note: Bulk prediction averaging %.2f ms per request (may be acceptable)", avgTimePerPrediction)
	} else {
		t.Logf("✓ Good bulk prediction performance: %.2f ms per request", avgTimePerPrediction)
	}
}

func testBulkPredictionsStrict(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing strict bulk predictions...")

	if !predictor.IsReady() {
		t.Skip("Predictor not ready for strict bulk prediction testing")
	}

	// Create valid prediction requests
	requests := make([]PredictionRequest, 5)
	for i := 0; i < 5; i++ {
		requests[i] = PredictionRequest{
			KVCachePercentage:  0.6,
			InputTokenLength:   300 + i*100,
			NumRequestWaiting:  i,
			NumRequestRunning:  1,
			NumTokensGenerated: 50,
			PrefixCacheScore:   float64(i) / 4.0, // 0.0 to 1.0
		}
	}

	t.Logf("Making strict bulk prediction request with %d requests", len(requests))

	bulkResponse, err := predictor.PredictBulkStrict(ctx, requests)
	if err != nil {
		t.Fatalf("Strict bulk prediction failed: %v", err)
	}

	t.Logf("Strict Bulk Prediction Response:")
	t.Logf("  Total Requests: %d", bulkResponse.TotalRequests)
	t.Logf("  Successful: %d", bulkResponse.SuccessfulPredictions)
	t.Logf("  Failed: %d", bulkResponse.FailedPredictions)
	t.Logf("  Processing Time: %.2f ms", bulkResponse.ProcessingTimeMs)

	// In strict mode, we expect all requests to succeed or the entire batch to fail
	if bulkResponse.FailedPredictions > 0 {
		t.Errorf("Strict bulk prediction should not have partial failures, got %d failed",
			bulkResponse.FailedPredictions)
	}

	if bulkResponse.SuccessfulPredictions != len(requests) {
		t.Errorf("Expected all %d requests to succeed, got %d",
			len(requests), bulkResponse.SuccessfulPredictions)
	}

	// Test bulk size limits
	t.Log("Testing bulk size limits...")
	largeRequests := make([]PredictionRequest, 150) // Over the limit
	for i := range largeRequests {
		largeRequests[i] = requests[0] // Use a valid request template
	}

	_, err = predictor.PredictBulkStrict(ctx, largeRequests)
	if err == nil {
		t.Error("Expected error for oversized bulk request, but got none")
	} else {
		t.Logf("✓ Correctly rejected oversized bulk request: %v", err)
	}
}

func testLightGBMSupport(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing LightGBM support...")

	currentModelType := predictor.GetCurrentModelType()
	t.Logf("Current model type: %s", currentModelType)

	if currentModelType == gbmModelType {
		t.Log("Testing LightGBM-specific functionality...")

		// Test LightGBM readiness
		isReady := predictor.IsLightGBMReady()
		t.Logf("LightGBM ready: %t", isReady)

		if isReady {
			// Test LightGBM prediction
			req := PredictionRequest{
				KVCachePercentage:  0.7,
				InputTokenLength:   400,
				NumRequestWaiting:  2,
				NumRequestRunning:  1,
				NumTokensGenerated: 60,
				PrefixCacheScore:   0.8,
			}

			response, err := predictor.Predict(ctx, req)
			if err != nil {
				t.Errorf("LightGBM prediction failed: %v", err)
			} else {
				t.Logf("LightGBM prediction successful: TTFT=%.2f, TPOT=%.2f",
					response.TTFT, response.TPOT)

				if response.ModelType != gbmModelType {
					t.Errorf("Expected model type 'lightgbm', got '%s'", response.ModelType)
				}
			}
		} else {
			t.Log("LightGBM not ready, skipping LightGBM-specific tests")
		}
	} else {
		t.Logf("Current model type is %s, not LightGBM. LightGBM-specific tests skipped.", currentModelType)
	}

	// Test that the client handles all model types properly
	t.Log("Verifying model type handling...")
	switch currentModelType {
	case bayesianRidgeModelType:
		t.Log("✓ Bayesian Ridge model type recognized")
	case xgBoostModelType:
		t.Log("✓ XGBoost model type recognized")
	case gbmModelType:
		t.Log("✓ LightGBM model type recognized")
	default:
		t.Logf("⚠ Unknown model type: %s", currentModelType)
	}
}

func testPredictionWithPrefixCache(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing prefix cache score impact on predictions...")

	if !predictor.IsReady() {
		t.Skip("Predictor not ready for prefix cache testing")
	}

	// Test with different prefix cache scores to see impact
	baseRequest := PredictionRequest{
		KVCachePercentage:  0.6,
		InputTokenLength:   500,
		NumRequestWaiting:  3,
		NumRequestRunning:  2,
		NumTokensGenerated: 75,
	}

	prefixCacheScores := []float64{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
	ttftResults := make([]float64, 0, len(prefixCacheScores))
	quantileResults := make([]float64, 0, len(prefixCacheScores))

	for _, prefixScore := range prefixCacheScores {
		req := baseRequest
		req.PrefixCacheScore = prefixScore

		response, err := predictor.Predict(ctx, req)
		if err != nil {
			t.Errorf("Prediction failed for prefix cache score %.1f: %v", prefixScore, err)
			continue
		}

		ttftResults = append(ttftResults, response.TTFT)
		quantileResults = append(quantileResults, response.Quantile)
		t.Logf("Prefix cache %.0f%%: TTFT=%.2f ms, TPOT=%.2f ms, quantile=%.2f",
			prefixScore*100, response.TTFT, response.TPOT, response.Quantile)
	}

	// Verify quantile consistency
	if len(quantileResults) > 1 {
		firstQuantile := quantileResults[0]
		for i, q := range quantileResults {
			if abs(q-firstQuantile) > 0.01 {
				t.Errorf("Quantile inconsistency: prediction %d has quantile %.2f, expected %.2f",
					i, q, firstQuantile)
			}
		}
		t.Log("✓ Quantile values consistent across predictions")
	}

	// Analyze the relationship between prefix cache and TTFT
	if len(ttftResults) >= 2 {
		t.Log("Prefix cache impact analysis:")
		lowCacheTTFT := ttftResults[0]                   // 0% prefix cache
		highCacheTTFT := ttftResults[len(ttftResults)-1] // 100% prefix cache
		difference := highCacheTTFT - lowCacheTTFT

		t.Logf("  TTFT at 0%% prefix cache: %.2f ms", lowCacheTTFT)
		t.Logf("  TTFT at 100%% prefix cache: %.2f ms", highCacheTTFT)
		t.Logf("  Difference: %.2f ms", difference)

		if predictor.GetCurrentModelType() == bayesianRidgeModelType {
			// For Bayesian Ridge, we expect to see the linear relationship
			if difference > 5 {
				t.Logf("✓ Detected prefix cache impact: %.2f ms difference", difference)
			} else {
				t.Logf("ℹ Small prefix cache impact: %.2f ms difference", difference)
			}
		}
	}
}

func testHTTPFallbackPrediction(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing HTTP fallback prediction...")

	modelType := predictor.GetCurrentModelType()
	if modelType == bayesianRidgeModelType {
		t.Skip("HTTP fallback test not applicable for Bayesian Ridge")
	}

	// Test prediction with HTTP fallback including prefix cache score
	req := PredictionRequest{
		KVCachePercentage:  0.8, // 80% as a fraction
		InputTokenLength:   1024,
		NumRequestWaiting:  5,
		NumRequestRunning:  3,
		NumTokensGenerated: 150,
		PrefixCacheScore:   0.9, // 90% prefix cache hit rate
	}

	t.Logf("Making HTTP prediction request: %+v", req)

	response, err := predictor.Predict(ctx, req)
	if err != nil {
		t.Fatalf("HTTP prediction failed: %v", err)
	}

	t.Logf("HTTP Prediction Response:")
	t.Logf("  TTFT: %.2f ms", response.TTFT)
	t.Logf("  TPOT: %.2f ms", response.TPOT)
	t.Logf("  Model Type: %s", response.ModelType)
	t.Logf("  Quantile: %.2f", response.Quantile)
	t.Logf("  Prefix Cache Score Used: %.1f%%", req.PrefixCacheScore*100)

	// Validate that we got a reasonable response
	if response.TTFT <= 0 {
		t.Error("TTFT should be positive")
	}
	if response.TPOT <= 0 {
		t.Error("TPOT should be positive")
	}

	// The model type should indicate the correct type
	if response.ModelType == "" {
		t.Error("Model type should not be empty")
	}

	t.Logf("Successfully tested HTTP prediction with prefix cache")
}

func testPredictionPerformance(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing prediction performance (target: < 300ms) with prefix cache scores...")

	// Ensure predictor is ready
	if !predictor.IsReady() {
		t.Skip("Predictor not ready for performance test")
	}

	req := PredictionRequest{
		KVCachePercentage:  0.6, // 60% as a fraction
		InputTokenLength:   768,
		NumRequestWaiting:  2,
		NumRequestRunning:  1,
		NumTokensGenerated: 80,
		PrefixCacheScore:   0.7, // 70% prefix cache hit rate
	}

	// Warm up with a few predictions
	for i := 0; i < 3; i++ {
		_, err := predictor.Predict(ctx, req)
		if err != nil {
			t.Fatalf("Warmup prediction %d failed: %v", i+1, err)
		}
	}

	// Test multiple predictions and measure time
	const numTests = 10
	const avgDurationMs = 250

	var totalDuration time.Duration
	var maxSingleDuration time.Duration
	var minSingleDuration = time.Hour // Initialize to large value

	t.Logf("Running %d prediction performance tests...", numTests)

	for i := 0; i < numTests; i++ {
		// Vary prefix cache score for each test
		testReq := req
		testReq.PrefixCacheScore = float64(i) / float64(numTests-1) // 0.0 to 1.0

		start := time.Now()

		response, err := predictor.Predict(ctx, testReq)

		duration := time.Since(start)
		totalDuration += duration

		if err != nil {
			t.Errorf("Prediction %d failed: %v", i+1, err)
			continue
		}

		// Track min/max durations
		if duration > maxSingleDuration {
			maxSingleDuration = duration
		}
		if duration < minSingleDuration {
			minSingleDuration = duration
		}

		durationMs := float64(duration.Nanoseconds()) / 1e6
		t.Logf("Prediction %d: %.2fms - TTFT: %.1fms, TPOT: %.1fms (prefix: %.0f%%, quantile: %.2f)",
			i+1, durationMs, response.TTFT, response.TPOT, testReq.PrefixCacheScore*100, response.Quantile)
	}

	// Calculate statistics
	avgDuration := totalDuration / numTests
	avgMs := float64(avgDuration.Nanoseconds()) / 1e6
	maxMs := float64(maxSingleDuration.Nanoseconds()) / 1e6
	minMs := float64(minSingleDuration.Nanoseconds()) / 1e6

	t.Logf("Performance Results:")
	t.Logf("  Average: %.2fms", avgMs)
	t.Logf("  Minimum: %.2fms", minMs)
	t.Logf("  Maximum: %.2fms", maxMs)
	t.Logf("  Target:  < %dms", avgDurationMs)

	// Overall performance check
	if avgMs > avgDurationMs {
		t.Errorf("Average prediction time %.2fms exceeded target of %dms", avgMs, avgDurationMs)
	} else {
		t.Logf("✅ Performance target met: avg %.2fms < %dms", avgMs, avgDurationMs)
	}

	// Check for consistency (max shouldn't be too much higher than average)
	if maxMs > avgMs*3 {
		t.Logf("⚠️  High variance detected: max %.2fms is %.1fx the average", maxMs, maxMs/avgMs)
	} else {
		t.Logf("✅ Good consistency: max %.2fms is %.1fx the average", maxMs, maxMs/avgMs)
	}
}

func testBulkPredictionPerformance(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing bulk prediction performance...")

	if !predictor.IsReady() {
		t.Skip("Predictor not ready for bulk performance test")
	}

	// Create batch of prediction requests
	const batchSize = 20
	requests := make([]PredictionRequest, batchSize)
	for i := 0; i < batchSize; i++ {
		requests[i] = PredictionRequest{
			KVCachePercentage:  0.6 + float64(i%5)*0.05,           // Vary between 0.6 and 0.8
			InputTokenLength:   300 + i*10,                        // Vary input length
			NumRequestWaiting:  i % 4,                             // 0 to 3
			NumRequestRunning:  (i % 2) + 1,                       // 1 to 2
			NumTokensGenerated: 50 + i*2,                          // Vary generated tokens
			PrefixCacheScore:   float64(i) / float64(batchSize-1), // 0.0 to 1.0
		}
	}

	// Warm up
	warmupRequests := requests[:3]
	_, err := predictor.PredictBulk(ctx, warmupRequests)
	if err != nil {
		t.Fatalf("Warmup bulk prediction failed: %v", err)
	}

	// Performance test
	const numTests = 5
	var totalDuration time.Duration
	var totalRequests int
	var totalSuccessful int

	t.Logf("Running %d bulk prediction performance tests with %d requests each...", numTests, batchSize)

	for i := 0; i < numTests; i++ {
		start := time.Now()

		response, err := predictor.PredictBulk(ctx, requests)

		duration := time.Since(start)
		totalDuration += duration

		if err != nil {
			t.Errorf("Bulk prediction %d failed: %v", i+1, err)
			continue
		}

		totalRequests += response.TotalRequests
		totalSuccessful += response.SuccessfulPredictions

		durationMs := float64(duration.Nanoseconds()) / 1e6
		avgPerRequest := durationMs / float64(response.SuccessfulPredictions)

		t.Logf("Bulk test %d: %.2fms total, %.2fms per request (%d/%d successful)",
			i+1, durationMs, avgPerRequest, response.SuccessfulPredictions, response.TotalRequests)
	}

	// Calculate bulk performance statistics
	avgTotalDuration := totalDuration / numTests
	avgTotalMs := float64(avgTotalDuration.Nanoseconds()) / 1e6
	avgPerRequest := avgTotalMs / float64(batchSize)

	t.Logf("Bulk Performance Results:")
	t.Logf("  Average total time: %.2fms", avgTotalMs)
	t.Logf("  Average per request: %.2fms", avgPerRequest)
	t.Logf("  Success rate: %.1f%%", float64(totalSuccessful)/float64(totalRequests)*100)

	// Compare with single prediction performance target
	singlePredictionTarget := 250.0                         // ms
	bulkEfficiencyThreshold := singlePredictionTarget * 0.7 // Bulk should be more efficient

	switch {
	case avgPerRequest <= bulkEfficiencyThreshold:
		t.Logf("✅ Bulk predictions are efficient: %.2fms per request < %.2fms threshold",
			avgPerRequest, bulkEfficiencyThreshold)
	case avgPerRequest <= singlePredictionTarget:
		t.Logf("✓ Bulk predictions acceptable: %.2fms per request < %.2fms single target",
			avgPerRequest, singlePredictionTarget)
	default:
		t.Errorf("❌ Bulk predictions slow: %.2fms per request > %.2fms target",
			avgPerRequest, singlePredictionTarget)
	}
}

func testHTTPOnlyPerformance(t *testing.T, ctx context.Context) {
	t.Log("Testing HTTP-only prediction performance (no native XGBoost interference) with prefix cache...")

	predictionURLs := os.Getenv("PREDICTION_SERVER_URL")
	trainingURL := os.Getenv("TRAINING_SERVER_URL")
	if predictionURLs == "" {
		t.Skip("PREDICTION_SERVER_URL not set")
	}
	if trainingURL == "" {
		// Use first prediction URL as fallback
		urls := strings.Split(predictionURLs, ",")
		if len(urls) > 0 {
			trainingURL = strings.TrimSpace(urls[0])
		} else {
			t.Skip("No valid URLs available for testing")
		}
	}

	// Parse prediction URLs
	urls := strings.Split(predictionURLs, ",")
	parsedPredictionURLs := make([]string, 0, len(urls))
	for _, url := range urls {
		parsedPredictionURLs = append(parsedPredictionURLs, strings.TrimSpace(url))
	}

	// Create a dedicated HTTP-only predictor for clean performance testing
	zapLog, err := zap.NewDevelopment()
	if err != nil {
		t.Fatalf("Failed to create logger: %v", err)
	}
	logger := zapr.NewLogger(zapLog)

	httpOnlyConfig := &Config{
		TrainingURL:            trainingURL,
		PredictionURLs:         parsedPredictionURLs,
		MaxSampleSize:          1000,
		FlushInterval:          1 * time.Second, // Long interval to avoid interference
		MetricsRefreshInterval: 1 * time.Second, // Longer for metrics
		UseNativeXGBoost:       false,           // Force HTTP-only
		HTTPTimeout:            5 * time.Second, // Reasonable timeout
		MaxBulkSize:            50,
	}

	httpPredictor := New(httpOnlyConfig, logger)
	defer httpPredictor.Stop()

	err = httpPredictor.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start HTTP-only predictor: %v", err)
	}

	// Wait for readiness
	time.Sleep(1 * time.Second)

	// Wait for coefficients to be cached
	maxWaitTime := 10 * time.Second
	waitInterval := 200 * time.Millisecond
	elapsed := time.Duration(0)

	for elapsed < maxWaitTime {
		if httpPredictor.IsReady() {
			break
		}
		time.Sleep(waitInterval)
		elapsed += waitInterval
	}

	if !httpPredictor.IsReady() {
		t.Skip("model not ready yet")
	}

	req := PredictionRequest{
		KVCachePercentage:  0.65,
		InputTokenLength:   512,
		NumRequestWaiting:  1,
		NumRequestRunning:  2,
		NumTokensGenerated: 100,
		PrefixCacheScore:   0.75, // 75% prefix cache hit rate
	}

	// Warm up
	for i := 0; i < 2; i++ {
		_, err := httpPredictor.Predict(ctx, req)
		if err != nil {
			t.Fatalf("HTTP warmup prediction %d failed: %v", i+1, err)
		}
	}

	// Performance test
	const numTests = 15
	const targetMs = 250

	var durations []time.Duration
	var successful int

	t.Logf("Running %d HTTP-only prediction tests...", numTests)

	for i := 0; i < numTests; i++ {
		// Vary prefix cache for each test
		testReq := req
		testReq.PrefixCacheScore = 0.5 + (float64(i)/float64(numTests-1))*0.5 // 0.5 to 1.0

		start := time.Now()

		response, err := httpPredictor.Predict(ctx, testReq)

		duration := time.Since(start)
		durations = append(durations, duration)

		if err != nil {
			t.Errorf("HTTP prediction %d failed: %v", i+1, err)
			continue
		}

		successful++
		durationMs := float64(duration.Nanoseconds()) / 1e6

		status := "✅"

		t.Logf("%s Test %d: %.1fms (TTFT: %.0fms, TPOT: %.0fms, prefix: %.0f%%, quantile: %.2f)",
			status, i+1, durationMs, response.TTFT, response.TPOT, testReq.PrefixCacheScore*100, response.Quantile)
	}

	// Calculate statistics
	if len(durations) == 0 {
		t.Fatal("No successful predictions to analyze")
	}

	var total time.Duration
	min := durations[0]
	max := durations[0]

	for _, d := range durations {
		total += d
		if d < min {
			min = d
		}
		if d > max {
			max = d
		}
	}

	avg := total / time.Duration(len(durations))
	avgMs := float64(avg.Nanoseconds()) / 1e6
	minMs := float64(min.Nanoseconds()) / 1e6
	maxMs := float64(max.Nanoseconds()) / 1e6

	// Count fast predictions
	fastCount := 0
	for _, d := range durations {
		if float64(d.Nanoseconds())/1e6 <= targetMs {
			fastCount++
		}
	}

	t.Logf("\n📊 HTTP-Only Performance Summary:")
	t.Logf("  Success Rate: %d/%d (%.1f%%)", successful, numTests, float64(successful)/float64(numTests)*100)
	t.Logf("  Average: %.1fms", avgMs)
	t.Logf("  Minimum: %.1fms", minMs)
	t.Logf("  Maximum: %.1fms", maxMs)
	t.Logf("  Under %dms: %d/%d (%.1f%%)", targetMs, fastCount, len(durations), float64(fastCount)/float64(len(durations))*100)

	// Performance assertions
	if successful < numTests {
		t.Errorf("Some predictions failed: %d/%d successful", successful, numTests)
	}

	if avgMs <= targetMs {
		t.Logf("✅ PASS: Average response time %.1fms ≤ %dms target", avgMs, targetMs)
	} else {
		t.Errorf("❌ FAIL: Average response time %.1fms > %dms target", avgMs, targetMs)
	}

	// Check that at least 80% of requests are under target
	fastPercentage := float64(fastCount) / float64(len(durations)) * 100
	if fastPercentage >= 80 {
		t.Logf("✅ PASS: %.1f%% of requests under %dms (≥80%% target)", fastPercentage, targetMs)
	} else {
		t.Errorf("❌ FAIL: Only %.1f%% of requests under %dms (<80%% target)", fastPercentage, targetMs)
	}
}

func testHTTPOnlyPrediction(t *testing.T, ctx context.Context) {
	t.Log("Testing HTTP-only prediction (bypassing native XGBoost) with prefix cache...")

	// Create a predictor with native XGBoost disabled to force HTTP usage
	predictionURLs := os.Getenv("PREDICTION_SERVER_URL")
	trainingURL := os.Getenv("TRAINING_SERVER_URL")
	if predictionURLs == "" {
		t.Skip("PREDICTION_SERVER_URL not set")
	}
	if trainingURL == "" {
		// Use first prediction URL as fallback
		urls := strings.Split(predictionURLs, ",")
		if len(urls) > 0 {
			trainingURL = strings.TrimSpace(urls[0])
		} else {
			t.Skip("No valid URLs available for testing")
		}
	}

	// Parse prediction URLs
	urls := strings.Split(predictionURLs, ",")
	parsedPredictionURLs := make([]string, 0, len(urls))
	for _, url := range urls {
		parsedPredictionURLs = append(parsedPredictionURLs, strings.TrimSpace(url))
	}

	zapLog, err := zap.NewDevelopment()
	if err != nil {
		t.Fatalf("Failed to create logger: %v", err)
	}
	logger := zapr.NewLogger(zapLog)

	httpOnlyConfig := &Config{
		TrainingURL:            trainingURL,
		PredictionURLs:         parsedPredictionURLs,
		MaxSampleSize:          1000,
		FlushInterval:          1 * time.Second,
		MetricsRefreshInterval: 1 * time.Second, // Longer for metrics
		UseNativeXGBoost:       false,           // Force HTTP fallback
		HTTPTimeout:            30 * time.Second,
		MaxBulkSize:            25,
	}

	httpPredictor := New(httpOnlyConfig, logger)
	defer httpPredictor.Stop()

	err = httpPredictor.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start HTTP-only predictor: %v", err)
	}

	// Wait a moment for startup and coefficient caching
	time.Sleep(3 * time.Second)

	// Ensure coefficients are ready
	maxWait := 10 * time.Second
	waited := time.Duration(0)
	for waited < maxWait {
		if httpPredictor.IsReady() {
			break
		}
		time.Sleep(500 * time.Millisecond)
		waited += 500 * time.Millisecond
	}

	if !httpPredictor.IsReady() {
		t.Skip("Model not ready yet")
	}

	// Test prediction using HTTP only with prefix cache
	req := PredictionRequest{
		KVCachePercentage:  0.6, // 60% as a fraction
		InputTokenLength:   256,
		NumRequestWaiting:  1,
		NumRequestRunning:  2,
		NumTokensGenerated: 75,
		PrefixCacheScore:   0.85, // 85% prefix cache hit rate
	}

	t.Logf("Making HTTP-only prediction request: %+v", req)

	response, err := httpPredictor.Predict(ctx, req)
	if err != nil {
		t.Fatalf("HTTP-only prediction failed: %v", err)
	}

	t.Logf("HTTP-Only Prediction Response:")
	t.Logf("  TTFT: %.2f ms", response.TTFT)
	t.Logf("  TPOT: %.2f ms", response.TPOT)
	t.Logf("  Model Type: %s", response.ModelType)
	t.Logf("  Quantile: %.2f", response.Quantile)
	t.Logf("  TTFT Uncertainty: %.2f", response.TTFTUncertainty)
	t.Logf("  TPOT Uncertainty: %.2f", response.TPOTUncertainty)
	t.Logf("  Prefix Cache Score Used: %.1f%%", req.PrefixCacheScore*100)

	// Validate response
	if response.TTFT <= 0 {
		t.Error("TTFT should be positive")
	}
	if response.TPOT <= 0 {
		t.Error("TPOT should be positive")
	}

	// Test multiple HTTP-only predictions with varying prefix cache
	t.Log("Testing multiple HTTP-only predictions with different prefix cache scores...")
	for i := 0; i < 3; i++ {
		testReq := PredictionRequest{
			KVCachePercentage:  float64(30+i*20) / 100.0,
			InputTokenLength:   128 + i*256,
			NumRequestWaiting:  i,
			NumRequestRunning:  1,
			NumTokensGenerated: 25 + i*50,
			PrefixCacheScore:   float64(60+i*20) / 100.0, // 60%, 80%, 100%
		}

		resp, err := httpPredictor.Predict(ctx, testReq)
		if err != nil {
			t.Errorf("HTTP-only prediction %d failed: %v", i+1, err)
			continue
		}

		t.Logf("HTTP-only prediction %d: TTFT=%.2f, TPOT=%.2f (prefix: %.0f%%, quantile: %.2f)",
			i+1, resp.TTFT, resp.TPOT, testReq.PrefixCacheScore*100, resp.Quantile)
	}

	t.Log("Successfully tested HTTP-only predictions with prefix cache")
}

func testLoadBalancing(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing load balancing across multiple prediction URLs with prefix cache...")

	predictionURLs := predictor.GetPredictionURLs()
	if len(predictionURLs) <= 1 {
		t.Skip("Need multiple prediction URLs to test load balancing")
	}

	t.Logf("Testing load balancing across %d prediction URLs: %v", len(predictionURLs), predictionURLs)

	// Make multiple predictions to test load balancing
	const numPredictions = 20
	req := PredictionRequest{
		KVCachePercentage:  0.7,
		InputTokenLength:   512,
		NumRequestWaiting:  2,
		NumRequestRunning:  1,
		NumTokensGenerated: 100,
		PrefixCacheScore:   0.8, // 80% prefix cache hit rate
	}

	successfulPredictions := 0
	for i := 0; i < numPredictions; i++ {
		// Vary prefix cache score across requests
		testReq := req
		testReq.PrefixCacheScore = 0.5 + (float64(i)/float64(numPredictions-1))*0.5 // 0.5 to 1.0

		response, err := predictor.Predict(ctx, testReq)
		if err != nil {
			t.Logf("Prediction %d failed: %v", i+1, err)
			continue
		}

		successfulPredictions++
		t.Logf("Prediction %d: TTFT=%.2f, TPOT=%.2f (prefix: %.0f%%, quantile: %.2f)",
			i+1, response.TTFT, response.TPOT, testReq.PrefixCacheScore*100, response.Quantile)
	}

	successRate := float64(successfulPredictions) / float64(numPredictions) * 100
	t.Logf("Load balancing test results: %d/%d successful (%.1f%%)", successfulPredictions, numPredictions, successRate)

	if successRate < 80 {
		t.Errorf("Low success rate in load balancing test: %.1f%% < 80%%", successRate)
	} else {
		t.Logf("✅ Load balancing test successful with %.1f%% success rate", successRate)
	}
}

func testPrefixCacheValidation(t *testing.T, predictor *Predictor) {
	t.Log("Testing prefix cache score validation...")

	// Test valid prefix cache scores
	validScores := []float64{0.0, 0.25, 0.5, 0.75, 1.0}
	for _, score := range validScores {
		req := PredictionRequest{
			KVCachePercentage:  0.5,
			InputTokenLength:   100,
			NumRequestWaiting:  1,
			NumRequestRunning:  1,
			NumTokensGenerated: 10,
			PrefixCacheScore:   score,
		}

		err := predictor.ValidatePredictionRequest(req)
		if err != nil {
			t.Errorf("Valid prefix cache score %.2f should not cause validation error: %v", score, err)
		}
	}

	// Test invalid prefix cache scores
	invalidScores := []float64{-0.1, -1.0, 1.1, 2.0}
	for _, score := range invalidScores {
		req := PredictionRequest{
			KVCachePercentage:  0.5,
			InputTokenLength:   100,
			NumRequestWaiting:  1,
			NumRequestRunning:  1,
			NumTokensGenerated: 10,
			PrefixCacheScore:   score,
		}

		err := predictor.ValidatePredictionRequest(req)
		if err == nil {
			t.Errorf("Invalid prefix cache score %.2f should cause validation error", score)
		} else {
			t.Logf("✓ Invalid prefix cache score %.2f correctly rejected: %v", score, err)
		}
	}

	// Test training entry validation
	validEntry := TrainingEntry{
		KVCachePercentage:  0.6,
		InputTokenLength:   200,
		NumRequestWaiting:  2,
		NumRequestRunning:  1,
		NumTokensGenerated: 20,
		ActualTTFT:         50.0,
		ActualTPOT:         15.0,
		PrefixCacheScore:   0.8,
		Timestamp:          time.Now(),
	}

	err := predictor.ValidateTrainingEntry(validEntry)
	if err != nil {
		t.Errorf("Valid training entry should not cause validation error: %v", err)
	}

	// Test invalid training entry
	invalidEntry := validEntry
	invalidEntry.PrefixCacheScore = 1.5 // Invalid

	err = predictor.ValidateTrainingEntry(invalidEntry)
	if err == nil {
		t.Error("Invalid training entry should cause validation error")
	} else {
		t.Logf("✓ Invalid training entry correctly rejected: %v", err)
	}

	t.Log("✅ Prefix cache validation tests completed")
}

func testPredictionConstructors(t *testing.T) {
	t.Log("Testing prediction and training entry constructors with prefix cache...")

	// Test valid prediction request constructor
	req, err := NewPredictionRequest(
		0.7,  // kv_cache_percentage
		500,  // input_token_length
		3,    // num_request_waiting
		2,    // num_request_running
		100,  // num_tokens_generated
		0.85, // prefix_cache_score
	)
	if err != nil {
		t.Errorf("Valid prediction request constructor failed: %v", err)
	} else {
		t.Logf("✓ Created prediction request: TTFT features with %.0f%% prefix cache", req.PrefixCacheScore*100)
	}

	// Test invalid prediction request constructor
	_, err = NewPredictionRequest(
		0.7, // kv_cache_percentage
		500, // input_token_length
		3,   // num_request_waiting
		2,   // num_request_running
		100, // num_tokens_generated
		1.5, // prefix_cache_score (invalid)
	)
	if err == nil {
		t.Error("Invalid prediction request constructor should have failed")
	} else {
		t.Logf("✓ Invalid prediction request correctly rejected: %v", err)
	}

	// Test valid training entry constructor
	entry, err := NewTrainingEntry(
		0.6,  // kv_cache_percentage
		300,  // input_token_length
		2,    // num_request_waiting
		1,    // num_request_running
		50,   // num_tokens_generated
		45.5, // actual_ttft_ms
		12.3, // actual_tpot_ms
		0.75, // prefix_cache_score
	)
	if err != nil {
		t.Errorf("Valid training entry constructor failed: %v", err)
	} else {
		t.Logf("✓ Created training entry: TTFT=%.1fms, TPOT=%.1fms, prefix cache=%.0f%%",
			entry.ActualTTFT, entry.ActualTPOT, entry.PrefixCacheScore*100)
	}

	// Test invalid training entry constructor
	_, err = NewTrainingEntry(
		0.6,  // kv_cache_percentage
		300,  // input_token_length
		2,    // num_request_waiting
		1,    // num_request_running
		50,   // num_tokens_generated
		45.5, // actual_ttft_ms
		12.3, // actual_tpot_ms
		-0.1, // prefix_cache_score (invalid)
	)
	if err == nil {
		t.Error("Invalid training entry constructor should have failed")
	} else {
		t.Logf("✓ Invalid training entry correctly rejected: %v", err)
	}

	t.Log("✅ Constructor validation tests completed")
}

func testXGBoostJSONStructure(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing XGBoost JSON structure from server...")

	if predictor.GetCurrentModelType() != xgBoostModelType {
		t.Skip("This test is specific to XGBoost model type")
	}

	// Get raw trees to examine structure
	trees, err := predictor.GetXGBoostTrees(ctx)
	if err != nil {
		t.Fatalf("Failed to get XGBoost trees: %v", err)
	}

	if len(trees.TTFTTrees) == 0 {
		t.Fatal("No TTFT trees available")
	}

	// Examine the first tree structure
	firstTree := trees.TTFTTrees[0]
	t.Logf("First TTFT tree structure: %T", firstTree)

	// Convert to map to examine fields
	if treeMap, ok := firstTree.(map[string]interface{}); ok {
		t.Log("First tree fields:")
		for key, value := range treeMap {
			switch key {
			case "split":
				t.Logf("  %s: %T = %v", key, value, value)
			case "children":
				if value != nil {
					if children, ok := value.([]interface{}); ok {
						// This is the unique execution path for valid children
						t.Logf("  %s: []interface{} with %d children", key, len(children))
						// Examine first child
						if len(children) > 0 {
							if childMap, ok := children[0].(map[string]interface{}); ok {
								for childKey, childValue := range childMap {
									if childKey == "split" {
										t.Logf("    child[0].%s: %T = %v", childKey, childValue, childValue)
									}
								}
							}
						}
					} else {
						// Fallback if type assertion fails
						t.Logf("  %s: %T = %v", key, value, value)
					}
				} else {
					// Fallback if value is nil
					t.Logf("  %s: %T = %v", key, value, value)
				}
			default:
				t.Logf("  %s: %T = %v", key, value, value)
			}
		}
	}

	// Try to understand why the conversion is failing
	t.Log("Analyzing conversion issue...")
	if len(trees.TTFTTrees) > 0 {
		// Test the conversion function manually
		testConvertXGBoostJSON(t, trees.TTFTTrees[0])
	}

	t.Log("XGBoost JSON structure analysis complete")
}

// Helper function to test the conversion logic
func testConvertXGBoostJSON(t *testing.T, tree interface{}) {
	featureMap := map[string]int{
		"kv_cache_percentage":  0,
		"input_token_length":   1,
		"num_request_waiting":  2,
		"num_request_running":  3,
		"num_tokens_generated": 4,
		"prefix_cache_score":   5, // Added prefix cache score mapping
	}

	t.Log("Testing XGBoost JSON conversion...")

	treeMap, ok := tree.(map[string]interface{})
	if !ok {
		t.Log("Tree is not a map[string]interface{}")
		return
	}

	// Check if split field exists and what type it is
	if split, exists := treeMap["split"]; exists {
		t.Logf("Split field exists: %T = %v", split, split)

		switch splitVal := split.(type) {
		case string:
			t.Logf("Split is string: '%s'", splitVal)
			if featureIdx, found := featureMap[splitVal]; found {
				t.Logf("Found feature index for '%s': %d", splitVal, featureIdx)
			} else {
				t.Logf("Feature '%s' not found in feature map", splitVal)
			}
		case float64:
			t.Logf("Split is float64: %v (already numeric, no conversion needed)", splitVal)
		case int:
			t.Logf("Split is int: %v (already numeric, no conversion needed)", splitVal)
		default:
			t.Logf("Split is unexpected type: %T = %v", splitVal, splitVal)
		}
	} else {
		t.Log("Split field does not exist")
	}
}

func testMetricsRetrieval(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing metrics retrieval...")

	modelType := predictor.GetCurrentModelType()
	quantile := predictor.GetCurrentQuantile()
	t.Logf("Testing metrics for model type: %s, quantile: %.2f", modelType, quantile)

	switch modelType {
	case bayesianRidgeModelType:
		testBayesianRidgeMetrics(t, ctx, predictor)
	case xgBoostModelType:
		testXGBoostMetrics(t, ctx, predictor)
	case gbmModelType:
		testLightGBMMetrics(t, ctx, predictor)
	default:
		t.Logf("Unknown model type %s, testing cached metrics only", modelType)
	}

	// Test cached metrics
	cachedMetrics, hasCached := predictor.GetCachedMetrics()
	if hasCached {
		t.Logf("Cached metrics available - Model Type: %s", cachedMetrics.ModelType)
		if len(cachedMetrics.RawMetrics) > 0 {
			t.Logf("Raw metrics length: %d characters", len(cachedMetrics.RawMetrics))
		}
	} else {
		t.Log("No cached metrics available")
	}

	// Test readiness status
	t.Logf("Predictor readiness status:")
	t.Logf("  Overall Ready: %t", predictor.IsReady())
	t.Logf("  XGBoost Ready: %t", predictor.IsXGBoostReady())
	t.Logf("  LightGBM Ready: %t", predictor.IsLightGBMReady())
	t.Logf("  Bayesian Ridge Ready: %t", predictor.IsBayesianRidgeReady())
}

func testBayesianRidgeMetrics(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing Bayesian Ridge specific metrics with prefix cache support...")

	metrics, err := predictor.GetMetrics(ctx)
	if err != nil {
		t.Errorf("Failed to get Bayesian Ridge metrics: %v", err)
		return
	}

	if metrics.Coefficients == nil {
		t.Error("Bayesian Ridge coefficients should not be nil")
		return
	}

	t.Logf("TTFT Coefficients (should include prefix_cache_score):")
	t.Logf("  Intercept: %.6f", metrics.Coefficients.TTFTIntercept)
	for feature, coeff := range metrics.Coefficients.TTFTCoeffs {
		t.Logf("  %s: %.6f", feature, coeff)
	}

	t.Logf("TPOT Coefficients (should NOT include prefix_cache_score):")
	t.Logf("  Intercept: %.6f", metrics.Coefficients.TPOTIntercept)
	for feature, coeff := range metrics.Coefficients.TPOTCoeffs {
		t.Logf("  %s: %.6f", feature, coeff)
	}

	// Validate prefix cache score is in TTFT but not TPOT
	if _, hasPrefixCache := metrics.Coefficients.TTFTCoeffs["prefix_cache_score"]; hasPrefixCache {
		t.Log("✓ TTFT model includes prefix_cache_score coefficient")
	} else {
		t.Log("ℹ TTFT model does not include prefix_cache_score coefficient (may not be trained yet)")
	}

	if _, hasPrefixCache := metrics.Coefficients.TPOTCoeffs["prefix_cache_score"]; hasPrefixCache {
		t.Error("❌ TPOT model should NOT include prefix_cache_score coefficient")
	} else {
		t.Log("✓ TPOT model correctly excludes prefix_cache_score coefficient")
	}

	// Test individual coefficient and bucket retrieval
	coeffs, err := predictor.GetModelCoefficients(ctx)
	if err != nil {
		t.Errorf("Failed to get model coefficients: %v", err)
	} else {
		t.Logf("Retrieved coefficients separately: %d TTFT, %d TPOT features",
			len(coeffs.TTFTCoeffs), len(coeffs.TPOTCoeffs))
	}

	buckets, err := predictor.GetBucketCounts(ctx)
	if err != nil {
		t.Errorf("Failed to get bucket counts: %v", err)
	} else {
		t.Logf("Retrieved bucket counts: %d TTFT, %d TPOT buckets",
			len(buckets.TTFTBuckets), len(buckets.TPOTBuckets))
	}
}

func testXGBoostMetrics(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing XGBoost specific metrics...")

	// Wait a bit for XGBoost models to potentially load
	time.Sleep(3 * time.Second)

	trees, err := predictor.GetXGBoostTrees(ctx)
	if err != nil {
		t.Errorf("Failed to get XGBoost trees: %v", err)
		return
	}

	t.Logf("XGBoost Trees:")
	t.Logf("  TTFT Trees: %d", len(trees.TTFTTrees))
	t.Logf("  TPOT Trees: %d", len(trees.TPOTTrees))

	if len(trees.TTFTTrees) == 0 {
		t.Error("Expected at least one TTFT tree")
	}
	if len(trees.TPOTTrees) == 0 {
		t.Error("Expected at least one TPOT tree")
	}

	// Test native XGBoost readiness
	if predictor.IsXGBoostReady() {
		t.Log("Native XGBoost models are ready for local prediction")
	} else {
		t.Log("Native XGBoost models not ready, will use HTTP fallback")
	}
}

func testLightGBMMetrics(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing LightGBM specific metrics...")

	// For LightGBM, we primarily use HTTP calls, so test the HTTP connectivity
	if predictor.IsLightGBMReady() {
		t.Log("LightGBM models are ready via HTTP")

		// Test a simple prediction to ensure the HTTP endpoint works
		req := PredictionRequest{
			KVCachePercentage:  0.6,
			InputTokenLength:   300,
			NumRequestWaiting:  1,
			NumRequestRunning:  1,
			NumTokensGenerated: 50,
			PrefixCacheScore:   0.7,
		}

		_, err := predictor.Predict(ctx, req)
		if err != nil {
			t.Errorf("LightGBM test prediction failed: %v", err)
		} else {
			t.Log("✓ LightGBM HTTP prediction working")
		}
	} else {
		t.Log("LightGBM models not ready")
	}
}

// generateTrainingEntries creates random training data for testing with prefix cache scores
func generateTrainingEntries(count int) []TrainingEntry {
	entries := make([]TrainingEntry, count)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	for i := 0; i < count; i++ {
		// Generate TTFT and TPOT using a simple equation based on features, plus some noise
		kv := rng.Float64() // 0.0 to 1.0
		inputLen := rng.Intn(2048) + 1
		waiting := rng.Intn(20)
		running := rng.Intn(10) + 1
		generated := rng.Intn(500) + 1
		prefixCache := rng.Float64() // 0.0 to 1.0

		// Updated equations to include prefix cache impact on TTFT:
		// TTFT includes prefix cache, TPOT does not
		ttft := 100 + 2*float64(inputLen) + 10*kv + 5*float64(waiting) + 30*prefixCache + rng.NormFloat64()*20
		tpot := 20 + 0.5*float64(generated) + 2*float64(running) + rng.NormFloat64()*5 + 9*kv

		entries[i] = TrainingEntry{
			KVCachePercentage:  kv,
			InputTokenLength:   inputLen,
			NumRequestWaiting:  waiting,
			NumRequestRunning:  running,
			NumTokensGenerated: generated,
			ActualTTFT:         ttft,
			ActualTPOT:         tpot,
			PrefixCacheScore:   prefixCache, // Added prefix cache score
			Timestamp:          time.Now().Add(-time.Duration(rng.Intn(3600)) * time.Second),
		}
	}

	return entries
}

// Benchmark test for prediction performance with prefix cache
func BenchmarkPrediction(b *testing.B) {
	predictionURLs := os.Getenv("PREDICTION_SERVER_URL")
	trainingURL := os.Getenv("TRAINING_SERVER_URL")
	if predictionURLs == "" {
		b.Skip("PREDICTION_SERVER_URL not set, skipping benchmark")
	}
	if trainingURL == "" {
		// Use first prediction URL as fallback
		urls := strings.Split(predictionURLs, ",")
		if len(urls) > 0 {
			trainingURL = strings.TrimSpace(urls[0])
		} else {
			b.Skip("No valid URLs available for benchmarking")
		}
	}

	// Parse prediction URLs
	urls := strings.Split(predictionURLs, ",")
	parsedPredictionURLs := make([]string, 0, len(urls))
	for _, url := range urls {
		parsedPredictionURLs = append(parsedPredictionURLs, strings.TrimSpace(url))
	}

	logger := logr.Discard() // Silent logger for benchmark
	config := &Config{
		TrainingURL:            trainingURL,
		PredictionURLs:         parsedPredictionURLs,
		MaxSampleSize:          1000,
		FlushInterval:          1 * time.Second, // Long interval for benchmark
		MetricsRefreshInterval: 1 * time.Second,
		UseNativeXGBoost:       true,
		HTTPTimeout:            10 * time.Second,
		MaxBulkSize:            100,
	}

	predictor := New(config, logger)
	defer predictor.Stop()

	ctx := context.Background()
	err := predictor.Start(ctx)
	if err != nil {
		b.Fatalf("Failed to start predictor: %v", err)
	}

	// Wait for predictor to be ready
	for i := 0; i < 100; i++ {
		if predictor.IsReady() {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	req := PredictionRequest{
		KVCachePercentage:  0.75, // 75% as a fraction
		InputTokenLength:   512,
		NumRequestWaiting:  2,
		NumRequestRunning:  1,
		NumTokensGenerated: 100,
		PrefixCacheScore:   0.8, // 80% prefix cache hit rate
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := predictor.Predict(ctx, req)
			if err != nil {
				b.Errorf("Prediction failed: %v", err)
			}
		}
	})
}

// Benchmark test for bulk prediction performance
func BenchmarkBulkPrediction(b *testing.B) {
	predictionURLs := os.Getenv("PREDICTION_SERVER_URL")
	trainingURL := os.Getenv("TRAINING_SERVER_URL")
	if predictionURLs == "" {
		b.Skip("PREDICTION_SERVER_URL not set, skipping benchmark")
	}
	if trainingURL == "" {
		urls := strings.Split(predictionURLs, ",")
		if len(urls) > 0 {
			trainingURL = strings.TrimSpace(urls[0])
		} else {
			b.Skip("No valid URLs available for benchmarking")
		}
	}

	// Parse prediction URLs
	urls := strings.Split(predictionURLs, ",")
	parsedPredictionURLs := make([]string, 0, len(urls))
	for _, url := range urls {
		parsedPredictionURLs = append(parsedPredictionURLs, strings.TrimSpace(url))
	}

	logger := logr.Discard()
	config := &Config{
		TrainingURL:            trainingURL,
		PredictionURLs:         parsedPredictionURLs,
		MaxSampleSize:          1000,
		FlushInterval:          1 * time.Second,
		MetricsRefreshInterval: 1 * time.Second,
		UseNativeXGBoost:       true,
		HTTPTimeout:            10 * time.Second,
		MaxBulkSize:            50,
	}

	predictor := New(config, logger)
	defer predictor.Stop()

	ctx := context.Background()
	err := predictor.Start(ctx)
	if err != nil {
		b.Fatalf("Failed to start predictor: %v", err)
	}

	for i := 0; i < 100; i++ {
		if predictor.IsReady() {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Create batch of requests
	const batchSize = 20
	requests := make([]PredictionRequest, batchSize)
	for i := 0; i < batchSize; i++ {
		requests[i] = PredictionRequest{
			KVCachePercentage:  0.6 + float64(i%5)*0.05,
			InputTokenLength:   300 + i*10,
			NumRequestWaiting:  i % 4,
			NumRequestRunning:  (i % 2) + 1,
			NumTokensGenerated: 50 + i*2,
			PrefixCacheScore:   float64(i) / float64(batchSize-1),
		}
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := predictor.PredictBulk(ctx, requests)
			if err != nil {
				b.Errorf("Bulk prediction failed: %v", err)
			}
		}
	})
}

// Test to verify config loading from environment
func TestConfigFromEnv(t *testing.T) {
	// Save original env vars
	originalLatencyURL := os.Getenv("PREDICTION_SERVER_URL")
	originalTrainingURL := os.Getenv("TRAINING_SERVER_URL")
	originalSample := os.Getenv("LATENCY_MAX_SAMPLE_SIZE")
	originalInterval := os.Getenv("LATENCY_FLUSH_INTERVAL_SEC")
	originalNative := os.Getenv("LATENCY_USE_NATIVE_XGBOOST")
	originalTimeout := os.Getenv("LATENCY_HTTP_TIMEOUT_SEC")
	originalBulkSize := os.Getenv("LATENCY_MAX_BULK_SIZE")

	// Set test env vars
	os.Setenv("PREDICTION_SERVER_URL", "http://pred1.example.com,http://pred2.example.com,http://pred3.example.com")
	os.Setenv("TRAINING_SERVER_URL", "http://training.example.com")
	os.Setenv("LATENCY_MAX_SAMPLE_SIZE", "500")
	os.Setenv("LATENCY_FLUSH_INTERVAL_SEC", "5")
	os.Setenv("LATENCY_USE_NATIVE_XGBOOST", "false")
	os.Setenv("LATENCY_HTTP_TIMEOUT_SEC", "20")
	os.Setenv("LATENCY_MAX_BULK_SIZE", "75")

	defer func() {
		// Restore original env vars (handle empty strings properly)
		if originalLatencyURL != "" {
			os.Setenv("PREDICTION_SERVER_URL", originalLatencyURL)
		} else {
			os.Unsetenv("PREDICTION_SERVER_URL")
		}
		if originalTrainingURL != "" {
			os.Setenv("TRAINING_SERVER_URL", originalTrainingURL)
		} else {
			os.Unsetenv("TRAINING_SERVER_URL")
		}
		if originalSample != "" {
			os.Setenv("LATENCY_MAX_SAMPLE_SIZE", originalSample)
		} else {
			os.Unsetenv("LATENCY_MAX_SAMPLE_SIZE")
		}
		if originalInterval != "" {
			os.Setenv("LATENCY_FLUSH_INTERVAL_SEC", originalInterval)
		} else {
			os.Unsetenv("LATENCY_FLUSH_INTERVAL_SEC")
		}
		if originalNative != "" {
			os.Setenv("LATENCY_USE_NATIVE_XGBOOST", originalNative)
		} else {
			os.Unsetenv("LATENCY_USE_NATIVE_XGBOOST")
		}
		if originalTimeout != "" {
			os.Setenv("LATENCY_HTTP_TIMEOUT_SEC", originalTimeout)
		} else {
			os.Unsetenv("LATENCY_HTTP_TIMEOUT_SEC")
		}
		if originalBulkSize != "" {
			os.Setenv("LATENCY_MAX_BULK_SIZE", originalBulkSize)
		} else {
			os.Unsetenv("LATENCY_MAX_BULK_SIZE")
		}
	}()

	config := ConfigFromEnv()

	// Test training URL
	if config.TrainingURL != "http://training.example.com" {
		t.Errorf("Expected TrainingURL to be 'http://training.example.com', got '%s'", config.TrainingURL)
	}

	// Test prediction URLs
	expectedPredictionURLs := []string{
		"http://pred1.example.com",
		"http://pred2.example.com",
		"http://pred3.example.com",
	}
	if len(config.PredictionURLs) != len(expectedPredictionURLs) {
		t.Errorf("Expected %d prediction URLs, got %d", len(expectedPredictionURLs), len(config.PredictionURLs))
	}
	for i, expected := range expectedPredictionURLs {
		if i >= len(config.PredictionURLs) || config.PredictionURLs[i] != expected {
			t.Errorf("Expected PredictionURLs[%d] to be '%s', got '%s'", i, expected, config.PredictionURLs[i])
		}
	}

	// Test other config values
	if config.MaxSampleSize != 500 {
		t.Errorf("Expected MaxSampleSize to be 500, got %d", config.MaxSampleSize)
	}
	if config.FlushInterval != 5*time.Second {
		t.Errorf("Expected FlushInterval to be 5s, got %v", config.FlushInterval)
	}
	if config.MetricsRefreshInterval != 60*time.Second {
		t.Errorf("Expected MetricsRefreshInterval to be 60s, got %v", config.MetricsRefreshInterval)
	}
	if config.UseNativeXGBoost != false {
		t.Errorf("Expected UseNativeXGBoost to be false, got %t", config.UseNativeXGBoost)
	}
	if config.HTTPTimeout != 20*time.Second {
		t.Errorf("Expected HTTPTimeout to be 20s, got %v", config.HTTPTimeout)
	}
	if config.MaxBulkSize != 75 {
		t.Errorf("Expected MaxBulkSize to be 75, got %d", config.MaxBulkSize)
	}
}

// Helper function for absolute value
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// Test comprehensive bulk prediction functionality
func TestBulkPredictionValidation(t *testing.T) {
	t.Log("Testing bulk prediction validation...")

	predictor := &Predictor{
		config: &Config{MaxBulkSize: 5},
	}

	// Test empty request list
	_, err := predictor.PredictBulk(context.Background(), []PredictionRequest{})
	if err == nil {
		t.Error("Expected error for empty request list")
	} else {
		t.Logf("✓ Correctly rejected empty request list: %v", err)
	}

	// Test oversized request list
	oversizedRequests := make([]PredictionRequest, 10) // Over the limit of 5
	for i := range oversizedRequests {
		oversizedRequests[i] = PredictionRequest{
			KVCachePercentage:  0.5,
			InputTokenLength:   100,
			NumRequestWaiting:  1,
			NumRequestRunning:  1,
			NumTokensGenerated: 10,
			PrefixCacheScore:   0.5,
		}
	}

	_, err = predictor.PredictBulk(context.Background(), oversizedRequests)
	if err == nil {
		t.Error("Expected error for oversized request list")
	} else {
		t.Logf("✓ Correctly rejected oversized request list: %v", err)
	}

	// Test invalid request in the list
	invalidRequests := []PredictionRequest{
		{
			KVCachePercentage:  0.5,
			InputTokenLength:   100,
			NumRequestWaiting:  1,
			NumRequestRunning:  1,
			NumTokensGenerated: 10,
			PrefixCacheScore:   0.5,
		},
		{
			KVCachePercentage:  1.5, // Invalid
			InputTokenLength:   100,
			NumRequestWaiting:  1,
			NumRequestRunning:  1,
			NumTokensGenerated: 10,
			PrefixCacheScore:   0.5,
		},
	}

	_, err = predictor.PredictBulk(context.Background(), invalidRequests)
	if err == nil {
		t.Error("Expected error for invalid request in list")
	} else {
		t.Logf("✓ Correctly rejected list with invalid request: %v", err)
	}

	t.Log("✅ Bulk prediction validation tests completed")
}

// Test comprehensive prefix cache integration
func TestPrefixCacheIntegration(t *testing.T) {
	t.Log("Testing comprehensive prefix cache integration...")

	// Test all components work together with prefix cache
	zapLog, err := zap.NewDevelopment()
	if err != nil {
		t.Fatalf("Failed to create logger: %v", err)
	}
	logger := zapr.NewLogger(zapLog)

	// Use a minimal config that doesn't require network calls
	config := &Config{
		TrainingURL:            "http://mock-training.local",
		PredictionURLs:         []string{"http://mock-prediction.local"},
		MaxSampleSize:          100,
		FlushInterval:          1 * time.Hour, // Very long interval to avoid network calls
		MetricsRefreshInterval: 1 * time.Hour, // Very long interval to avoid network calls
		UseNativeXGBoost:       false,
		HTTPTimeout:            5 * time.Second,
		MaxBulkSize:            10,
	}

	predictor := New(config, logger)

	// Manually stop background processes without triggering final flush/refresh
	defer func() {
		// Stop background loop without calling Stop() which does final flush
		close(predictor.done)
		predictor.wg.Wait()
		t.Log("Background processes stopped without network calls")
	}()

	// Test training entries with prefix cache can be created and validated
	entries := make([]TrainingEntry, 5)
	for i := 0; i < 5; i++ {
		entry, err := NewTrainingEntry(
			float64(i)/10.0,   // kv_cache_percentage
			100+i*50,          // input_token_length
			i%3,               // num_request_waiting
			1,                 // num_request_running (always > 0)
			10+i*5,            // num_tokens_generated
			50.0+float64(i)*5, // actual_ttft_ms
			10.0+float64(i)*2, // actual_tpot_ms
			float64(i)/4.0,    // prefix_cache_score (0.0 to 1.0)
		)
		if err != nil {
			t.Fatalf("Failed to create training entry %d: %v", i, err)
		}
		entries[i] = entry

		t.Logf("Entry %d: prefix_cache=%.1f%%, ttft=%.1f, tpot=%.1f",
			i, entry.PrefixCacheScore*100, entry.ActualTTFT, entry.ActualTPOT)
	}

	// Add training data to buffer (won't flush due to long interval)
	err = predictor.AddTrainingDataBulk(entries)
	if err != nil {
		t.Fatalf("Failed to add training entries: %v", err)
	}
	t.Log("✓ Successfully added training entries with prefix cache scores to buffer")

	// Test prediction requests with prefix cache can be created and validated
	requests := make([]PredictionRequest, 3)
	for i := 0; i < 3; i++ {
		req, err := NewPredictionRequest(
			float64(i*20)/100.0, // kv_cache_percentage
			200+i*100,           // input_token_length
			i%2,                 // num_request_waiting
			1,                   // num_request_running (always > 0)
			20+i*10,             // num_tokens_generated
			float64(i)/2.0,      // prefix_cache_score
		)
		if err != nil {
			t.Fatalf("Failed to create prediction request %d: %v", i, err)
		}
		requests[i] = req

		err = predictor.ValidatePredictionRequest(req)
		if err != nil {
			t.Errorf("Valid prediction request %d failed validation: %v", i, err)
		}

		t.Logf("Request %d: prefix_cache=%.1f%%, kv_cache=%.1f%%, input_len=%d",
			i, req.PrefixCacheScore*100, req.KVCachePercentage*100, req.InputTokenLength)
	}
	t.Log("✓ Successfully created and validated prediction requests with prefix cache")

	// Test validation edge cases
	edgeCases := []struct {
		name        string
		prefixCache float64
		shouldPass  bool
	}{
		{"Zero prefix cache", 0.0, true},
		{"Max prefix cache", 1.0, true},
		{"Negative prefix cache", -0.1, false},
		{"Over-max prefix cache", 1.1, false},
	}

	for _, tc := range edgeCases {
		req := PredictionRequest{
			KVCachePercentage:  0.5,
			InputTokenLength:   100,
			NumRequestWaiting:  1,
			NumRequestRunning:  1,
			NumTokensGenerated: 10,
			PrefixCacheScore:   tc.prefixCache,
		}

		err := predictor.ValidatePredictionRequest(req)
		switch {
		case tc.shouldPass && err != nil:
			t.Errorf("Edge case '%s' should pass but failed: %v", tc.name, err)
		case !tc.shouldPass && err == nil:
			t.Errorf("Edge case '%s' should fail but passed", tc.name)
		default:
			t.Logf("✓ Edge case '%s' handled correctly", tc.name)
		}
	}

	// Test that we can access the configuration
	t.Logf("Configuration validation:")
	t.Logf("  Training URL: %s", predictor.GetTrainingURL())
	t.Logf("  Prediction URLs: %v", predictor.GetPredictionURLs())
	t.Logf("  Max Bulk Size: %d", config.MaxBulkSize)

	// Validate configuration consistency
	if len(predictor.GetPredictionURLs()) != len(config.PredictionURLs) {
		t.Errorf("Prediction URLs mismatch: expected %d, got %d",
			len(config.PredictionURLs), len(predictor.GetPredictionURLs()))
	}

	if predictor.GetTrainingURL() != config.TrainingURL {
		t.Errorf("Training URL mismatch: expected %s, got %s",
			config.TrainingURL, predictor.GetTrainingURL())
	}

	// Test data structure integrity
	t.Log("Validating data structure integrity...")

	// Check that training entries maintain their prefix cache scores
	for i, entry := range entries {
		expectedPrefixCache := float64(i) / 4.0
		if abs(entry.PrefixCacheScore-expectedPrefixCache) > 0.001 {
			t.Errorf("Training entry %d prefix cache score mismatch: expected %.3f, got %.3f",
				i, expectedPrefixCache, entry.PrefixCacheScore)
		}
	}

	// Check that prediction requests maintain their prefix cache scores
	for i, req := range requests {
		expectedPrefixCache := float64(i) / 2.0
		if abs(req.PrefixCacheScore-expectedPrefixCache) > 0.001 {
			t.Errorf("Prediction request %d prefix cache score mismatch: expected %.3f, got %.3f",
				i, expectedPrefixCache, req.PrefixCacheScore)
		}
	}

	// Verify that training data is properly buffered (not flushed due to long interval)
	predictor.bufferMu.Lock()
	bufferedCount := len(predictor.pending)
	predictor.bufferMu.Unlock()

	if bufferedCount != len(entries) {
		t.Errorf("Expected %d buffered entries, got %d", len(entries), bufferedCount)
	} else {
		t.Logf("✓ Training data properly buffered: %d entries", bufferedCount)
	}

	t.Log("✅ Comprehensive prefix cache integration test completed (offline mode)")
}

// Test offline validation functionality without network dependencies
func TestOfflineValidation(t *testing.T) {
	t.Log("Testing offline validation functionality...")

	// Create a minimal predictor for validation testing
	predictor := &Predictor{}

	// Test prediction request validation
	t.Run("PredictionRequestValidation", func(t *testing.T) {
		validReq := PredictionRequest{
			KVCachePercentage:  0.7,
			InputTokenLength:   500,
			NumRequestWaiting:  2,
			NumRequestRunning:  1,
			NumTokensGenerated: 80,
			PrefixCacheScore:   0.85,
		}

		err := predictor.ValidatePredictionRequest(validReq)
		if err != nil {
			t.Errorf("Valid prediction request failed validation: %v", err)
		}

		// Test invalid requests
		invalidTests := []struct {
			name string
			req  PredictionRequest
		}{
			{
				name: "Negative KV cache",
				req: PredictionRequest{
					KVCachePercentage:  -0.1,
					InputTokenLength:   100,
					NumRequestWaiting:  1,
					NumRequestRunning:  1,
					NumTokensGenerated: 10,
					PrefixCacheScore:   0.5,
				},
			},
			{
				name: "Invalid prefix cache",
				req: PredictionRequest{
					KVCachePercentage:  0.5,
					InputTokenLength:   100,
					NumRequestWaiting:  1,
					NumRequestRunning:  1,
					NumTokensGenerated: 10,
					PrefixCacheScore:   1.5,
				},
			},
		}

		for _, test := range invalidTests {
			err := predictor.ValidatePredictionRequest(test.req)
			if err == nil {
				t.Errorf("Invalid request '%s' should have failed validation", test.name)
			} else {
				t.Logf("✓ '%s' correctly rejected: %v", test.name, err)
			}
		}
	})

	// Test training entry validation
	t.Run("TrainingEntryValidation", func(t *testing.T) {
		validEntry := TrainingEntry{
			KVCachePercentage:  0.6,
			InputTokenLength:   300,
			NumRequestWaiting:  1,
			NumRequestRunning:  1,
			NumTokensGenerated: 50,
			ActualTTFT:         45.5,
			ActualTPOT:         12.3,
			PrefixCacheScore:   0.75,
			Timestamp:          time.Now(),
		}

		err := predictor.ValidateTrainingEntry(validEntry)
		if err != nil {
			t.Errorf("Valid training entry failed validation: %v", err)
		}

		// Test invalid entry
		invalidEntry := validEntry
		invalidEntry.PrefixCacheScore = -0.5

		err = predictor.ValidateTrainingEntry(invalidEntry)
		if err == nil {
			t.Error("Invalid training entry should have failed validation")
		} else {
			t.Logf("✓ Invalid training entry correctly rejected: %v", err)
		}
	})

	// Test constructor functions
	t.Run("ConstructorFunctions", func(t *testing.T) {
		// Test valid constructors
		_, err := NewPredictionRequest(0.5, 100, 1, 1, 10, 0.8)
		if err != nil {
			t.Errorf("Valid prediction request constructor failed: %v", err)
		}

		_, err = NewTrainingEntry(0.5, 100, 1, 1, 10, 30.0, 8.0, 0.8)
		if err != nil {
			t.Errorf("Valid training entry constructor failed: %v", err)
		}

		// Test invalid constructors
		_, err = NewPredictionRequest(0.5, 100, 1, 1, 10, 1.5) // Invalid prefix cache
		if err == nil {
			t.Error("Invalid prediction request constructor should have failed")
		}

		_, err = NewTrainingEntry(0.5, 100, 1, 1, 10, 30.0, 8.0, -0.1) // Invalid prefix cache
		if err == nil {
			t.Error("Invalid training entry constructor should have failed")
		}
	})

	t.Log("✅ Offline validation tests completed")
}

// Test configuration handling without network calls
func TestConfigurationHandling(t *testing.T) {
	t.Log("Testing configuration handling...")

	// Test default configuration
	defaultConfig := DefaultConfig()
	if defaultConfig.MaxBulkSize != 100 {
		t.Errorf("Expected default MaxBulkSize to be 100, got %d", defaultConfig.MaxBulkSize)
	}

	if defaultConfig.UseNativeXGBoost != true {
		t.Errorf("Expected default UseNativeXGBoost to be true, got %t", defaultConfig.UseNativeXGBoost)
	}

	// Test configuration with mock URLs (no network calls)
	config := &Config{
		TrainingURL:            "http://mock-training.local",
		PredictionURLs:         []string{"http://mock1.local", "http://mock2.local"},
		MaxSampleSize:          500,
		FlushInterval:          2 * time.Second,
		MetricsRefreshInterval: 5 * time.Second,
		UseNativeXGBoost:       false,
		HTTPTimeout:            10 * time.Second,
		MaxBulkSize:            50,
	}

	// Create logger
	zapLog, err := zap.NewDevelopment()
	if err != nil {
		t.Fatalf("Failed to create logger: %v", err)
	}
	logger := zapr.NewLogger(zapLog)

	// Create predictor but don't start it (to avoid network calls)
	predictor := New(config, logger)

	// Test configuration access
	if predictor.GetTrainingURL() != config.TrainingURL {
		t.Errorf("Training URL mismatch: expected %s, got %s",
			config.TrainingURL, predictor.GetTrainingURL())
	}

	predictionURLs := predictor.GetPredictionURLs()
	if len(predictionURLs) != len(config.PredictionURLs) {
		t.Errorf("Prediction URLs length mismatch: expected %d, got %d",
			len(config.PredictionURLs), len(predictionURLs))
	}

	// Cleanup without starting background processes
	close(predictor.done)
	predictor.wg.Wait()

	t.Log("✅ Configuration handling tests completed")
}
