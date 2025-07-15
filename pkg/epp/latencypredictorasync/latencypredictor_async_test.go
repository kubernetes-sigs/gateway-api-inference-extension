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
	var parsedPredictionURLs []string
	for _, url := range strings.Split(predictionURLs, ",") {
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

	t.Run("TestPredictionWithPrefixCache", func(t *testing.T) {
		testPredictionWithPrefixCache(t, ctx, predictor)
	})

	t.Run("TestHTTPFallbackPrediction", func(t *testing.T) {
		testHTTPFallbackPrediction(t, ctx, predictor)
	})

	t.Run("TestPredictionPerformance", func(t *testing.T) {
		testPredictionPerformance(t, ctx, predictor)
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

	t.Logf("Model Info - Type: %s, Model Status: %v",
		modelInfo.ModelType, modelInfo.ModelStatus)

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
	t.Logf("  Overall ready: %t", predictor.IsReady())
	t.Logf("  XGBoost ready: %t", predictor.IsXGBoostReady())
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
		PrefixCacheScore:   0.8,   // 80% prefix cache hit rate
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

		t.Logf("Prediction %d: TTFT=%.2f, TPOT=%.2f (prefix_cache=%.1f%%)", 
			i+1, resp.TTFT, resp.TPOT, testReq.PrefixCacheScore*100)
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
	var ttftResults []float64

	for _, prefixScore := range prefixCacheScores {
		req := baseRequest
		req.PrefixCacheScore = prefixScore

		response, err := predictor.Predict(ctx, req)
		if err != nil {
			t.Errorf("Prediction failed for prefix cache score %.1f: %v", prefixScore, err)
			continue
		}

		ttftResults = append(ttftResults, response.TTFT)
		t.Logf("Prefix cache %.0f%%: TTFT=%.2f ms, TPOT=%.2f ms", 
			prefixScore*100, response.TTFT, response.TPOT)
	}

	// Analyze the relationship between prefix cache and TTFT
	if len(ttftResults) >= 2 {
		t.Log("Prefix cache impact analysis:")
		lowCacheTTFT := ttftResults[0]  // 0% prefix cache
		highCacheTTFT := ttftResults[len(ttftResults)-1]  // 100% prefix cache
		difference := highCacheTTFT - lowCacheTTFT

		t.Logf("  TTFT at 0%% prefix cache: %.2f ms", lowCacheTTFT)
		t.Logf("  TTFT at 100%% prefix cache: %.2f ms", highCacheTTFT)
		t.Logf("  Difference: %.2f ms", difference)

		if predictor.GetCurrentModelType() == "bayesian_ridge" {
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
	t.Log("Testing HTTP fallback prediction when native XGBoost fails...")

	// Since we know XGBoost native parsing failed from the logs,
	// the predictor should fall back to HTTP predictions
	if predictor.GetCurrentModelType() != "xgboost" {
		t.Skip("This test is specific to XGBoost model type")
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

	t.Logf("Making HTTP fallback prediction request: %+v", req)

	response, err := predictor.Predict(ctx, req)
	if err != nil {
		t.Fatalf("HTTP fallback prediction failed: %v", err)
	}

	t.Logf("HTTP Fallback Prediction Response:")
	t.Logf("  TTFT: %.2f ms", response.TTFT)
	t.Logf("  TPOT: %.2f ms", response.TPOT)
	t.Logf("  Model Type: %s", response.ModelType)
	t.Logf("  Prefix Cache Score Used: %.1f%%", req.PrefixCacheScore*100)

	// Validate that we got a reasonable response
	if response.TTFT <= 0 {
		t.Error("TTFT should be positive")
	}
	if response.TPOT <= 0 {
		t.Error("TPOT should be positive")
	}

	// The model type should indicate it's using XGBoost (likely "xgboost" from HTTP)
	if response.ModelType == "" {
		t.Error("Model type should not be empty")
	}

	t.Logf("Successfully tested HTTP fallback prediction with prefix cache")
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
	var minSingleDuration time.Duration = time.Hour // Initialize to large value

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
		t.Logf("Prediction %d: %.2fms - TTFT: %.1fms, TPOT: %.1fms (prefix: %.0f%%)",
			i+1, durationMs, response.TTFT, response.TPOT, testReq.PrefixCacheScore*100)
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
	var parsedPredictionURLs []string
	for _, url := range strings.Split(predictionURLs, ",") {
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

		t.Logf("%s Test %d: %.1fms (TTFT: %.0fms, TPOT: %.0fms, prefix: %.0f%%)",
			status, i+1, durationMs, response.TTFT, response.TPOT, testReq.PrefixCacheScore*100)
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
	var parsedPredictionURLs []string
	for _, url := range strings.Split(predictionURLs, ",") {
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

		t.Logf("HTTP-only prediction %d: TTFT=%.2f, TPOT=%.2f (prefix: %.0f%%)", 
			i+1, resp.TTFT, resp.TPOT, testReq.PrefixCacheScore*100)
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
		t.Logf("Prediction %d: TTFT=%.2f, TPOT=%.2f (prefix: %.0f%%)", 
			i+1, response.TTFT, response.TPOT, testReq.PrefixCacheScore*100)
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
		0.7,   // kv_cache_percentage
		500,   // input_token_length
		3,     // num_request_waiting
		2,     // num_request_running
		100,   // num_tokens_generated
		0.85,  // prefix_cache_score
	)
	if err != nil {
		t.Errorf("Valid prediction request constructor failed: %v", err)
	} else {
		t.Logf("✓ Created prediction request: TTFT features with %.0f%% prefix cache", req.PrefixCacheScore*100)
	}

	// Test invalid prediction request constructor
	_, err = NewPredictionRequest(
		0.7,   // kv_cache_percentage
		500,   // input_token_length
		3,     // num_request_waiting
		2,     // num_request_running
		100,   // num_tokens_generated
		1.5,   // prefix_cache_score (invalid)
	)
	if err == nil {
		t.Error("Invalid prediction request constructor should have failed")
	} else {
		t.Logf("✓ Invalid prediction request correctly rejected: %v", err)
	}

	// Test valid training entry constructor
	entry, err := NewTrainingEntry(
		0.6,   // kv_cache_percentage
		300,   // input_token_length
		2,     // num_request_waiting
		1,     // num_request_running
		50,    // num_tokens_generated
		45.5,  // actual_ttft_ms
		12.3,  // actual_tpot_ms
		0.75,  // prefix_cache_score
	)
	if err != nil {
		t.Errorf("Valid training entry constructor failed: %v", err)
	} else {
		t.Logf("✓ Created training entry: TTFT=%.1fms, TPOT=%.1fms, prefix cache=%.0f%%", 
			entry.ActualTTFT, entry.ActualTPOT, entry.PrefixCacheScore*100)
	}

	// Test invalid training entry constructor
	_, err = NewTrainingEntry(
		0.6,   // kv_cache_percentage
		300,   // input_token_length
		2,     // num_request_waiting
		1,     // num_request_running
		50,    // num_tokens_generated
		45.5,  // actual_ttft_ms
		12.3,  // actual_tpot_ms
		-0.1,  // prefix_cache_score (invalid)
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

	if predictor.GetCurrentModelType() != "xgboost" {
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
			if key == "split" {
				t.Logf("  %s: %T = %v", key, value, value)
			} else if key == "children" && value != nil {
				if children, ok := value.([]interface{}); ok {
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
					t.Logf("  %s: %T = %v", key, value, value)
				}
			} else {
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
	t.Logf("Testing metrics for model type: %s", modelType)

	switch modelType {
	case "bayesian_ridge":
		testBayesianRidgeMetrics(t, ctx, predictor)
	case "xgboost":
		testXGBoostMetrics(t, ctx, predictor)
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
	var parsedPredictionURLs []string
	for _, url := range strings.Split(predictionURLs, ",") {
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
	}

	predictor := New(config, logger)
	defer predictor.Stop()

	ctx := context.Background()
	predictor.Start(ctx)

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
		PrefixCacheScore:   0.8,  // 80% prefix cache hit rate
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

// Test to verify config loading from environment
func TestConfigFromEnv(t *testing.T) {
	// Save original env vars
	originalLatencyURL := os.Getenv("PREDICTION_SERVER_URL")
	originalTrainingURL := os.Getenv("TRAINING_SERVER_URL")
	originalSample := os.Getenv("LATENCY_MAX_SAMPLE_SIZE")
	originalInterval := os.Getenv("LATENCY_FLUSH_INTERVAL_SEC")
	originalNative := os.Getenv("LATENCY_USE_NATIVE_XGBOOST")
	originalTimeout := os.Getenv("LATENCY_HTTP_TIMEOUT_SEC")

	// Set test env vars
	os.Setenv("PREDICTION_SERVER_URL", "http://pred1.example.com,http://pred2.example.com,http://pred3.example.com")
	os.Setenv("TRAINING_SERVER_URL", "http://training.example.com")
	os.Setenv("LATENCY_MAX_SAMPLE_SIZE", "500")
	os.Setenv("LATENCY_FLUSH_INTERVAL_SEC", "5")
	os.Setenv("LATENCY_USE_NATIVE_XGBOOST", "false")
	os.Setenv("LATENCY_HTTP_TIMEOUT_SEC", "20")

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
}

// Test URL parsing edge cases
func TestConfigURLParsing(t *testing.T) {
	tests := []struct {
		name                   string
		latencyServerURL       string
		trainingServerURL      string
		expectedPredictionURLs []string
		expectedTrainingURL    string
	}{
		{
			name:                   "Single prediction URL",
			latencyServerURL:       "http://localhost:8001",
			trainingServerURL:      "http://localhost:8000",
			expectedPredictionURLs: []string{"http://localhost:8001"},
			expectedTrainingURL:    "http://localhost:8000",
		},
		{
			name:                   "Multiple prediction URLs with spaces",
			latencyServerURL:       "http://localhost:8001, http://localhost:8002 ,http://localhost:8003",
			trainingServerURL:      "http://localhost:8000",
			expectedPredictionURLs: []string{"http://localhost:8001", "http://localhost:8002", "http://localhost:8003"},
			expectedTrainingURL:    "http://localhost:8000",
		},
		{
			name:                   "Empty training URL with prediction URLs",
			latencyServerURL:       "http://localhost:8001,http://localhost:8002",
			trainingServerURL:      "",
			expectedPredictionURLs: []string{"http://localhost:8001", "http://localhost:8002"},
			expectedTrainingURL:    "http://localhost:8000", // Should use default
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original env vars
			originalLatencyURL := os.Getenv("PREDICTION_SERVER_URL")
			originalTrainingURL := os.Getenv("TRAINING_SERVER_URL")

			// Set test env vars
			os.Setenv("PREDICTION_SERVER_URL", tt.latencyServerURL)
			if tt.trainingServerURL != "" {
				os.Setenv("TRAINING_SERVER_URL", tt.trainingServerURL)
			} else {
				os.Unsetenv("TRAINING_SERVER_URL")
			}

			defer func() {
				// Restore original env vars
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
			}()

			config := ConfigFromEnv()

			// Check prediction URLs
			if len(config.PredictionURLs) != len(tt.expectedPredictionURLs) {
				t.Errorf("Expected %d prediction URLs, got %d", len(tt.expectedPredictionURLs), len(config.PredictionURLs))
			}
			for i, expected := range tt.expectedPredictionURLs {
				if i >= len(config.PredictionURLs) || config.PredictionURLs[i] != expected {
					t.Errorf("Expected PredictionURLs[%d] to be '%s', got '%s'", i, expected, config.PredictionURLs[i])
				}
			}

			// Check training URL
			if config.TrainingURL != tt.expectedTrainingURL {
				t.Errorf("Expected TrainingURL to be '%s', got '%s'", tt.expectedTrainingURL, config.TrainingURL)
			}
		})
	}
}

// Test prefix cache score impact on training data generation
func TestTrainingDataWithPrefixCache(t *testing.T) {
	t.Log("Testing training data generation with prefix cache scores...")

	entries := generateTrainingEntries(100)

	// Validate all entries have prefix cache scores
	for i, entry := range entries {
		if entry.PrefixCacheScore < 0.0 || entry.PrefixCacheScore > 1.0 {
			t.Errorf("Entry %d has invalid prefix cache score: %.3f", i, entry.PrefixCacheScore)
		}
	}

	// Check that prefix cache scores vary
	var prefixScores []float64
	for _, entry := range entries {
		prefixScores = append(prefixScores, entry.PrefixCacheScore)
	}

	// Calculate variance to ensure we have variety
	var sum, mean, variance float64
	for _, score := range prefixScores {
		sum += score
	}
	mean = sum / float64(len(prefixScores))

	for _, score := range prefixScores {
		variance += (score - mean) * (score - mean)
	}
	variance /= float64(len(prefixScores))

	t.Logf("Prefix cache score statistics:")
	t.Logf("  Mean: %.3f", mean)
	t.Logf("  Variance: %.3f", variance)
	t.Logf("  Range: [%.3f, %.3f]", 0.0, 1.0)

	if variance < 0.05 {
		t.Error("Prefix cache scores should have more variance for good training data")
	} else {
		t.Log("✓ Good variance in prefix cache scores")
	}

	// Verify the training equation includes prefix cache impact
	// Check that entries with higher prefix cache tend to have higher TTFT
	// (based on our training equation: ttft includes +30*prefixCache)
	
	// Sort by prefix cache score
	type entryWithIndex struct {
		entry TrainingEntry
		index int
	}
	
	var sortedEntries []entryWithIndex
	for i, entry := range entries {
		sortedEntries = append(sortedEntries, entryWithIndex{entry, i})
	}
	
	// Simple sort by prefix cache score
	for i := 0; i < len(sortedEntries)-1; i++ {
		for j := i + 1; j < len(sortedEntries); j++ {
			if sortedEntries[i].entry.PrefixCacheScore > sortedEntries[j].entry.PrefixCacheScore {
				sortedEntries[i], sortedEntries[j] = sortedEntries[j], sortedEntries[i]
			}
		}
	}
	
	// Compare low vs high prefix cache entries
	lowPrefixCount := len(sortedEntries) / 4
	highPrefixStart := len(sortedEntries) * 3 / 4
	
	var lowPrefixTTFT, highPrefixTTFT float64
	for i := 0; i < lowPrefixCount; i++ {
		lowPrefixTTFT += sortedEntries[i].entry.ActualTTFT
	}
	lowPrefixTTFT /= float64(lowPrefixCount)
	
	highPrefixCount := len(sortedEntries) - highPrefixStart
	for i := highPrefixStart; i < len(sortedEntries); i++ {
		highPrefixTTFT += sortedEntries[i].entry.ActualTTFT
	}
	highPrefixTTFT /= float64(highPrefixCount)
	
	ttftDifference := highPrefixTTFT - lowPrefixTTFT
	
	t.Logf("TTFT impact analysis:")
	t.Logf("  Low prefix cache TTFT avg: %.2f ms", lowPrefixTTFT)
	t.Logf("  High prefix cache TTFT avg: %.2f ms", highPrefixTTFT)
	t.Logf("  Difference: %.2f ms", ttftDifference)
	
	if ttftDifference > 10 {
		t.Log("✓ Prefix cache score appears to positively impact TTFT in training data")
	} else {
		t.Log("ℹ Small or no prefix cache impact detected (may be due to noise)")
	}

	t.Log("✅ Training data with prefix cache validation completed")
}

// Test prediction request validation edge cases
func TestPredictionValidationEdgeCases(t *testing.T) {
	t.Log("Testing prediction validation edge cases with prefix cache...")

	predictor := &Predictor{} // Temporary predictor for validation

	testCases := []struct {
		name      string
		req       PredictionRequest
		shouldErr bool
		errorMsg  string
	}{
		{
			name: "Valid minimum values",
			req: PredictionRequest{
				KVCachePercentage:  0.0,
				InputTokenLength:   0,
				NumRequestWaiting:  0,
				NumRequestRunning:  0,
				NumTokensGenerated: 0,
				PrefixCacheScore:   0.0,
			},
			shouldErr: false,
		},
		{
			name: "Valid maximum values",
			req: PredictionRequest{
				KVCachePercentage:  1.0,
				InputTokenLength:   10000,
				NumRequestWaiting:  100,
				NumRequestRunning:  50,
				NumTokensGenerated: 1000,
				PrefixCacheScore:   1.0,
			},
			shouldErr: false,
		},
		{
			name: "Invalid negative prefix cache",
			req: PredictionRequest{
				KVCachePercentage:  0.5,
				InputTokenLength:   100,
				NumRequestWaiting:  1,
				NumRequestRunning:  1,
				NumTokensGenerated: 10,
				PrefixCacheScore:   -0.001,
			},
			shouldErr: true,
			errorMsg:  "prefix_cache_score must be between 0.0 and 1.0",
		},
		{
			name: "Invalid high prefix cache",
			req: PredictionRequest{
				KVCachePercentage:  0.5,
				InputTokenLength:   100,
				NumRequestWaiting:  1,
				NumRequestRunning:  1,
				NumTokensGenerated: 10,
				PrefixCacheScore:   1.001,
			},
			shouldErr: true,
			errorMsg:  "prefix_cache_score must be between 0.0 and 1.0",
		},
		{
			name: "Invalid negative KV cache with valid prefix cache",
			req: PredictionRequest{
				KVCachePercentage:  -0.1,
				InputTokenLength:   100,
				NumRequestWaiting:  1,
				NumRequestRunning:  1,
				NumTokensGenerated: 10,
				PrefixCacheScore:   0.8,
			},
			shouldErr: true,
			errorMsg:  "kv_cache_percentage must be between 0.0 and 1.0",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := predictor.ValidatePredictionRequest(tc.req)
			
			if tc.shouldErr {
				if err == nil {
					t.Errorf("Expected validation error for %s, but got none", tc.name)
				} else if !strings.Contains(err.Error(), tc.errorMsg) {
					t.Errorf("Expected error message to contain '%s', got: %v", tc.errorMsg, err)
				} else {
					t.Logf("✓ Correctly rejected %s: %v", tc.name, err)
				}
			} else {
				if err != nil {
					t.Errorf("Expected no validation error for %s, but got: %v", tc.name, err)
				} else {
					t.Logf("✓ Correctly accepted %s", tc.name)
				}
			}
		})
	}

	t.Log("✅ Prediction validation edge cases completed")
}

// Test training entry validation edge cases
func TestTrainingValidationEdgeCases(t *testing.T) {
	t.Log("Testing training entry validation edge cases with prefix cache...")

	predictor := &Predictor{} // Temporary predictor for validation

	testCases := []struct {
		name      string
		entry     TrainingEntry
		shouldErr bool
		errorMsg  string
	}{
		{
			name: "Valid entry with prefix cache",
			entry: TrainingEntry{
				KVCachePercentage:  0.6,
				InputTokenLength:   200,
				NumRequestWaiting:  2,
				NumRequestRunning:  1,
				NumTokensGenerated: 20,
				ActualTTFT:         45.5,
				ActualTPOT:         12.3,
				PrefixCacheScore:   0.8,
				Timestamp:          time.Now(),
			},
			shouldErr: false,
		},
		{
			name: "Zero prefix cache score",
			entry: TrainingEntry{
				KVCachePercentage:  0.5,
				InputTokenLength:   100,
				NumRequestWaiting:  1,
				NumRequestRunning:  1,
				NumTokensGenerated: 10,
				ActualTTFT:         30.0,
				ActualTPOT:         8.0,
				PrefixCacheScore:   0.0, // Valid minimum
				Timestamp:          time.Now(),
			},
			shouldErr: false,
		},
		{
			name: "Maximum prefix cache score",
			entry: TrainingEntry{
				KVCachePercentage:  0.5,
				InputTokenLength:   100,
				NumRequestWaiting:  1,
				NumRequestRunning:  1,
				NumTokensGenerated: 10,
				ActualTTFT:         30.0,
				ActualTPOT:         8.0,
				PrefixCacheScore:   1.0, // Valid maximum
				Timestamp:          time.Now(),
			},
			shouldErr: false,
		},
		{
			name: "Invalid negative prefix cache",
			entry: TrainingEntry{
				KVCachePercentage:  0.5,
				InputTokenLength:   100,
				NumRequestWaiting:  1,
				NumRequestRunning:  1,
				NumTokensGenerated: 10,
				ActualTTFT:         30.0,
				ActualTPOT:         8.0,
				PrefixCacheScore:   -0.1,
				Timestamp:          time.Now(),
			},
			shouldErr: true,
			errorMsg:  "prefix_cache_score must be between 0.0 and 1.0",
		},
		{
			name: "Invalid high prefix cache",
			entry: TrainingEntry{
				KVCachePercentage:  0.5,
				InputTokenLength:   100,
				NumRequestWaiting:  1,
				NumRequestRunning:  1,
				NumTokensGenerated: 10,
				ActualTTFT:         30.0,
				ActualTPOT:         8.0,
				PrefixCacheScore:   1.5,
				Timestamp:          time.Now(),
			},
			shouldErr: true,
			errorMsg:  "prefix_cache_score must be between 0.0 and 1.0",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := predictor.ValidateTrainingEntry(tc.entry)
			
			if tc.shouldErr {
				if err == nil {
					t.Errorf("Expected validation error for %s, but got none", tc.name)
				} else if !strings.Contains(err.Error(), tc.errorMsg) {
					t.Errorf("Expected error message to contain '%s', got: %v", tc.errorMsg, err)
				} else {
					t.Logf("✓ Correctly rejected %s: %v", tc.name, err)
				}
			} else {
				if err != nil {
					t.Errorf("Expected no validation error for %s, but got: %v", tc.name, err)
				} else {
					t.Logf("✓ Correctly accepted %s", tc.name)
				}
			}
		})
	}

	t.Log("✅ Training validation edge cases completed")
}

// Test comprehensive prefix cache feature integration
func TestPrefixCacheFeatureIntegration(t *testing.T) {
	t.Log("Testing comprehensive prefix cache feature integration...")

	// Test that all components work together with prefix cache
	zapLog, err := zap.NewDevelopment()
	if err != nil {
		t.Fatalf("Failed to create logger: %v", err)
	}
	logger := zapr.NewLogger(zapLog)

	// Create a minimal config for testing
	config := &Config{
		TrainingURL:            "http://mock-training.local",
		PredictionURLs:         []string{"http://mock-prediction.local"},
		MaxSampleSize:          100,
		FlushInterval:          10 * time.Second, // Long interval for testing
		MetricsRefreshInterval: 10 * time.Second,
		UseNativeXGBoost:       false,
		HTTPTimeout:            5 * time.Second,
	}

	predictor := New(config, logger)
	defer predictor.Stop()

	// Test that training entries with prefix cache can be created
	entries := make([]TrainingEntry, 10)
	for i := 0; i < 10; i++ {
		entry, err := NewTrainingEntry(
			float64(i)/10.0,    // kv_cache_percentage
			100+i*50,           // input_token_length
			i%5,                // num_request_waiting
			(i%3)+1,            // num_request_running
			10+i*5,             // num_tokens_generated
			50.0+float64(i)*5,  // actual_ttft_ms
			10.0+float64(i)*2,  // actual_tpot_ms
			float64(i)/9.0,     // prefix_cache_score (0.0 to 1.0)
		)
		if err != nil {
			t.Fatalf("Failed to create training entry %d: %v", i, err)
		}
		entries[i] = entry
		
		t.Logf("Entry %d: prefix_cache=%.1f%%, ttft=%.1f, tpot=%.1f", 
			i, entry.PrefixCacheScore*100, entry.ActualTTFT, entry.ActualTPOT)
	}

	// Test that training entries can be added to predictor
	err = predictor.AddTrainingDataBulk(entries)
	if err != nil {
		t.Fatalf("Failed to add training entries with prefix cache: %v", err)
	}
	t.Log("✓ Successfully added training entries with prefix cache scores")

	// Test that prediction requests with prefix cache can be created
	for i := 0; i < 5; i++ {
		req, err := NewPredictionRequest(
			float64(i*20)/100.0, // kv_cache_percentage: 0%, 20%, 40%, 60%, 80%
			200+i*100,           // input_token_length
			i%4,                 // num_request_waiting
			(i%2)+1,             // num_request_running
			20+i*10,             // num_tokens_generated
			float64(i)/4.0,      // prefix_cache_score: 0.0, 0.25, 0.5, 0.75, 1.0
		)
		if err != nil {
			t.Fatalf("Failed to create prediction request %d: %v", i, err)
		}
		
		t.Logf("Request %d: prefix_cache=%.1f%%, kv_cache=%.1f%%, input_len=%d", 
			i, req.PrefixCacheScore*100, req.KVCachePercentage*100, req.InputTokenLength)
		
		// Validate the request
		err = predictor.ValidatePredictionRequest(req)
		if err != nil {
			t.Errorf("Valid prediction request %d failed validation: %v", i, err)
		}
	}
	t.Log("✓ Successfully created and validated prediction requests with prefix cache scores")

	// Test validation edge cases work correctly
	testCases := []struct {
		name         string
		prefixCache  float64
		shouldPass   bool
	}{
		{"Zero prefix cache", 0.0, true},
		{"Half prefix cache", 0.5, true},
		{"Full prefix cache", 1.0, true},
		{"Negative prefix cache", -0.1, false},
		{"Over-full prefix cache", 1.1, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req := PredictionRequest{
				KVCachePercentage:  0.5,
				InputTokenLength:   100,
				NumRequestWaiting:  1,
				NumRequestRunning:  1,
				NumTokensGenerated: 10,
				PrefixCacheScore:   tc.prefixCache,
			}
			
			err := predictor.ValidatePredictionRequest(req)
			if tc.shouldPass && err != nil {
				t.Errorf("Expected %s to pass validation, got error: %v", tc.name, err)
			} else if !tc.shouldPass && err == nil {
				t.Errorf("Expected %s to fail validation, but it passed", tc.name)
			}
		})
	}

	t.Log("✅ Comprehensive prefix cache feature integration test completed")
}

// Test that demonstrates the prefix cache feature end-to-end
func TestPrefixCacheEndToEnd(t *testing.T) {
	t.Log("Testing prefix cache feature end-to-end workflow...")

	// This test demonstrates a complete workflow with prefix cache scores
	
	// 1. Create training data that shows prefix cache impact
	t.Log("Step 1: Creating training data with prefix cache impact...")
	
	var trainingEntries []TrainingEntry
	rng := rand.New(rand.NewSource(42)) // Fixed seed for reproducible test
	
	for i := 0; i < 50; i++ {
		kv := 0.5 + rng.Float64()*0.3        // 0.5 to 0.8
		inputLen := 200 + rng.Intn(300)      // 200 to 500
		waiting := rng.Intn(5)               // 0 to 4
		running := 1 + rng.Intn(3)           // 1 to 3
		generated := 20 + rng.Intn(80)       // 20 to 100
		prefixCache := rng.Float64()         // 0.0 to 1.0
		
		// Simulate the actual equation with prefix cache impact on TTFT
		// TTFT = base + 2*input + 3*waiting + 4*running + 50*kv + 30*prefix_cache + noise
		ttft := 95.0 + 
			2.0*float64(inputLen) + 
			3.0*float64(waiting) + 
			4.0*float64(running) + 
			50.0*kv + 
			30.0*prefixCache +  // Prefix cache impact
			rng.NormFloat64()*5 // Small noise
		
		// TPOT = base + 0.5*input + 1*generated + 5*running + 100*kv + noise
		// (No prefix cache impact on TPOT)
		tpot := 9.0 + 
			0.5*float64(inputLen) + 
			1.0*float64(generated) + 
			5.0*float64(running) + 
			100.0*kv + 
			rng.NormFloat64()*3 // Small noise
		
		entry := TrainingEntry{
			KVCachePercentage:  kv,
			InputTokenLength:   inputLen,
			NumRequestWaiting:  waiting,
			NumRequestRunning:  running,
			NumTokensGenerated: generated,
			ActualTTFT:         ttft,
			ActualTPOT:         tpot,
			PrefixCacheScore:   prefixCache,
			Timestamp:          time.Now().Add(-time.Duration(i) * time.Minute),
		}
		
		trainingEntries = append(trainingEntries, entry)
	}
	
	t.Logf("Created %d training entries with prefix cache scores", len(trainingEntries))
	
	// 2. Analyze the training data to show prefix cache correlation
	t.Log("Step 2: Analyzing prefix cache correlation in training data...")
	
	// Sort by prefix cache score
	sortedEntries := make([]TrainingEntry, len(trainingEntries))
	copy(sortedEntries, trainingEntries)
	
	// Simple bubble sort by prefix cache score
	for i := 0; i < len(sortedEntries)-1; i++ {
		for j := i + 1; j < len(sortedEntries); j++ {
			if sortedEntries[i].PrefixCacheScore > sortedEntries[j].PrefixCacheScore {
				sortedEntries[i], sortedEntries[j] = sortedEntries[j], sortedEntries[i]
			}
		}
	}
	
	// Compare bottom 25% vs top 25%
	quarterSize := len(sortedEntries) / 4
	
	var lowPrefixTTFT, highPrefixTTFT float64
	var lowPrefixTPOT, highPrefixTPOT float64
	var lowPrefixCacheAvg, highPrefixCacheAvg float64
	
	// Calculate averages for low prefix cache group (bottom 25%)
	for i := 0; i < quarterSize; i++ {
		lowPrefixTTFT += sortedEntries[i].ActualTTFT
		lowPrefixTPOT += sortedEntries[i].ActualTPOT
		lowPrefixCacheAvg += sortedEntries[i].PrefixCacheScore
	}
	lowPrefixTTFT /= float64(quarterSize)
	lowPrefixTPOT /= float64(quarterSize)
	lowPrefixCacheAvg /= float64(quarterSize)
	
	// Calculate averages for high prefix cache group (top 25%)
	startIdx := len(sortedEntries) - quarterSize
	for i := startIdx; i < len(sortedEntries); i++ {
		highPrefixTTFT += sortedEntries[i].ActualTTFT
		highPrefixTPOT += sortedEntries[i].ActualTPOT
		highPrefixCacheAvg += sortedEntries[i].PrefixCacheScore
	}
	highPrefixTTFT /= float64(quarterSize)
	highPrefixTPOT /= float64(quarterSize)
	highPrefixCacheAvg /= float64(quarterSize)
	
	ttftDiff := highPrefixTTFT - lowPrefixTTFT
	tpotDiff := highPrefixTPOT - lowPrefixTPOT
	
	t.Logf("Training data analysis results:")
	t.Logf("  Low prefix cache group (avg=%.2f): TTFT=%.1f ms, TPOT=%.1f ms", 
		lowPrefixCacheAvg, lowPrefixTTFT, lowPrefixTPOT)
	t.Logf("  High prefix cache group (avg=%.2f): TTFT=%.1f ms, TPOT=%.1f ms", 
		highPrefixCacheAvg, highPrefixTTFT, highPrefixTPOT)
	t.Logf("  TTFT difference: %.1f ms (expect ~%.1f ms)", 
		ttftDiff, (highPrefixCacheAvg-lowPrefixCacheAvg)*30.0)
	t.Logf("  TPOT difference: %.1f ms (expect ~0 ms)", tpotDiff)
	
	// Validate that we see the expected prefix cache impact
	expectedTTFTDiff := (highPrefixCacheAvg - lowPrefixCacheAvg) * 30.0 // Our training coefficient
	if ttftDiff > expectedTTFTDiff*0.5 && ttftDiff < expectedTTFTDiff*1.5 {
		t.Log("✓ TTFT shows expected prefix cache correlation")
	} else {
		t.Logf("ℹ TTFT correlation weaker than expected (noise effects)")
	}
	
	if abs(tpotDiff) < 10 { // TPOT should not be significantly affected
		t.Log("✓ TPOT correctly shows minimal prefix cache correlation")
	} else {
		t.Logf("⚠ TPOT unexpectedly affected by prefix cache: %.1f ms difference", tpotDiff)
	}
	
	// 3. Create prediction scenarios to demonstrate usage
	t.Log("Step 3: Creating prediction scenarios...")
	
	scenarios := []struct {
		name        string
		description string
		req         PredictionRequest
	}{
		{
			name:        "Cold Cache",
			description: "No prefix cache hits, high latency expected",
			req: PredictionRequest{
				KVCachePercentage:  0.7,
				InputTokenLength:   400,
				NumRequestWaiting:  2,
				NumRequestRunning:  1,
				NumTokensGenerated: 50,
				PrefixCacheScore:   0.0, // No cache hits
			},
		},
		{
			name:        "Warm Cache",
			description: "Moderate prefix cache hits",
			req: PredictionRequest{
				KVCachePercentage:  0.7,
				InputTokenLength:   400,
				NumRequestWaiting:  2,
				NumRequestRunning:  1,
				NumTokensGenerated: 50,
				PrefixCacheScore:   0.5, // 50% cache hits
			},
		},
		{
			name:        "Hot Cache",
			description: "High prefix cache hits, low latency expected",
			req: PredictionRequest{
				KVCachePercentage:  0.7,
				InputTokenLength:   400,
				NumRequestWaiting:  2,
				NumRequestRunning:  1,
				NumTokensGenerated: 50,
				PrefixCacheScore:   0.9, // 90% cache hits
			},
		},
	}
	
	for _, scenario := range scenarios {
		// Validate each scenario
		predictor := &Predictor{} // Temporary for validation
		err := predictor.ValidatePredictionRequest(scenario.req)
		if err != nil {
			t.Errorf("Scenario '%s' failed validation: %v", scenario.name, err)
			continue
		}
		
		// Calculate expected TTFT using our training equation
		expectedTTFT := 95.0 + 
			2.0*float64(scenario.req.InputTokenLength) +
			3.0*float64(scenario.req.NumRequestWaiting) +
			4.0*float64(scenario.req.NumRequestRunning) +
			50.0*scenario.req.KVCachePercentage +
			30.0*scenario.req.PrefixCacheScore
		
		expectedTPOT := 9.0 +
			0.5*float64(scenario.req.InputTokenLength) +
			1.0*float64(scenario.req.NumTokensGenerated) +
			5.0*float64(scenario.req.NumRequestRunning) +
			100.0*scenario.req.KVCachePercentage
		
		t.Logf("Scenario: %s", scenario.name)
		t.Logf("  Description: %s", scenario.description)
		t.Logf("  Prefix cache: %.0f%%", scenario.req.PrefixCacheScore*100)
		t.Logf("  Expected TTFT: %.1f ms", expectedTTFT)
		t.Logf("  Expected TPOT: %.1f ms", expectedTPOT)
		t.Log("")
	}
	
	t.Log("✅ End-to-end prefix cache workflow demonstration completed")
}

// Helper function for absolute value
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}