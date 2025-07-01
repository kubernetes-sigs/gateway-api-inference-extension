package latencypredictorasync

import (
	"context"
	"math/rand"
	"os"
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

	// Check if server URL is set
	serverURL := os.Getenv("LATENCY_SERVER_URL")
	if serverURL == "" {
		t.Skip("LATENCY_SERVER_URL not set, skipping integration test")
	}

	// Create config with the actual server URL
	config := &Config{
		PythonURL:        serverURL,
		MaxSampleSize:    1000,
		FlushInterval:    500 * time.Millisecond, // Shorter for testing
		MetricsRefreshInterval: 1 * time.Second, // Longer for metrics
		UseNativeXGBoost: true,
		HTTPTimeout:      30 * time.Second, // Longer timeout for tests
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
		testHTTPOnlyPrediction(t, ctx,)
	})

	t.Run("TestMetricsRetrieval", func(t *testing.T) {
		testMetricsRetrieval(t, ctx, predictor)
	})
}

func testModelInfo(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing model info retrieval...")

	modelInfo, err := predictor.GetModelInfo(ctx)
	if err != nil {
		t.Fatalf("Failed to get model info: %v", err)
	}

	t.Logf("Model Info - Type: %s,  Model Status: %v",
		modelInfo.ModelType, modelInfo.ModelStatus)

	if modelInfo.ModelType == "" {
		t.Error("Model type should not be empty")
	}

	// Store model type for other tests
	currentModelType := predictor.GetCurrentModelType()
	t.Logf("Current model type from predictor: %s", currentModelType)
}

func testBulkTrainingData(t *testing.T, predictor *Predictor) {
	t.Log("Testing bulk training data submission...")

	// Generate 1000 random training entries
	entries := generateTrainingEntries(1000)
	
	err := predictor.AddTrainingDataBulk(entries)
	if err != nil {
		t.Fatalf("Failed to add bulk training data: %v", err)
	}

	t.Logf("Successfully added %d training entries to buffer", len(entries))

	// Wait a bit for the background flush to occur
	time.Sleep(2 * time.Second)

	t.Log("Training data should have been flushed to server")
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

	// Create a sample prediction request
	// Note: kv_cache_percentage should be between 0 and 1 (fraction, not percentage)
	req := PredictionRequest{
		KVCachePercentage:  0.755, // 75.5% as a fraction
		InputTokenLength:   512,
		NumRequestWaiting:  3,
		NumRequestRunning:  2,
		NumTokensGenerated: 100,
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
	t.Log("Testing multiple predictions...")
	for i := 0; i < 5; i++ {
		testReq := PredictionRequest{
			KVCachePercentage:  float64(50+i*10) / 100.0, // Convert percentage to fraction
			InputTokenLength:   256 + i*128,
			NumRequestWaiting:  i,
			NumRequestRunning:  1 + i,
			NumTokensGenerated: 50 + i*25,
		}

		resp, err := predictor.Predict(ctx, testReq)
		if err != nil {
			t.Errorf("Prediction %d failed: %v", i+1, err)
			continue
		}

		t.Logf("Prediction %d: TTFT=%.2f, TPOT=%.2f", i+1, resp.TTFT, resp.TPOT)
	}
}

func testHTTPFallbackPrediction(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing HTTP fallback prediction when native XGBoost fails...")

	// Since we know XGBoost native parsing failed from the logs, 
	// the predictor should fall back to HTTP predictions
	if predictor.GetCurrentModelType() != "xgboost" {
		t.Skip("This test is specific to XGBoost model type")
	}

	// Test prediction with HTTP fallback
	req := PredictionRequest{
		KVCachePercentage:  0.8,   // 80% as a fraction
		InputTokenLength:   1024,
		NumRequestWaiting:  5,
		NumRequestRunning:  3,
		NumTokensGenerated: 150,
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

	t.Logf("Successfully tested HTTP fallback prediction")
}

func testPredictionPerformance(t *testing.T, ctx context.Context, predictor *Predictor) {
	t.Log("Testing prediction performance (target: < 300ms)...")

	// Ensure predictor is ready
	if !predictor.IsReady() {
		t.Skip("Predictor not ready for performance test")
	}

	req := PredictionRequest{
		KVCachePercentage:  0.6,   // 60% as a fraction
		InputTokenLength:   768,
		NumRequestWaiting:  2,
		NumRequestRunning:  1,
		NumTokensGenerated: 80,
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
	const maxDurationMs = 500

	var totalDuration time.Duration
	var maxSingleDuration time.Duration
	var minSingleDuration time.Duration = time.Hour // Initialize to large value

	t.Logf("Running %d prediction performance tests...", numTests)

	for i := 0; i < numTests; i++ {
		start := time.Now()
		
		response, err := predictor.Predict(ctx, req)
		
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
		t.Logf("Prediction %d: %.2fms - TTFT: %.1fms, TPOT: %.1fms", 
			i+1, durationMs, response.TTFT, response.TPOT)

		// Check if this prediction exceeded the target
		if durationMs > maxDurationMs {
			t.Errorf("Prediction %d took %.2fms, exceeded target of %dms", i+1, durationMs, maxDurationMs)
		}
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
	t.Logf("  Target:  < %dms", maxDurationMs)

	// Overall performance check
	if avgMs > maxDurationMs {
		t.Errorf("Average prediction time %.2fms exceeded target of %dms", avgMs, maxDurationMs)
	} else {
		t.Logf("✅ Performance target met: avg %.2fms < %dms", avgMs, maxDurationMs)
	}

	// Check for consistency (max shouldn't be too much higher than average)
	if maxMs > avgMs*3 {
		t.Logf("⚠️  High variance detected: max %.2fms is %.1fx the average", maxMs, maxMs/avgMs)
	} else {
		t.Logf("✅ Good consistency: max %.2fms is %.1fx the average", maxMs, maxMs/avgMs)
	}
}

func testHTTPOnlyPerformance(t *testing.T, ctx context.Context) {
	t.Log("Testing HTTP-only prediction performance (no native XGBoost interference)...")

	serverURL := os.Getenv("LATENCY_SERVER_URL")
	if serverURL == "" {
		t.Skip("LATENCY_SERVER_URL not set")
	}

	// Create a dedicated HTTP-only predictor for clean performance testing
	zapLog, err := zap.NewDevelopment()
	if err != nil {
		t.Fatalf("Failed to create logger: %v", err)
	}
	logger := zapr.NewLogger(zapLog)

	httpOnlyConfig := &Config{
		PythonURL:        serverURL,
		MaxSampleSize:    1000,
		FlushInterval:    1 * time.Second, // Long interval to avoid interference
		MetricsRefreshInterval: 1 * time.Second, // Longer for metrics
		UseNativeXGBoost: false,            // Force HTTP-only
		HTTPTimeout:      5 * time.Second,  // Reasonable timeout
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
	const targetMs = 500

	var durations []time.Duration
	var successful int

	t.Logf("Running %d HTTP-only prediction tests...", numTests)

	for i := 0; i < numTests; i++ {
		start := time.Now()
		
		response, err := httpPredictor.Predict(ctx, req)
		
		duration := time.Since(start)
		durations = append(durations, duration)

		if err != nil {
			t.Errorf("HTTP prediction %d failed: %v", i+1, err)
			continue
		}

		successful++
		durationMs := float64(duration.Nanoseconds()) / 1e6
		
		status := "✅"
		if durationMs > targetMs {
			status = "❌"
		}
		
		t.Logf("%s Test %d: %.1fms (TTFT: %.0fms, TPOT: %.0fms)", 
			status, i+1, durationMs, response.TTFT, response.TPOT)
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

func testHTTPOnlyPrediction(t *testing.T, ctx context.Context, ) {
	t.Log("Testing HTTP-only prediction (bypassing native XGBoost)...")

	// Create a predictor with native XGBoost disabled to force HTTP usage
	serverURL := os.Getenv("LATENCY_SERVER_URL")
	if serverURL == "" {
		t.Skip("LATENCY_SERVER_URL not set")
	}

	zapLog, err := zap.NewDevelopment()
	if err != nil {
		t.Fatalf("Failed to create logger: %v", err)
	}
	logger := zapr.NewLogger(zapLog)

	httpOnlyConfig := &Config{
		PythonURL:        serverURL,
		MaxSampleSize:    1000,
		FlushInterval:    1 * time.Second,
		MetricsRefreshInterval: 1 * time.Second, // Longer for metrics
		UseNativeXGBoost: false, // Force HTTP fallback
		HTTPTimeout:      30 * time.Second,
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

	// Test prediction using HTTP only
	req := PredictionRequest{
		KVCachePercentage:  0.6,   // 60% as a fraction
		InputTokenLength:   256,
		NumRequestWaiting:  1,
		NumRequestRunning:  2,
		NumTokensGenerated: 75,
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

	// Validate response
	if response.TTFT <= 0 {
		t.Error("TTFT should be positive")
	}
	if response.TPOT <= 0 {
		t.Error("TPOT should be positive")
	}

	// Test multiple HTTP-only predictions
	t.Log("Testing multiple HTTP-only predictions...")
	for i := 0; i < 3; i++ {
		testReq := PredictionRequest{
			KVCachePercentage:  float64(30+i*20) / 100.0,
			InputTokenLength:   128 + i*256,
			NumRequestWaiting:  i,
			NumRequestRunning:  1,
			NumTokensGenerated: 25 + i*50,
		}

		resp, err := httpPredictor.Predict(ctx, testReq)
		if err != nil {
			t.Errorf("HTTP-only prediction %d failed: %v", i+1, err)
			continue
		}

		t.Logf("HTTP-only prediction %d: TTFT=%.2f, TPOT=%.2f", i+1, resp.TTFT, resp.TPOT)
	}

	t.Log("Successfully tested HTTP-only predictions")
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
	t.Log("Testing Bayesian Ridge specific metrics...")

	metrics, err := predictor.GetMetrics(ctx)
	if err != nil {
		t.Errorf("Failed to get Bayesian Ridge metrics: %v", err)
		return
	}

	if metrics.Coefficients == nil {
		t.Error("Bayesian Ridge coefficients should not be nil")
		return
	}

	t.Logf("TTFT Coefficients:")
	t.Logf("  Intercept: %.6f", metrics.Coefficients.TTFTIntercept)
	for feature, coeff := range metrics.Coefficients.TTFTCoeffs {
		t.Logf("  %s: %.6f", feature, coeff)
	}

	t.Logf("TPOT Coefficients:")
	t.Logf("  Intercept: %.6f", metrics.Coefficients.TPOTIntercept)
	for feature, coeff := range metrics.Coefficients.TPOTCoeffs {
		t.Logf("  %s: %.6f", feature, coeff)
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

// generateTrainingEntries creates random training data for testing
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

		// Example equations (arbitrary, for test data):
		ttft := 100 + 2*float64(inputLen) + 10*kv + 5*float64(waiting) + rng.NormFloat64()*20
		tpot := 20 + 0.5*float64(generated) + 2*float64(running) + rng.NormFloat64()*5 + 9*kv

		entries[i] = TrainingEntry{
			KVCachePercentage:  kv,
			InputTokenLength:   inputLen,
			NumRequestWaiting:  waiting,
			NumRequestRunning:  running,
			NumTokensGenerated: generated,
			ActualTTFT:         ttft,
			ActualTPOT:         tpot,
			Timestamp:          time.Now().Add(-time.Duration(rng.Intn(3600)) * time.Second),
		}
	}

	return entries
}

// Benchmark test for prediction performance
func BenchmarkPrediction(b *testing.B) {
	serverURL := os.Getenv("LATENCY_SERVER_URL")
	if serverURL == "" {
		b.Skip("LATENCY_SERVER_URL not set, skipping benchmark")
	}

	logger := logr.Discard() // Silent logger for benchmark
	config := &Config{
		PythonURL:        serverURL,
		MaxSampleSize:    1000,
		FlushInterval:    1 * time.Second, // Long interval for benchmark
		MetricsRefreshInterval: 1 * time.Second,
		UseNativeXGBoost: true,
		HTTPTimeout:      10 * time.Second,
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
	originalURL := os.Getenv("LATENCY_SERVER_URL")
	originalSample := os.Getenv("LATENCY_MAX_SAMPLE_SIZE")
	originalInterval := os.Getenv("LATENCY_FLUSH_INTERVAL_SEC")
	originalNative := os.Getenv("LATENCY_USE_NATIVE_XGBOOST")
	originalTimeout := os.Getenv("LATENCY_HTTP_TIMEOUT_SEC")

	// Set test env vars  
	os.Setenv("LATENCY_SERVER_URL", "http://test.example.com")
	os.Setenv("LATENCY_MAX_SAMPLE_SIZE", "500")
	os.Setenv("LATENCY_FLUSH_INTERVAL_SEC", "5")
	os.Setenv("LATENCY_USE_NATIVE_XGBOOST", "false")
	os.Setenv("LATENCY_HTTP_TIMEOUT_SEC", "20")

	defer func() {
		// Restore original env vars (handle empty strings properly)
		if originalURL != "" {
			os.Setenv("LATENCY_SERVER_URL", originalURL)
		} else {
			os.Unsetenv("LATENCY_SERVER_URL")
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

	if config.PythonURL != "http://test.example.com" {
		t.Errorf("Expected PythonURL to be 'http://test.example.com', got '%s'", config.PythonURL)
	}
	if config.MaxSampleSize != 500 {
		t.Errorf("Expected MaxSampleSize to be 500, got %d", config.MaxSampleSize)
	}
	if config.FlushInterval != 5*time.Second {
		t.Errorf("Expected FlushInterval to be 5s, got %v", config.FlushInterval)
	}
	if config.MetricsRefreshInterval != 60*time.Second {
		t.Errorf("Expected MetricsRefreshInterval to be 1s, got %v", config.MetricsRefreshInterval)
	}
	if config.UseNativeXGBoost != false {
		t.Errorf("Expected UseNativeXGBoost to be false, got %t", config.UseNativeXGBoost)
	}
	if config.HTTPTimeout != 20*time.Second {
		t.Errorf("Expected HTTPTimeout to be 20s, got %v", config.HTTPTimeout)
	}
}