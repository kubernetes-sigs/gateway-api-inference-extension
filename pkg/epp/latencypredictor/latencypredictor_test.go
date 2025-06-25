// Package latencypredictor provides a Go client for the Python-based
// latency prediction service.
package latencypredictor

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
)

// --- Test Helpers ---

// contains is a helper to check if a substring exists in a string.
func contains(s, substr string) bool {
	return strings.Contains(s, substr)
}

// --- Unit Tests ---

func TestConfigFromEnv(t *testing.T) {
	t.Run("with env var set", func(t *testing.T) {
		testURL := "http://test-server:9000"
		t.Setenv("LATENCY_SERVER_URL", testURL)
		cfg := ConfigFromEnv()
		if cfg.PythonURL != testURL {
			t.Errorf("expected PythonURL to be '%s', got '%s'", testURL, cfg.PythonURL)
		}
	})

	t.Run("with env var unset", func(t *testing.T) {
		// Temporarily unset the environment variable for this specific test
		// and ensure it gets restored after the test runs.
		originalValue, wasSet := os.LookupEnv("LATENCY_SERVER_URL")
		os.Unsetenv("LATENCY_SERVER_URL")
		t.Cleanup(func() {
			if wasSet {
				os.Setenv("LATENCY_SERVER_URL", originalValue)
			}
		})

		cfg := ConfigFromEnv()
		if cfg.PythonURL != "http://localhost:8000" {
			t.Errorf("expected default PythonURL when env var unset, got '%s'", cfg.PythonURL)
		}
	})
}

func TestNetworkErrors(t *testing.T) {
    // Create predictor with an invalid URL that will cause a network error.
    config := &Config{PythonURL: "http://localhost:9999"}
    logger := testr.New(t)
    p := New(config, logger)

    t.Run("Predict network error", func(t *testing.T) {
        _, err := p.Predict(PredictionRequest{})
        if err == nil {
            t.Fatal("expected a network error but got none")
        }
        if !contains(err.Error(), "failed to call Python /predict endpoint") {
            t.Errorf("expected error message to indicate a connection failure, got: %v", err)
        }
    })

    t.Run("BulkAdd network error", func(t *testing.T) {
        err := p.AddTrainingDataBulk([]TrainingEntry{})
        if err == nil {
            t.Fatal("expected a network error but got none")
        }
        // should mention the bulk path so we know it tried that endpoint
        if !contains(err.Error(), "/add_training_data_bulk") {
            t.Errorf("expected error to mention /add_training_data_bulk, got: %v", err)
        }
    })
}

// --- Integration Test ---
// This test runs against a live Python server.
// Set the LATENCY_SERVER_URL environment variable to enable it.
// Example: LATENCY_SERVER_URL=http://localhost:8000 go test -v -run TestIntegration
func TestIntegration_AddDataThenPredict(t *testing.T) {
	serverURL := os.Getenv("LATENCY_SERVER_URL")
	if serverURL == "" {
		t.Skip("Skipping integration test: LATENCY_SERVER_URL environment variable is not set")
	}

	logger := testr.New(t)
	config := &Config{PythonURL: serverURL}
	predictor := New(config, logger)

	// Step 1: Send a training sample to the live server
	trainingSample := TrainingEntry{
		KVCachePercentage: 0.8,
		InputTokenLength:  256,
		NumRequestWaiting: 10,
		NumRequestRunning: 4,
		ActualTTFT:        800.0,
		ActualTPOT:        75.0,
		NumTokensGenerated: 1000,
		Timestamp:         time.Now(),
	}
	trainingJSON, _ := json.MarshalIndent(trainingSample, "", "  ")
	t.Logf("Sending training sample to %s:\n%s", serverURL, string(trainingJSON))

	err := predictor.AddTrainingDataBulk([]TrainingEntry{trainingSample})
	if err != nil {
		t.Fatalf("Failed to add training sample during integration test: %v", err)
	}
	t.Log("Successfully sent training sample.")

	// Step 2: Request a prediction from the live server
	predictionRequest := PredictionRequest{
		KVCachePercentage: 0.8,
		InputTokenLength:  256,
		NumRequestWaiting: 10,
		NumRequestRunning: 4,
		NumTokensGenerated: 1000,
	}
	predictionJSON, _ := json.MarshalIndent(predictionRequest, "", "  ")
	t.Logf("Requesting prediction from %s with body:\n%s", serverURL, string(predictionJSON))

	result, err := predictor.Predict(predictionRequest)
	if err != nil {
		t.Fatalf("Failed to get prediction during integration test: %v", err)
	}
	resultJSON, _ := json.MarshalIndent(result, "", "  ")
	t.Logf("Successfully received prediction:\n%s", string(resultJSON))

	// Step 3: Perform basic validation on the result
	if result.TTFT <= 0 {
		t.Errorf("Expected a positive TTFT value, but got %f", result.TTFT)
	}
	if result.TPOT <= 0 {
		t.Errorf("Expected a positive TPOT value, but got %f", result.TPOT)
	}
	if result.PredictedAt.IsZero() {
		t.Error("Expected a valid 'PredictedAt' timestamp, but it was zero")
	}
}


func TestIntegration_MetricsAndCache(t *testing.T) {
	serverURL := os.Getenv("LATENCY_SERVER_URL")
	if serverURL == "" {
		t.Skip("Skipping integration test: LATENCY_SERVER_URL environment variable is not set")
	}

	logger := testr.New(t)
	config := &Config{PythonURL: serverURL}
	predictor := New(config, logger)

	// First fetch: populate both remote and cache
	t.Logf("Fetching metrics from %s/metrics", serverURL)
	metrics, err := predictor.GetMetrics()
	if err != nil {
		t.Fatalf("GetMetrics failed: %v", err)
	}

	metricsJSON, _ := json.MarshalIndent(metrics, "", "  ")
	t.Logf("Metrics payload:\n%s", string(metricsJSON))

	// Basic validation
	if metrics == nil || len(metrics.RawMetrics) == 0 {
		t.Fatal("Expected non-empty RawMetrics")
	}

	// Now test the cache
	cached, ok := predictor.GetCachedMetrics()
	if !ok {
		t.Fatal("Expected cache to be populated, but GetCachedMetrics returned ok=false")
	}

	// Compare RawMetrics from cache with the one we just fetched
	if cached.RawMetrics != metrics.RawMetrics {
		t.Error("Cached RawMetrics does not match the last fetched metrics")
	}

	// If structured data was parsed, ensure it matches too
	if metrics.Coefficients != nil {
		if cached.Coefficients == nil {
			t.Error("Expected cached.Coefficients to be non-nil")
		} else if cached.Coefficients.TTFTIntercept != metrics.Coefficients.TTFTIntercept {
			t.Errorf("Cached TTFTIntercept (%f) != fetched (%f)",
				cached.Coefficients.TTFTIntercept, metrics.Coefficients.TTFTIntercept)
		}
	}

	if metrics.BucketCounts != nil {
		if cached.BucketCounts == nil {
			t.Error("Expected cached.BucketCounts to be non-nil")
		} else if len(cached.BucketCounts.TTFTBuckets) != len(metrics.BucketCounts.TTFTBuckets) {
			t.Errorf("Cached TTFTBuckets length (%d) != fetched (%d)",
				len(cached.BucketCounts.TTFTBuckets), len(metrics.BucketCounts.TTFTBuckets))
		}
	}

	// Finally, ensure GetMetrics still works a second time
	metrics2, err := predictor.GetMetrics()
	if err != nil {
		t.Fatalf("Second GetMetrics call failed: %v", err)
	}
	if metrics2.RawMetrics == "" {
		t.Error("Second GetMetrics returned empty RawMetrics")
	}
}