package latencypredictorasync

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
)

// TestBackgroundPredictIntegration assumes a real predictor server is running.
// Set LATENCY_SERVER_URL to point at it before running.
func TestBackgroundPredictIntegration(t *testing.T) {
	url := os.Getenv("LATENCY_SERVER_URL")
	if url == "" {
		t.Skip("Skipping integration: LATENCY_SERVER_URL not set")
	}

	logger := testr.New(t)
	cfg := &Config{
		PythonURL:     url,
		MaxSampleSize: 1000,
		FlushInterval: 1 * time.Second,
	}
	p := New(cfg, logger)
	defer p.Stop()

	// Wait for at least one metric refresh
	time.Sleep(cfg.FlushInterval + 1000*time.Millisecond)

	// Grab cached metrics
	mr, ok := p.GetCachedMetrics()
	if !ok || mr.Coefficients == nil {
		t.Fatalf("no metrics in cache after refresh")
	}
	c := mr.Coefficients

	// Build a simple prediction request using one feature for which we know a coefficient
	// We'll set only one non-zero feature: input_token_length = 100
	req := PredictionRequest{InputTokenLength: 100}

	// Calculate expected TTFT = intercept + coef_input_token_length * 100
	expTTFT := c.TTFTIntercept + c.TTFTCoeffs["input_token_length"]*100

	// Calculate expected TPOT = intercept + coef_num_tokens_generated * 0 (zero input)
	expTPOT := c.TPOTIntercept

	resp, err := p.Predict(context.Background(), req)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	if resp.TTFT != expTTFT {
		t.Errorf("Predict TTFT: expected %.6f, got %.6f", expTTFT, resp.TTFT)
	}
	if resp.TPOT != expTPOT {
		t.Errorf("Predict TPOT: expected %.6f, got %.6f", expTPOT, resp.TPOT)
	}
}

// TestAddTrainingDataBulkMethod tests that calling AddTrainingDataBulk buffers entries
// and that flushTraining sends them to the server.
func TestAddTrainingDataBulkMethod(t *testing.T) {
	// Capture incoming bulk training requests
	var received BulkTrainingRequest
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/add_training_data_bulk" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusAccepted)
	}))
	defer ts.Close()

	logger := testr.New(t)
	cfg := &Config{
		PythonURL:     ts.URL,
		MaxSampleSize: 1000,
		FlushInterval: 1 * time.Second,
	}
	p := New(cfg, logger)
	// Override the HTTP client so flushTraining hits our fake server
	p.httpClient = ts.Client()
	defer p.Stop()

	// Buffer two entries
	entries := []TrainingEntry{
		{KVCachePercentage: 0.5, InputTokenLength: 10, NumRequestWaiting: 2, NumRequestRunning: 1, NumTokensGenerated: 4, ActualTTFT: 150.0, ActualTPOT: 70.0, Timestamp: time.Now()},
		{KVCachePercentage: 0.6, InputTokenLength: 20, NumRequestWaiting: 3, NumRequestRunning: 2, NumTokensGenerated: 8, ActualTTFT: 160.0, ActualTPOT: 80.0, Timestamp: time.Now()},
	}
	if err := p.AddTrainingDataBulk(entries); err != nil {
		t.Fatalf("AddTrainingDataBulk error: %v", err)
	}

	// Manually flush now that MaxSampleSize is sufficient
	p.flushTraining()

	// Expect server to have received exactly the two entries
	if len(received.Entries) != len(entries) {
		t.Errorf("expected %d entries, got %d", len(entries), len(received.Entries))
	}

	// Buffer now should be empty
	if len(p.pending) != 0 {
		t.Errorf("expected pending buffer to be empty after flush, got %d", len(p.pending))
	}
}
