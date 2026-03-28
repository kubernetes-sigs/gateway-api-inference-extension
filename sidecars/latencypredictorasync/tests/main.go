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

package main

import (
	"context"
	"log"
	"math"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/stdr"
	"golang.org/x/time/rate"
	"sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

type TestMetrics struct {
	TotalRequests         int64
	SuccessfulRequests    int64
	FailedRequests        int64
	TotalLatencyMs        int64
	MinLatency            int64 // accessed via atomic operations only
	MaxLatency            int64 // accessed via atomic operations only
	Latencies             []int64
	LatenciesMutex        sync.Mutex
	TotalPredictionsCount int64
	SuccessfulPredictions int64
	FailedPredictions     int64
}

type TrainingMetrics struct {
	TotalBatches    int64
	TotalEntries    int64
	SuccessfulFlush int64
	FailedFlush     int64
}

func main() {

	trainingServerURL := os.Getenv("TRAINING_SERVER_URL")
	if trainingServerURL == "" {
		trainingServerURL = "http://training-service:8000"
	}

	predictionServersStr := os.Getenv("PREDICTION_SERVERS")
	if predictionServersStr == "" {
		log.Fatal("PREDICTION_SERVERS env var not set")
	}
	predictionServers := strings.Split(predictionServersStr, ",")
	for i := range predictionServers {
		predictionServers[i] = strings.TrimSpace(predictionServers[i])
	}

	testQPS := parseEnvInt("TEST_QPS", 100)
	testDurationSeconds := parseEnvInt("TEST_DURATION_SECONDS", 120)
	maxBulkSize := parseEnvInt("MAX_BULK_SIZE", 200)
	numEndpointsPerRequest := parseEnvInt("NUM_ENDPOINTS_PER_REQUEST", 10)
	trainingBatchSize := parseEnvInt("TRAINING_BATCH_SIZE", 100)
	trainingIntervalMs := parseEnvInt("TRAINING_INTERVAL_MS", 500)
	maxConcurrentDispatches := parseEnvInt("MAX_CONCURRENT_DISPATCHES", 8)
	// maxInFlightRequests caps concurrent PredictBulkStrict goroutines.
	// Without this, goroutines accumulate at (testQPS - throughput)/sec and OOM
	// the pod when the system can't keep up with the target QPS.
	maxInFlightRequests := parseEnvInt("MAX_INFLIGHT_REQUESTS", testQPS/5+500)
	// coalesceWindowMs and maxCoalescedCallers together control batch size.
	// Smaller values → smaller batches → lower Python latency but more HTTP calls.
	// MaxCoalescedRows = numEndpointsPerRequest × maxCoalescedCallers.
	coalesceWindowMs := parseEnvInt("COALESCE_WINDOW_MS", 5)
	maxCoalescedCallers := parseEnvInt("MAX_COALESCED_CALLERS", 50)

	if err := os.WriteFile("/tmp/test_running", []byte("running"), 0644); err != nil {
		log.Printf("Warning: could not create test_running marker: %v", err)
	}
	// Removed defer — cleaned up explicitly before all exit points to satisfy gocritic.

	// Create logger using stdr (standard log -> logr.Logger)
	logger := stdr.New(log.New(os.Stdout, "[predictor-test] ", log.LstdFlags))

	logger.Info("Predictor Client Test starting",
		"prediction_servers", predictionServers,
		"training_server", trainingServerURL,
		"target_qps", testQPS,
		"duration_seconds", testDurationSeconds,
		"endpoints_per_request", numEndpointsPerRequest,
		"training_batch_size", trainingBatchSize,
		"training_interval_ms", trainingIntervalMs)

	// Create predictor config
	cfg := &latencypredictorasync.Config{
		PredictionURLs:          predictionServers,
		TrainingURL:             trainingServerURL,
		MaxBulkSize:             maxBulkSize,
		MetricsRefreshInterval:  30 * time.Second,
		FlushInterval:           time.Duration(trainingIntervalMs) * time.Millisecond,
		HTTPTimeout:             10 * time.Second,
		MaxSampleSize:           1000,
		UseNativeXGBoost:        false,
		CoalesceWindow:          time.Duration(coalesceWindowMs) * time.Millisecond,
		MaxCoalescedRows:        numEndpointsPerRequest * maxCoalescedCallers,
		MaxConcurrentDispatches: maxConcurrentDispatches,
	}

	// Create predictor
	predictor := latencypredictorasync.New(cfg, logger)

	testCtx, cancel := context.WithTimeout(context.Background(), time.Duration(testDurationSeconds)*time.Second+30*time.Second)

	if err := predictor.Start(testCtx); err != nil {
		cancel()
		os.Remove("/tmp/test_running")
		logger.Error(err, "Failed to start predictor")
		return
	}

	predMetrics := &TestMetrics{
		MinLatency: math.MaxInt64,
		Latencies:  make([]int64, 0, testQPS*testDurationSeconds),
	}

	trainMetrics := &TrainingMetrics{}

	testStart := time.Now()
	testEnd := testStart.Add(time.Duration(testDurationSeconds) * time.Second)

	logger.Info("Starting test requests...")

	// Use separate WaitGroups so generators don't block on in-flight requests
	var generatorWg sync.WaitGroup // for generator goroutines
	var requestWg sync.WaitGroup   // for in-flight prediction requests

	// Semaphore that caps concurrent PredictBulkStrict goroutines.
	// Without this, goroutines pile up at (testQPS - throughput)/sec and OOM
	// the pod when the server can't keep up with the target QPS.
	inFlightSem := make(chan struct{}, maxInFlightRequests)

	// ---------------------------------------------------------------
	// Goroutine 1: Continuously send training data in parallel
	// ---------------------------------------------------------------
	generatorWg.Go(func() {

		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		trainingTicker := time.NewTicker(time.Duration(trainingIntervalMs) * time.Millisecond)
		defer trainingTicker.Stop()

		logger.Info("Training data generator started",
			"batch_size", trainingBatchSize,
			"interval_ms", trainingIntervalMs)

		for {
			select {
			case <-testCtx.Done():
				return
			case <-trainingTicker.C:
				if time.Now().After(testEnd) {
					return
				}

				entries := generateTrainingBatch(rng, trainingBatchSize)
				atomic.AddInt64(&trainMetrics.TotalBatches, 1)
				atomic.AddInt64(&trainMetrics.TotalEntries, int64(len(entries)))

				if err := predictor.AddTrainingDataBulk(entries); err != nil {
					atomic.AddInt64(&trainMetrics.FailedFlush, 1)
					logger.Error(err, "Failed to buffer training data")
				} else {
					atomic.AddInt64(&trainMetrics.SuccessfulFlush, 1)
				}
			}
		}
	})

	// ---------------------------------------------------------------
	// Goroutine 2: Generate prediction requests at target QPS.
	//
	// A single ticker fires at most once per ~1ms (Go timer resolution),
	// capping it at ~1k QPS. For higher rates we use a token-bucket
	// rate.Limiter shared across a pool of worker goroutines, each of
	// which calls Wait() and fires one request. This accurately sustains
	// rates well above 10k QPS.
	// ---------------------------------------------------------------
	// One worker per 100 QPS; min resolution per worker = 10ms.
	numWorkers := min(max(testQPS/100, 10), 500)
	// Burst = numWorkers so at most one token per worker fires immediately at
	// start, giving a clean ramp-up rather than a 10% spike.
	limiter := rate.NewLimiter(rate.Limit(testQPS), numWorkers)

	logger.Info("Prediction request generator started",
		"target_qps", testQPS,
		"workers", numWorkers)

	dispatchRequest := func() {
		// Block the rate-limiter worker when too many requests are in flight.
		// This applies backpressure instead of letting goroutines pile up and OOM.
		select {
		case inFlightSem <- struct{}{}:
		case <-testCtx.Done():
			return
		}
		requestWg.Go(func() {
			defer func() { <-inFlightSem }()

			requests := make([]latencypredictorasync.PredictionRequest, numEndpointsPerRequest)
			for i := range numEndpointsPerRequest {
				requests[i] = latencypredictorasync.PredictionRequest{
					KVCachePercentage:     float64(i%10) * 0.1,
					InputTokenLength:      512 + i*64,
					NumRequestWaiting:     i % 5,
					NumRequestRunning:     (i % 3) + 1,
					NumTokensGenerated:    256,
					PrefixCacheScore:      float64(i%10) * 0.1,
					PodType:               "monolithic",
					PrefillTokensInFlight: int64(i * 1500),
					DecodeTokensInFlight:  int64(i * 500),
				}
			}

			start := time.Now()
			resp, err := predictor.PredictBulkStrict(testCtx, requests)
			latency := time.Since(start).Milliseconds()

			atomic.AddInt64(&predMetrics.TotalRequests, 1)

			if err != nil {
				atomic.AddInt64(&predMetrics.FailedRequests, 1)
				return
			}

			atomic.AddInt64(&predMetrics.SuccessfulRequests, 1)
			atomic.AddInt64(&predMetrics.TotalLatencyMs, latency)
			atomic.AddInt64(&predMetrics.TotalPredictionsCount, int64(len(resp.Predictions)))
			atomic.AddInt64(&predMetrics.SuccessfulPredictions, int64(resp.SuccessfulPredictions))
			atomic.AddInt64(&predMetrics.FailedPredictions, int64(resp.FailedPredictions))

			for {
				cur := atomic.LoadInt64(&predMetrics.MinLatency)
				if latency >= cur || atomic.CompareAndSwapInt64(&predMetrics.MinLatency, cur, latency) {
					break
				}
			}
			for {
				cur := atomic.LoadInt64(&predMetrics.MaxLatency)
				if latency <= cur || atomic.CompareAndSwapInt64(&predMetrics.MaxLatency, cur, latency) {
					break
				}
			}

			predMetrics.LatenciesMutex.Lock()
			predMetrics.Latencies = append(predMetrics.Latencies, latency)
			predMetrics.LatenciesMutex.Unlock()
		})
	}

	for range numWorkers {
		generatorWg.Go(func() {
			for {
				if err := limiter.Wait(testCtx); err != nil {
					return // context cancelled or deadline exceeded
				}
				if time.Now().After(testEnd) {
					return
				}
				dispatchRequest()
			}
		})
	}

	// ---------------------------------------------------------------
	// Goroutine 3: Periodic stats printer
	// ---------------------------------------------------------------
	generatorWg.Go(func() {

		statsTicker := time.NewTicker(10 * time.Second)
		defer statsTicker.Stop()

		for {
			select {
			case <-testCtx.Done():
				return
			case <-statsTicker.C:
				if time.Now().After(testEnd) {
					return
				}

				req := atomic.LoadInt64(&predMetrics.TotalRequests)
				succ := atomic.LoadInt64(&predMetrics.SuccessfulRequests)
				failed := atomic.LoadInt64(&predMetrics.FailedRequests)
				totalLat := atomic.LoadInt64(&predMetrics.TotalLatencyMs)

				trainBatches := atomic.LoadInt64(&trainMetrics.TotalBatches)
				trainEntries := atomic.LoadInt64(&trainMetrics.TotalEntries)
				trainOK := atomic.LoadInt64(&trainMetrics.SuccessfulFlush)
				trainFail := atomic.LoadInt64(&trainMetrics.FailedFlush)

				var avgLat float64
				if succ > 0 {
					avgLat = float64(totalLat) / float64(succ)
				}

				logger.Info("Progress",
					"pred_requests", req,
					"pred_ok", succ,
					"pred_fail", failed,
					"avg_latency_ms", avgLat,
					"min_ms", atomic.LoadInt64(&predMetrics.MinLatency),
					"max_ms", atomic.LoadInt64(&predMetrics.MaxLatency),
					"train_batches", trainBatches,
					"train_entries", trainEntries,
					"train_ok", trainOK,
					"train_fail", trainFail)
			}
		}
	})

	// Wait for generators to finish (test duration elapsed)
	generatorWg.Wait()

	// Wait for all in-flight prediction requests to complete
	requestWg.Wait()

	// ---------------------------------------------------------------
	// Calculate percentiles
	// ---------------------------------------------------------------
	predMetrics.LatenciesMutex.Lock()
	slices.Sort(predMetrics.Latencies)
	var p50, p99, p999 int64
	n := len(predMetrics.Latencies)
	if n > 0 {
		p50 = predMetrics.Latencies[n*50/100]
		if idx := n * 99 / 100; idx < n {
			p99 = predMetrics.Latencies[idx]
		}
		if idx := n * 999 / 1000; idx < n {
			p999 = predMetrics.Latencies[idx]
		}
	}
	predMetrics.LatenciesMutex.Unlock()

	totalReq := atomic.LoadInt64(&predMetrics.TotalRequests)
	successReq := atomic.LoadInt64(&predMetrics.SuccessfulRequests)
	failedReq := atomic.LoadInt64(&predMetrics.FailedRequests)
	totalLat := atomic.LoadInt64(&predMetrics.TotalLatencyMs)
	totalPred := atomic.LoadInt64(&predMetrics.TotalPredictionsCount)
	succPred := atomic.LoadInt64(&predMetrics.SuccessfulPredictions)
	failPred := atomic.LoadInt64(&predMetrics.FailedPredictions)

	trainBatches := atomic.LoadInt64(&trainMetrics.TotalBatches)
	trainEntries := atomic.LoadInt64(&trainMetrics.TotalEntries)
	trainOK := atomic.LoadInt64(&trainMetrics.SuccessfulFlush)
	trainFail := atomic.LoadInt64(&trainMetrics.FailedFlush)

	var avgLatency float64
	if successReq > 0 {
		avgLatency = float64(totalLat) / float64(successReq)
	}

	// ---------------------------------------------------------------
	// Print final results
	// ---------------------------------------------------------------
	var reqSuccessRate, predSuccessRate, trainSuccessRate float64
	if totalReq > 0 {
		reqSuccessRate = float64(successReq) / float64(totalReq) * 100
	}
	if totalPred > 0 {
		predSuccessRate = float64(succPred) / float64(totalPred) * 100
	}
	if trainBatches > 0 {
		trainSuccessRate = float64(trainOK) / float64(trainBatches) * 100
	}

	logger.Info("TEST RESULTS - Predictions",
		"total_requests", totalReq,
		"successful", successReq,
		"failed", failedReq,
		"request_success_rate_pct", reqSuccessRate,
		"total_predictions", totalPred,
		"successful_predictions", succPred,
		"failed_predictions", failPred,
		"prediction_success_rate_pct", predSuccessRate,
		"avg_latency_ms", avgLatency,
		"min_latency_ms", atomic.LoadInt64(&predMetrics.MinLatency),
		"max_latency_ms", atomic.LoadInt64(&predMetrics.MaxLatency),
		"p50_ms", p50,
		"p99_ms", p99,
		"p999_ms", p999,
		"achieved_qps", float64(successReq)/time.Since(testStart).Seconds())

	logger.Info("TEST RESULTS - Training",
		"batches_sent", trainBatches,
		"batches_ok", trainOK,
		"batches_fail", trainFail,
		"entries_sent", trainEntries,
		"success_rate_pct", trainSuccessRate)

	// Cleanup
	cancel()
	stopCtx, stopCancel := context.WithTimeout(context.Background(), 10*time.Second)
	predictor.Stop(stopCtx)
	stopCancel()

	if failedReq > 0 {
		logger.Info("WARNING: Test had failed prediction requests", "failed_count", failedReq)
		os.Remove("/tmp/test_running")
		os.Exit(1)
	}

	os.Remove("/tmp/test_running")
	logger.Info("Test completed successfully!")
}

// generateTrainingBatch creates a batch of realistic training entries.
func generateTrainingBatch(rng *rand.Rand, batchSize int) []latencypredictorasync.TrainingEntry {
	entries := make([]latencypredictorasync.TrainingEntry, batchSize)

	for i := range batchSize {
		inputTokens := 128 + rng.Intn(2048)
		// Pick queue bucket uniformly across all 5 buckets: [0], [1-2], [3-5], [6-10], [11+]
		var numWaiting int
		switch rng.Intn(5) {
		case 0:
			numWaiting = 0
		case 1:
			numWaiting = rng.Intn(2) + 1 // 1-2
		case 2:
			numWaiting = rng.Intn(3) + 3 // 3-5
		case 3:
			numWaiting = rng.Intn(5) + 6 // 6-10
		default:
			numWaiting = rng.Intn(5) + 11 // 11-15
		}
		numRunning := rng.Intn(8) + 1
		kvCache := rng.Float64()
		prefixCache := rng.Float64()
		numTokensGenerated := 64 + rng.Intn(512)
		prefillTIF := int64(rng.Intn(15000))
		decodeTIF := int64(rng.Intn(5000))

		// Simulate realistic TTFT
		baseTTFT := float64(inputTokens)*0.05 +
			float64(prefillTIF)*0.05 +
			float64(numWaiting)*10.0 +
			float64(numRunning)*5.0

		if kvCache > 0.7 {
			baseTTFT *= 0.3
		} else if kvCache > 0.4 {
			baseTTFT *= 0.6
		}

		if prefixCache > 0.5 {
			baseTTFT *= 0.8
		}

		ttftMs := math.Max(1.0, baseTTFT+(rng.Float64()-0.5)*baseTTFT*0.2)

		// Simulate realistic TPOT
		baseTPOT := 8.0 +
			float64(decodeTIF)*0.02 +
			float64(numRunning-1)*2.0 +
			(rng.Float64()-0.5)*2.0

		tpotMs := math.Max(1.0, baseTPOT*float64(numTokensGenerated))

		// Mix of TTFT-only, TPOT-only, and both entries
		actualTTFT := 0.0
		actualTPOT := 0.0
		switch i % 3 {
		case 0:
			actualTTFT = math.Round(ttftMs*100) / 100
		case 1:
			actualTPOT = math.Round(tpotMs*100) / 100
		case 2:
			actualTTFT = math.Round(ttftMs*100) / 100
			actualTPOT = math.Round(tpotMs*100) / 100
		}

		entries[i] = latencypredictorasync.TrainingEntry{
			KVCachePercentage:     kvCache,
			InputTokenLength:      inputTokens,
			NumRequestWaiting:     numWaiting,
			NumRequestRunning:     numRunning,
			NumTokensGenerated:    numTokensGenerated,
			ActualTTFT:            actualTTFT,
			ActualTPOT:            actualTPOT,
			PrefixCacheScore:      prefixCache,
			PrefillTokensInFlight: prefillTIF,
			DecodeTokensInFlight:  decodeTIF,
			Timestamp:             time.Now(),
		}
	}

	return entries
}

// parseEnvInt reads an integer environment variable, logging a warning on parse
// failure, and returns the default value when the variable is unset or invalid.
func parseEnvInt(key string, defaultVal int) int {
	raw := os.Getenv(key)
	if raw == "" {
		return defaultVal
	}
	val, err := strconv.Atoi(raw)
	if err != nil {
		log.Printf("Warning: invalid value %q for %s, using default %d", raw, key, defaultVal)
		return defaultVal
	}
	if val == 0 {
		return defaultVal
	}
	return val
}
