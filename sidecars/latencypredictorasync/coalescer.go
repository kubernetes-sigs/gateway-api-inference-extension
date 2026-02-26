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

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
)

// batchSubmission represents a single caller's request to PredictBulkStrict
// waiting to be coalesced with other concurrent callers.
type batchSubmission struct {
	requests []PredictionRequest
	resultCh chan batchResult
}

// batchResult carries the response (or error) back to the caller.
type batchResult struct {
	response *BulkPredictionResponse
	err      error
}

// coalesceDispatcher is a background goroutine that accumulates PredictBulkStrict
// submissions and dispatches them as a single mega-batch HTTP call once either:
//   - the coalesce window expires (e.g. 5ms since first queued item), or
//   - the accumulated row count reaches MaxCoalescedRows (if > 0).
//
// Batches are dispatched asynchronously (in a goroutine) so the dispatcher
// never blocks waiting for an HTTP response. Concurrency is capped by
// MaxConcurrentDispatches via a semaphore.
func (p *Predictor) coalesceDispatcher() {
	defer p.wg.Done()

	var pending []*batchSubmission
	var totalRows int
	var timerCh <-chan time.Time
	var timer *time.Timer

	dispatch := func() {
		if len(pending) == 0 {
			return
		}
		batch := pending
		pending = nil
		totalRows = 0

		// Acquire semaphore slot inside a goroutine so the dispatcher is never
		// blocked and can keep accumulating new submissions while HTTP calls are
		// in flight.
		go func() {
			select {
			case <-p.dispatchSem:
			case <-p.done:
				for _, sub := range batch {
					sub.resultCh <- batchResult{err: errors.New("predictor shutting down")}
				}
				return
			}
			defer func() { p.dispatchSem <- struct{}{} }()
			p.dispatchBatch(batch)
		}()
	}

	for {
		select {
		case sub := <-p.coalesceCh:
			pending = append(pending, sub)
			totalRows += len(sub.requests)

			// Start the coalesce window timer on the first submission
			if timer == nil {
				timer = time.NewTimer(p.config.CoalesceWindow)
				timerCh = timer.C
			}

			// Early dispatch if row cap is set and reached
			if p.config.MaxCoalescedRows > 0 && totalRows >= p.config.MaxCoalescedRows {
				timer.Stop()
				timer = nil
				timerCh = nil
				dispatch()
			}

		case <-timerCh:
			// Coalesce window expired — dispatch whatever we have
			timer = nil
			timerCh = nil
			dispatch()

		case <-p.done:
			// Shutting down — reject all pending submissions
			if timer != nil {
				timer.Stop()
			}
			for _, sub := range pending {
				sub.resultCh <- batchResult{err: errors.New("predictor shutting down")}
			}
			return
		}
	}
}

// dispatchBatch merges all pending submissions into one HTTP call, then fans
// the results back to each individual caller.
func (p *Predictor) dispatchBatch(submissions []*batchSubmission) {
	// Merge all requests and track each caller's slice boundaries
	var totalReqs int
	for _, sub := range submissions {
		totalReqs += len(sub.requests)
	}
	allRequests := make([]PredictionRequest, 0, totalReqs)
	offsets := make([]int, len(submissions))
	counts := make([]int, len(submissions))

	offset := 0
	for i, sub := range submissions {
		offsets[i] = offset
		counts[i] = len(sub.requests)
		allRequests = append(allRequests, sub.requests...)
		offset += len(sub.requests)
	}

	p.logger.V(logutil.DEBUG).Info("Dispatching coalesced batch",
		"callers", len(submissions),
		"total_items", len(allRequests))

	// Make one HTTP call with the mega-batch
	ctx, cancel := context.WithTimeout(context.Background(), p.config.HTTPTimeout)
	defer cancel()

	resp, err := p.doPredictBulkStrictHTTP(ctx, allRequests)

	// Fan out results to each caller
	for i, sub := range submissions {
		if err != nil {
			sub.resultCh <- batchResult{err: err}
			continue
		}

		start := offsets[i]
		end := start + counts[i]

		// Guard against response length mismatch
		if end > len(resp.Predictions) {
			sub.resultCh <- batchResult{err: fmt.Errorf(
				"coalesced response too short: expected items [%d:%d] but got %d predictions",
				start, end, len(resp.Predictions))}
			continue
		}

		// Slice out this caller's portion of the response
		callerPredictions := make([]PredictionResponse, counts[i])
		copy(callerPredictions, resp.Predictions[start:end])

		sub.resultCh <- batchResult{
			response: &BulkPredictionResponse{
				Predictions:           callerPredictions,
				TotalRequests:         counts[i],
				SuccessfulPredictions: counts[i],
				FailedPredictions:     0,
				ProcessingTimeMs:      resp.ProcessingTimeMs,
			},
		}
	}
}

// doPredictBulkStrictHTTP performs the raw HTTP call for a strict bulk prediction.
// This is the internal method used by both the direct and coalesced paths.
func (p *Predictor) doPredictBulkStrictHTTP(ctx context.Context, requests []PredictionRequest) (*BulkPredictionResponse, error) {
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
		return nil, fmt.Errorf("bulk prediction server returned non-200 status: %d %s, body: %s",
			resp.StatusCode, resp.Status, string(body))
	}

	var bulkResp BulkPredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&bulkResp); err != nil {
		return nil, fmt.Errorf("failed to decode bulk prediction response: %w", err)
	}

	return &bulkResp, nil
}

// submitCoalesced sends a set of prediction requests through the coalescer
// and blocks until the coalesced result is available.
func (p *Predictor) submitCoalesced(ctx context.Context, requests []PredictionRequest) (*BulkPredictionResponse, error) {
	sub := &batchSubmission{
		requests: requests,
		resultCh: make(chan batchResult, 1),
	}

	select {
	case p.coalesceCh <- sub:
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-p.done:
		return nil, errors.New("predictor shutting down")
	}

	select {
	case result := <-sub.resultCh:
		return result.response, result.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
