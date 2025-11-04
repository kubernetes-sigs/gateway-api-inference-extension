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
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// GetMetrics fetches & parses metrics from the training server (for Bayesian Ridge).
func (p *Predictor) GetMetrics(ctx context.Context) (*MetricsResponse, error) {
	url := p.config.TrainingURL + "/metrics"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create metrics request: %w", err)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call training server /metrics endpoint: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("training server returned non-200 status: %d %s, body: %s", resp.StatusCode, resp.Status, string(body))
	}

	rawMetricsBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read metrics response body: %w", err)
	}
	rawMetrics := string(rawMetricsBytes)

	metricsResponse := &MetricsResponse{
		RawMetrics: rawMetrics,
		ModelType:  bayesianRidgeModelType, // Assume Bayesian Ridge when calling /metrics
	}

	coeffs, buckets, err := p.parsePrometheusMetrics(rawMetrics)
	if err != nil {
		p.logger.Error(err, "Failed to parse Prometheus metrics, caching raw only")
	} else {
		metricsResponse.Coefficients = coeffs
		metricsResponse.BucketCounts = buckets
	}

	p.metricsMu.Lock()
	p.cachedMetrics = metricsResponse
	p.metricsMu.Unlock()

	p.logger.V(logutil.DEBUG).Info("Successfully retrieved and cached Bayesian Ridge metrics.")
	return metricsResponse, nil
}

// parsePrometheusMetrics parses the Prometheus-format metrics into structured data.
func (p *Predictor) parsePrometheusMetrics(rawMetrics string) (*ModelCoefficients, *BucketCounts, error) {
	lines := strings.Split(rawMetrics, "\n")

	coefficients := &ModelCoefficients{
		TTFTCoeffs: make(map[string]float64),
		TPOTCoeffs: make(map[string]float64),
	}
	bucketCounts := &BucketCounts{
		TTFTBuckets: make(map[int]int),
		TPOTBuckets: make(map[int]int),
	}
	var firstErr error

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		if err := p.parseMetricLine(line, coefficients, bucketCounts); err != nil {
			if firstErr == nil {
				firstErr = err // Save first error to return
			}
			p.logger.V(logutil.TRACE).Info("Skipping unparseable metric line", "line", line, "error", err)
		}
	}
	return coefficients, bucketCounts, firstErr
}

// parseMetricLine parses a single line of Prometheus-formatted text.
func (p *Predictor) parseMetricLine(line string, coefficients *ModelCoefficients, bucketCounts *BucketCounts) error {
	lastSpaceIdx := strings.LastIndexAny(line, " \t")
	if lastSpaceIdx == -1 {
		return errors.New("invalid metric format: no space found")
	}

	metricPart := strings.TrimSpace(line[:lastSpaceIdx])
	valueStr := strings.TrimSpace(line[lastSpaceIdx+1:])

	value, err := strconv.ParseFloat(valueStr, 64)
	if err != nil {
		return fmt.Errorf("could not parse value '%s': %w", valueStr, err)
	}

	metricName := metricPart
	if openBrace := strings.Index(metricPart, "{"); openBrace != -1 {
		metricName = metricPart[:openBrace]
	}

	switch metricName {
	case "ttft_intercept":
		coefficients.TTFTIntercept = value
	case "tpot_intercept":
		coefficients.TPOTIntercept = value
	case "ttft_coef":
		if feature := p.extractLabel(metricPart, "feature"); feature != "" {
			coefficients.TTFTCoeffs[feature] = value
		}
	case "tpot_coef":
		if feature := p.extractLabel(metricPart, "feature"); feature != "" {
			coefficients.TPOTCoeffs[feature] = value
		}
	case "training_samples_count":
		model := p.extractLabel(metricPart, "model")
		bucketStr := p.extractLabel(metricPart, "bucket")
		if bucket, err := strconv.Atoi(bucketStr); err == nil {
			switch model {
			case "ttft":
				bucketCounts.TTFTBuckets[bucket] = int(value)
			case "tpot":
				bucketCounts.TPOTBuckets[bucket] = int(value)
			}
		}
	}
	return nil
}

// extractLabel extracts a label value from a Prometheus metric string.
// Example: `metric{key="value"}`, `key` -> `"value"`
func (p *Predictor) extractLabel(metricPart, labelName string) string {
	searchStr := labelName + `="`
	start := strings.Index(metricPart, searchStr)
	if start == -1 {
		return ""
	}
	start += len(searchStr)
	end := strings.Index(metricPart[start:], `"`)
	if end == -1 {
		return ""
	}
	return metricPart[start : start+end]
}

// GetModelCoefficients fetches the latest metrics and returns the parsed coefficients.
func (p *Predictor) GetModelCoefficients(ctx context.Context) (*ModelCoefficients, error) {
	metrics, err := p.GetMetrics(ctx)
	if err != nil {
		return nil, err
	}
	if metrics.Coefficients == nil {
		return nil, errors.New("coefficients not available in fetched metrics")
	}
	return metrics.Coefficients, nil
}

// GetBucketCounts fetches the latest metrics and returns the parsed bucket counts.
func (p *Predictor) GetBucketCounts(ctx context.Context) (*BucketCounts, error) {
	metrics, err := p.GetMetrics(ctx)
	if err != nil {
		return nil, err
	}
	if metrics.BucketCounts == nil {
		return nil, errors.New("bucket counts not available in fetched metrics")
	}
	return metrics.BucketCounts, nil
}
