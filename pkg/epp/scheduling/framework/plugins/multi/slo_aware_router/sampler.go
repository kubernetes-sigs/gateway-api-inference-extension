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

package slo_aware_router

import (
	"hash/fnv"
	"math"
	"math/rand"
	"time"
)

// TokenSampler handles Poisson-distributed sampling for predictions only
// Training happens on every token regardless of sampling
type TokenSampler struct {
	rng             *rand.Rand
	nextSampleToken int
	samplingMean    float64
	maxSamples      int
	sampleCount     int
}

// SetSamplingMean sets the sampling mean (lambda) for the Poisson distribution
func (ts *TokenSampler) SetSamplingMean(mean float64) {
	ts.samplingMean = mean
}

// SetMaxSamples sets the maximum number of samples
func (ts *TokenSampler) SetMaxSamples(max int) {
	ts.maxSamples = max
}

// SetSampleCount sets the current number of predictions made
func (ts *TokenSampler) SetSampleCount(count int) {
	ts.sampleCount = count
}

func NewTokenSampler(requestID string, samplingMean float64, maxSamples int) *TokenSampler {
	// Use request ID hash as seed for reproducibility
	seed := int64(0)
	if requestID != "" {
		hash := fnv.New64a()
		hash.Write([]byte(requestID))
		seed = int64(hash.Sum64())
	}
	if seed == 0 {
		seed = time.Now().UnixNano()
	}

	sampler := &TokenSampler{
		rng:          rand.New(rand.NewSource(seed)),
		samplingMean: samplingMean,
		maxSamples:   maxSamples,
	}

	// Set first sample token (skip token 1 since that's TTFT)
	sampler.nextSampleToken = 2 + sampler.poissonNext()

	return sampler
}

// poissonNext generates the next interval using Poisson distribution
func (ts *TokenSampler) poissonNext() int {
	lambda := ts.samplingMean
	if lambda <= 0 {
		return 1
	}

	// For small lambda, use Knuth's algorithm
	if lambda < 30 {
		l := math.Exp(-lambda)
		k := 0
		p := 1.0

		for p > l {
			k++
			p *= ts.rng.Float64()
		}
		return k - 1
	}

	// For larger lambda, use normal approximation
	normal := ts.rng.NormFloat64()
	interval := int(math.Round(lambda + math.Sqrt(lambda)*normal))
	if interval < 1 {
		return 1
	}
	return interval
}

// ShouldPredict determines if we should make a prediction for the current token
func (ts *TokenSampler) ShouldPredict(currentToken int) bool {
	return currentToken == ts.nextSampleToken && ts.sampleCount < ts.maxSamples
}

// RecordPrediction records that a prediction was made and calculates the next sample token
func (ts *TokenSampler) RecordPrediction(currentToken int) {
	if ts.sampleCount >= ts.maxSamples {
		return
	}

	ts.sampleCount++

	if ts.sampleCount < ts.maxSamples {
		interval := ts.poissonNext()
		ts.nextSampleToken = currentToken + interval
	}
}

// GetNextSampleToken returns the next token to predict for
func (ts *TokenSampler) GetNextSampleToken() int {
	return ts.nextSampleToken
}

// SetNextSampleToken sets the next token to predict for
func (ts *TokenSampler) SetNextSampleToken(token int) {
	ts.nextSampleToken = token
}

// GetSampleCount returns the current number of predictions made
func (ts *TokenSampler) GetSampleCount() int {
	return ts.sampleCount
}
