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

// tokenSampler handles Poisson-distributed sampling for predictions only
// Training happens on every token regardless of sampling
type tokenSampler struct {
	rng             *rand.Rand
	nextSampleToken int
	samplingMean    float64
	maxSamples      int
	sampleCount     int
}

func newTokenSampler(requestID string, samplingMean float64, maxSamples int) *tokenSampler {
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

	sampler := &tokenSampler{
		rng:          rand.New(rand.NewSource(seed)),
		samplingMean: samplingMean,
		maxSamples:   maxSamples,
	}

	// Set first sample token (skip token 1 since that's TTFT)
	sampler.nextSampleToken = 2 + sampler.poissonNext()

	return sampler
}

// poissonNext generates the next interval using Poisson distribution
func (ts *tokenSampler) poissonNext() int {
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

// shouldPredict determines if we should make a prediction for the current token
func (ts *tokenSampler) shouldPredict(currentToken int) bool {
	return currentToken == ts.nextSampleToken && ts.sampleCount < ts.maxSamples
}

// recordPrediction records that a prediction was made and calculates the next sample token
func (ts *tokenSampler) recordPrediction(currentToken int) {
	if ts.sampleCount >= ts.maxSamples {
		return
	}

	ts.sampleCount++

	if ts.sampleCount < ts.maxSamples {
		interval := ts.poissonNext()
		ts.nextSampleToken = currentToken + interval
	}
}

// getNextSampleToken returns the next token to predict for
func (ts *tokenSampler) getNextSampleToken() int {
	return ts.nextSampleToken
}
