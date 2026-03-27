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

package predictedlatency

import (
	"testing"
)

func TestNewTokenSampler(t *testing.T) {
	s := newDecodeTokenSampler("req-123", 100, 5)
	if s == nil {
		t.Fatal("sampler should not be nil")
	}
	if s.samplingMean != 100 {
		t.Errorf("samplingMean = %f, want 100", s.samplingMean)
	}
	if s.maxDecodeTokenSamplesForPrediction != 5 {
		t.Errorf("maxSamples = %d, want 5", s.maxDecodeTokenSamplesForPrediction)
	}
	// First sample should be >= 2 (skips token 1 = TTFT)
	if s.nextSampleToken < 2 {
		t.Errorf("nextSampleToken = %d, want >= 2", s.nextSampleToken)
	}
}

func TestTokenSamplerDeterministic(t *testing.T) {
	// Same request ID → same seed → same sequence
	s1 := newDecodeTokenSampler("req-abc", 50, 10)
	s2 := newDecodeTokenSampler("req-abc", 50, 10)

	if s1.nextSampleToken != s2.nextSampleToken {
		t.Errorf("same request ID should produce same first sample: %d vs %d",
			s1.nextSampleToken, s2.nextSampleToken)
	}
}

func TestTokenSamplerShouldPredict(t *testing.T) {
	s := newDecodeTokenSampler("req-1", 10, 3)
	nextToken := s.getNextSampleToken()

	// Before the sample token — should not predict
	if s.shouldPredict(nextToken - 1) {
		t.Error("should not predict before nextSampleToken")
	}

	// At the sample token — should predict
	if !s.shouldPredict(nextToken) {
		t.Error("should predict at nextSampleToken")
	}
}

func TestTokenSamplerMaxSamples(t *testing.T) {
	s := newDecodeTokenSampler("req-2", 1, 2) // mean=1, max=2 samples

	// Record 2 predictions
	token := s.getNextSampleToken()
	s.recordPrediction(token)
	token = s.getNextSampleToken()
	s.recordPrediction(token)

	// After max samples, shouldPredict should return false
	if s.shouldPredict(token + 1) {
		t.Error("should not predict after max samples reached")
	}
}

func TestTokenSamplerZeroMean(t *testing.T) {
	// samplingMean=0 → poissonNext returns 1
	s := newDecodeTokenSampler("req-3", 0, 10)
	if s.nextSampleToken < 2 {
		t.Errorf("nextSampleToken = %d, want >= 2", s.nextSampleToken)
	}
}

func TestTokenSamplerLargeLambda(t *testing.T) {
	// Large lambda uses normal approximation path
	s := newDecodeTokenSampler("req-4", 1000, 5)
	if s.nextSampleToken < 2 {
		t.Errorf("nextSampleToken = %d, want >= 2", s.nextSampleToken)
	}
}
