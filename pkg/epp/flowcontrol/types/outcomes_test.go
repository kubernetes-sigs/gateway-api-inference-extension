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

package types

import (
	"testing"
)

func TestQueueOutcomeString(t *testing.T) {
	tests := []struct {
		name     string
		outcome  QueueOutcome
		expected string
	}{
		{
			name:     "NotYetFinalized",
			outcome:  QueueOutcomeNotYetFinalized,
			expected: "NotYetFinalized",
		},
		{
			name:     "Dispatched",
			outcome:  QueueOutcomeDispatched,
			expected: "Dispatched",
		},
		{
			name:     "RejectedCapacity",
			outcome:  QueueOutcomeRejectedCapacity,
			expected: "RejectedCapacity",
		},
		{
			name:     "RejectedOther",
			outcome:  QueueOutcomeRejectedOther,
			expected: "RejectedOther",
		},
		{
			name:     "EvictedTTL",
			outcome:  QueueOutcomeEvictedTTL,
			expected: "EvictedTTL",
		},
		{
			name:     "EvictedContextCancelled",
			outcome:  QueueOutcomeEvictedContextCancelled,
			expected: "EvictedContextCancelled",
		},
		{
			name:     "EvictedOther",
			outcome:  QueueOutcomeEvictedOther,
			expected: "EvictedOther",
		},
		// default branch: unknown positive value
		{
			name:     "unknown positive value",
			outcome:  QueueOutcome(99),
			expected: "UnknownOutcome(99)",
		},
		// default branch: unknown negative value
		{
			name:     "unknown negative value",
			outcome:  QueueOutcome(-1),
			expected: "UnknownOutcome(-1)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.outcome.String()
			if result != tt.expected {
				t.Errorf("QueueOutcome(%d).String() = %q, want %q", int(tt.outcome), result, tt.expected)
			}
		})
	}
}
