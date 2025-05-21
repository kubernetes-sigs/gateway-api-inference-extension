package handlers

import (
	"crypto/rand"
	"testing"
)

func TestBuildCommonResponses(t *testing.T) {
	tests := []struct {
		name                 string
		count                int
		expectedMessageCount int
	}{
		{
			name:                 "zero case",
			count:                0,
			expectedMessageCount: 1,
		},
		{
			name:                 "below limit",
			count:                bodyByteLimit - 1000,
			expectedMessageCount: 1,
		},
		{
			name:                 "at limit",
			count:                bodyByteLimit,
			expectedMessageCount: 1,
		},
		{
			name:                 "off by one error?",
			count:                bodyByteLimit + 1,
			expectedMessageCount: 2,
		},
		{
			name:                 "above limit",
			count:                bodyByteLimit + 1000,
			expectedMessageCount: 2,
		},
		{
			name:                 "above limit",
			count:                (bodyByteLimit * 2) + 1000,
			expectedMessageCount: 3,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			arr := generateBytes(test.count)
			responses := buildCommonResponses(arr, bodyByteLimit, true)
			for i, response := range responses {
				eos := response.BodyMutation.GetStreamedResponse().GetEndOfStream()
				if eos == true && i+1 != len(responses) {
					t.Fatalf("EoS should not be set")
				}
				if eos == false && i+1 == len(responses) {
					t.Fatalf("EoS should be set")
				}
			}
			if len(responses) != test.expectedMessageCount {
				t.Fatalf("Expected: %v, Got %v", test.expectedMessageCount, len(responses))
			}
		})
	}
}

func generateBytes(count int) []byte {
	arr := make([]byte, count)
	_, _ = rand.Read(arr)
	return arr
}
