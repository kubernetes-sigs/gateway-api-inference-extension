package concurrencydetector

import (
	"testing"
	"fmt"

	"github.com/stretchr/testify/require"

	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestSimpleTokenEstimator_Estimate(t *testing.T) {
	estimator := NewSimpleTokenEstimator()

	testCases := []struct {
		name     string
		request  *framework.LLMRequest
		expected int64
	}{
		{
			name:     "Empty request",
			request:  &framework.LLMRequest{},
			expected: 0,
		},
		{
			name: "Less than 4 characters",
			request: &framework.LLMRequest{
				Body: &framework.LLMRequestBody{
					Completions: &framework.CompletionsRequest{
						Prompt: "123",
					},
				},
			},
			expected: 3,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := estimator.Estimate(tc.request)
			fmt.Println(tc.name, "actual", actual, "expected", tc.expected)
			require.Equal(t, tc.expected, actual)
		})
	}

}
