/*
Copyright 2026 The Kubernetes Authors.

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

package tests

import (
	"fmt"
	"testing"
	"time"

	gatewayconfig "sigs.k8s.io/gateway-api/conformance/utils/config"
	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
	"sigs.k8s.io/gateway-api/conformance/utils/roundtripper"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/utils/config"
)

const (
	warmUpBudgetFraction = 10
	warmUpRetryInterval  = 500 * time.Millisecond
)

// warmUpBackend is a best-effort alternative to MakeRequestAndExpectEventuallyConsistentResponse.
// It waits for consecutive matching responses and returns an error instead of failing the test.
func warmUpBackend(t *testing.T, rt roundtripper.RoundTripper, tc gatewayconfig.TimeoutConfig, gwAddr string, expected gwhttp.ExpectedResponse) error {
	t.Helper()

	req := gwhttp.MakeRequest(t, &expected, gwAddr, "HTTP", "http")
	budget := warmUpBudget(tc)
	threshold := max(1, tc.RequiredConsecutiveSuccesses)
	deadline := time.Now().Add(budget)
	successes := 0
	var lastErr error

	for time.Now().Before(deadline) {
		cReq, cRes, err := boundedRoundTripper(rt, time.Until(deadline)).CaptureRoundTrip(req)
		if err != nil {
			lastErr = err
		} else {
			lastErr = gwhttp.CompareRoundTrip(t, &req, cReq, cRes, expected)
		}

		if lastErr == nil {
			successes++
			if successes >= threshold {
				return nil
			}
			continue
		}

		successes = 0
		if time.Until(deadline) <= warmUpRetryInterval {
			// Stop if there is no time for another attempt after the backoff.
			break
		}
		time.Sleep(warmUpRetryInterval)
	}

	if lastErr != nil {
		return fmt.Errorf("did not converge within %s: %w", budget, lastErr)
	}
	return fmt.Errorf("reached only %d of %d consecutive successes within %s",
		successes, threshold, budget)
}

// boundedRoundTripper caps the default transport's request timeout without changing
// the suite's transport. Custom transports must enforce their own timeout.
func boundedRoundTripper(rt roundtripper.RoundTripper, remaining time.Duration) roundtripper.RoundTripper {
	d, ok := rt.(*roundtripper.DefaultRoundTripper)
	if !ok || d.TimeoutConfig.RequestTimeout <= remaining {
		return rt
	}
	bounded := *d
	bounded.TimeoutConfig.RequestTimeout = remaining
	return &bounded
}

// warmUpBudget gives each backend a tenth of the consistency timeout: 30s by default.
func warmUpBudget(tc gatewayconfig.TimeoutConfig) time.Duration {
	if tc.MaxTimeToConsistency <= 0 {
		tc = config.DefaultInferenceExtensionTimeoutConfig().TimeoutConfig
	}
	return tc.MaxTimeToConsistency / warmUpBudgetFraction
}
