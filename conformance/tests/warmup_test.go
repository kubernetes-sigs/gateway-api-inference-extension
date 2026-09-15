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
	"context"
	"errors"
	"net"
	"net/http"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/require"
	gatewayconfig "sigs.k8s.io/gateway-api/conformance/utils/config"
	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
	"sigs.k8s.io/gateway-api/conformance/utils/roundtripper"
)

type warmUpRoundTripper func(roundtripper.Request) (*roundtripper.CapturedRequest, *roundtripper.CapturedResponse, error)

func (f warmUpRoundTripper) CaptureRoundTrip(req roundtripper.Request) (*roundtripper.CapturedRequest, *roundtripper.CapturedResponse, error) {
	return f(req)
}

func warmUpProbe() gwhttp.ExpectedResponse {
	return gwhttp.ExpectedResponse{
		Request:   gwhttp.Request{Host: "example.com", Path: "/", Method: http.MethodPost},
		Response:  gwhttp.Response{StatusCode: http.StatusOK},
		Backend:   "backend-0",
		Namespace: "test",
	}
}

func warmUpResponse(req roundtripper.Request, pod string) (*roundtripper.CapturedRequest, *roundtripper.CapturedResponse, error) {
	return &roundtripper.CapturedRequest{
		Path: req.URL.Path, Host: req.Host, Method: req.Method,
		Namespace: "test", Pod: pod,
	}, &roundtripper.CapturedResponse{StatusCode: http.StatusOK}, nil
}

func TestWarmUpBudget(t *testing.T) {
	for _, tc := range []struct {
		name        string
		consistency time.Duration
		want        time.Duration
	}{
		{"default", 300 * time.Second, 30 * time.Second},
		{"overridden", 30 * time.Second, 3 * time.Second},
		{"unset", 0, 30 * time.Second},
		{"negative", -time.Second, 30 * time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, warmUpBudget(gatewayconfig.TimeoutConfig{MaxTimeToConsistency: tc.consistency}))
		})
	}
}

func TestWarmUpBackendRequiresConsecutiveSuccesses(t *testing.T) {
	for _, failure := range []string{"none", "transport error", "wrong backend"} {
		t.Run(failure, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				attempts := 0
				rt := warmUpRoundTripper(func(req roundtripper.Request) (*roundtripper.CapturedRequest, *roundtripper.CapturedResponse, error) {
					attempts++
					if attempts == 3 {
						switch failure {
						case "transport error":
							return nil, nil, errors.New("backend down")
						case "wrong backend":
							return warmUpResponse(req, "other-backend")
						}
					}
					return warmUpResponse(req, "backend-0")
				})
				cfg := gatewayconfig.TimeoutConfig{RequiredConsecutiveSuccesses: 3}

				start := time.Now()
				err := warmUpBackend(t, rt, cfg, "127.0.0.1:80", warmUpProbe())

				require.NoError(t, err)
				if failure == "none" {
					require.Equal(t, 3, attempts)
					require.Zero(t, time.Since(start), "successful attempts need no backoff")
				} else {
					require.Equal(t, 6, attempts, "failure must reset the consecutive-success count")
					require.Equal(t, 500*time.Millisecond, time.Since(start))
				}
			})
		})
	}
}

func TestWarmUpBackendStopsRetryingAtBudget(t *testing.T) {
	backendErr := errors.New("backend down")
	for _, tc := range []struct {
		name         string
		budget       time.Duration
		delay        time.Duration
		err          error
		wantAttempts int
		wantElapsed  time.Duration
	}{
		{"too few successes", 300 * time.Millisecond, 200 * time.Millisecond, nil, 2, 400 * time.Millisecond},
		{"persistent failure", 1200 * time.Millisecond, 0, backendErr, 3, time.Second},
		{"no time to retry", 300 * time.Millisecond, 200 * time.Millisecond, backendErr, 1, 200 * time.Millisecond},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				attempts := 0
				rt := warmUpRoundTripper(func(req roundtripper.Request) (*roundtripper.CapturedRequest, *roundtripper.CapturedResponse, error) {
					attempts++
					time.Sleep(tc.delay)
					if tc.err != nil {
						return nil, nil, tc.err
					}
					return warmUpResponse(req, "backend-0")
				})
				cfg := gatewayconfig.TimeoutConfig{
					MaxTimeToConsistency: tc.budget * 10, RequiredConsecutiveSuccesses: 3,
				}

				start := time.Now()
				err := warmUpBackend(t, rt, cfg, "127.0.0.1:80", warmUpProbe())

				require.Error(t, err)
				if tc.err != nil {
					require.ErrorIs(t, err, tc.err)
				} else {
					require.ErrorContains(t, err, "2 of 3 consecutive successes")
				}
				require.Equal(t, tc.wantAttempts, attempts)
				// Custom transports cannot be interrupted; a started request may finish after the budget.
				require.Equal(t, tc.wantElapsed, time.Since(start))
			})
		})
	}
}

func TestWarmUpBackendBoundsStalledRequest(t *testing.T) {
	for _, requestTimeout := range []time.Duration{2 * time.Second, 100 * time.Millisecond} {
		t.Run(requestTimeout.String(), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				// Stall the HTTP exchange using an in-memory connection and virtual time.
				client, server := net.Pipe()
				defer client.Close()
				defer server.Close()
				cfg := gatewayconfig.TimeoutConfig{
					MaxTimeToConsistency: 5 * time.Second, RequestTimeout: requestTimeout,
				}
				rt := &roundtripper.DefaultRoundTripper{
					TimeoutConfig: cfg,
					CustomDialContext: func(context.Context, string, string) (net.Conn, error) {
						return client, nil
					},
				}

				start := time.Now()
				err := warmUpBackend(t, rt, cfg, "127.0.0.1:80", warmUpProbe())

				require.ErrorIs(t, err, context.DeadlineExceeded)
				require.Equal(t, min(requestTimeout, 500*time.Millisecond), time.Since(start))
				require.Equal(t, cfg, rt.TimeoutConfig, "warm-up must preserve the suite's transport settings")
			})
		})
	}
}
