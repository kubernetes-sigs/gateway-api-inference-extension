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

package kubernetes

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
	"sigs.k8s.io/gateway-api/conformance/utils/roundtripper"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

func observation(id, endpoint string) string {
	return fmt.Sprintf(`{"msg":%q,"conformanceRequestID":%q,"servedEndpoint":%q}`, metadata.ServedEndpointObservation, id, endpoint) + "\n"
}

func TestServedEndpointObserved(t *testing.T) {
	for _, tc := range []struct {
		name, logs     string
		found, failure bool
	}{
		{"matching", observation("current", "10.0.0.1:3000"), true, false},
		{"unrelated request", observation("old", "10.0.0.1:3000"), false, false},
		{"wrong endpoint", observation("current", "10.0.0.2:3000"), false, true},
		{"missing metadata", observation("current", "fail: missing envoy lb metadata"), false, true},
		{"missing key", observation("current", "fail: missing destination endpoint served metadata"), false, true},
		{"echo header is not evidence", `{"msg":"Setting conformance test result header","conformanceRequestID":"current","servedEndpoint":"10.0.0.1:3000"}`, false, false},
		{"malformed unrelated line", "not-json\n" + observation("current", "10.0.0.1:3000"), true, false},
		{"conflicting retries", observation("current", "10.0.0.1:3000") + observation("current", "10.0.0.2:3000"), false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			found, err := servedEndpointObserved(tc.logs, "current", "10.0.0.1:3000")
			require.Equal(t, tc.found, found)
			require.Equal(t, tc.failure, err != nil)
		})
	}
}

func TestWaitForServedEndpoint(t *testing.T) {
	for _, mode := range []string{"delayed", "absent", "wrong", "forbidden"} {
		t.Run(mode, func(t *testing.T) {
			var calls atomic.Int32
			since := time.Now().Add(-time.Second)
			api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch r.URL.Path {
				case "/api/v1/namespaces/test/pods":
					require.Equal(t, "app=epp", r.URL.Query().Get("labelSelector"))
					fmt.Fprint(w, `{"apiVersion":"v1","kind":"PodList","items":[{"metadata":{"name":"epp-1"}}]}`)
				case "/api/v1/namespaces/test/pods/epp-1/log":
					require.Equal(t, "epp", r.URL.Query().Get("container"))
					require.NotEmpty(t, r.URL.Query().Get("sinceTime"))
					n := calls.Add(1)
					if mode == "forbidden" {
						w.WriteHeader(http.StatusForbidden)
						fmt.Fprint(w, `{"kind":"Status","apiVersion":"v1","status":"Failure","reason":"Forbidden","code":403}`)
						return
					}
					fmt.Fprint(w, observation("old", "10.0.0.1:3000"))
					if mode == "delayed" && n >= 3 {
						fmt.Fprint(w, observation("current", "10.0.0.1:3000"))
					}
					if mode == "wrong" {
						fmt.Fprint(w, observation("current", "10.0.0.2:3000"))
					}
				default:
					http.NotFound(w, r)
				}
			}))
			defer api.Close()
			client, err := kubernetes.NewForConfig(&rest.Config{Host: api.URL})
			require.NoError(t, err)
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			if mode == "absent" {
				cancel()
				ctx, cancel = context.WithTimeout(context.Background(), 60*time.Millisecond)
			}
			defer cancel()
			err = WaitForServedEndpoint(ctx, client, "test", "app=epp", "current", "10.0.0.1:3000", since, 5*time.Millisecond)
			if mode == "delayed" {
				require.NoError(t, err)
				require.GreaterOrEqual(t, calls.Load(), int32(3))
			} else {
				require.Error(t, err)
			}
		})
	}
}

// Reproduce the original header dependency with a real local HTTP response and
// the same comparison function used by the conformance suite. This is a local
// fixture, not a gateway implementation or a full conformance run.
func TestClientResponseDoesNotRequireServedEndpointHeader(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	response, err := http.Get(server.URL)
	require.NoError(t, err)
	defer response.Body.Close()
	captured := &roundtripper.CapturedResponse{StatusCode: response.StatusCode, Headers: response.Header}
	request := &roundtripper.Request{}
	backend := &roundtripper.CapturedRequest{Method: "GET"}
	expected := gwhttp.ExpectedResponse{Response: gwhttp.Response{StatusCodes: []int{http.StatusOK}}}
	require.NoError(t, gwhttp.CompareRoundTrip(t, request, backend, captured, expected))
	expected.Response.Headers = map[string]string{metadata.ConformanceTestResultHeader: "10.0.0.1:3000"}
	require.ErrorContains(t, gwhttp.CompareRoundTrip(t, request, backend, captured, expected), metadata.ConformanceTestResultHeader)
}
