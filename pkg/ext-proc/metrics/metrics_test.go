package metrics

import (
	"os"
	"testing"
	"time"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

const RequestTotalMetric = InferenceModelComponent + "_request_total"
const RequestLatenciesMetric = InferenceModelComponent + "_request_duration_seconds"
const RequestSizesMetric = InferenceModelComponent + "_request_sizes"

func TestMonitorRequest(t *testing.T) {
	type requests struct {
		modelName       string
		targetModelName string
		reqSize         int
		elapsed         time.Duration
	}
	scenarios := []struct {
		name string
		reqs []requests
	}{{
		name: "multiple requests",
		reqs: []requests{
			{
				modelName:       "m10",
				targetModelName: "t10",
				reqSize:         1200,
				elapsed:         time.Millisecond * 10,
			},
			{
				modelName:       "m10",
				targetModelName: "t10",
				reqSize:         500,
				elapsed:         time.Millisecond * 1600,
			},
			{
				modelName:       "m10",
				targetModelName: "t11",
				reqSize:         2480,
				elapsed:         time.Millisecond * 60,
			},
			{
				modelName:       "m20",
				targetModelName: "t20",
				reqSize:         80,
				elapsed:         time.Millisecond * 120,
			},
		},
	}}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				MonitorRequest(req.modelName, req.targetModelName, req.reqSize, req.elapsed)
			}
			wantRequestTotal, err := os.Open("testdata/request_total_metric")
			defer func() {
				if err := wantRequestTotal.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantRequestTotal, RequestTotalMetric); err != nil {
				t.Error(err)
			}
			wantRequestLatencies, err := os.Open("testdata/request_duration_seconds_metric")
			defer func() {
				if err := wantRequestLatencies.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantRequestLatencies, RequestLatenciesMetric); err != nil {
				t.Error(err)
			}
			wantRequestSizes, err := os.Open("testdata/request_sizes_metric")
			defer func() {
				if err := wantRequestSizes.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantRequestSizes, RequestSizesMetric); err != nil {
				t.Error(err)
			}

		})
	}
}
