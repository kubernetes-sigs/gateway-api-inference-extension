package metrics

import (
	"os"
	"testing"
	"time"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

const RequestTotalMetric = LLMServiceModelComponent + "_request_total"

func TestMonitorRequest(t *testing.T) {
	type requests struct {
		llmserviceName  string
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
				llmserviceName:  "s10",
				modelName:       "m10",
				targetModelName: "t10",
				reqSize:         10,
				elapsed:         time.Millisecond * 10,
			},
			{
				llmserviceName:  "s10",
				modelName:       "m10",
				targetModelName: "t10",
				reqSize:         20,
				elapsed:         time.Millisecond * 20,
			},
			{
				llmserviceName:  "s10",
				modelName:       "m10",
				targetModelName: "t11",
				reqSize:         30,
				elapsed:         time.Millisecond * 30,
			},
			{
				llmserviceName:  "s20",
				modelName:       "m20",
				targetModelName: "t20",
				reqSize:         40,
				elapsed:         time.Millisecond * 40,
			},
		},
	}}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				MonitorRequest(req.llmserviceName, req.modelName, req.targetModelName, req.reqSize, req.elapsed)
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
		})
	}
}
