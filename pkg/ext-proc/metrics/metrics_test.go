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
const ResponseSizesMetric = InferenceModelComponent + "_response_sizes"
const InputTokensMetric = InferenceModelComponent + "_input_tokens"
const OutputTokensMetric = InferenceModelComponent + "_output_tokens"

func TestRecordRequestCounterandSizes(t *testing.T) {
	type requests struct {
		modelName       string
		targetModelName string
		reqSize         int
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
			},
			{
				modelName:       "m10",
				targetModelName: "t10",
				reqSize:         500,
			},
			{
				modelName:       "m10",
				targetModelName: "t11",
				reqSize:         2480,
			},
			{
				modelName:       "m20",
				targetModelName: "t20",
				reqSize:         80,
			},
		},
	}}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				RecordRequestCounter(req.modelName, req.targetModelName)
				RecordRequestSizes(req.modelName, req.targetModelName, req.reqSize)
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

func TestRecordRequestLatencies(t *testing.T) {
	timeBaseline := time.Now()
	type requests struct {
		modelName       string
		targetModelName string
		receivedTime    time.Time
		completeTime    time.Time
	}
	scenarios := []struct {
		name    string
		reqs    []requests
		invalid bool
	}{{
		name: "multiple requests",
		reqs: []requests{
			{
				modelName:       "m10",
				targetModelName: "t10",
				receivedTime:    timeBaseline,
				completeTime:    timeBaseline.Add(time.Millisecond * 10),
			},
			{
				modelName:       "m10",
				targetModelName: "t10",
				receivedTime:    timeBaseline,
				completeTime:    timeBaseline.Add(time.Millisecond * 1600),
			},
			{
				modelName:       "m10",
				targetModelName: "t11",
				receivedTime:    timeBaseline,
				completeTime:    timeBaseline.Add(time.Millisecond * 60),
			},
			{
				modelName:       "m20",
				targetModelName: "t20",
				receivedTime:    timeBaseline,
				completeTime:    timeBaseline.Add(time.Millisecond * 120),
			},
		},
	},
		{
			name: "invalid elapsed time",
			reqs: []requests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline.Add(time.Millisecond * 10),
					completeTime:    timeBaseline,
				}},
			invalid: true,
		}}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				success := RecordRequestLatencies(req.modelName, req.targetModelName, req.receivedTime, req.completeTime)
				if success == scenario.invalid {
					t.Errorf("got record success(%v), but the request expects invalid(%v)", success, scenario.invalid)
				}
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
		})
	}
}

func TestRecordResponseMetrics(t *testing.T) {
	type responses struct {
		modelName       string
		targetModelName string
		inputToken      int
		outputToken     int
		respSize        int
	}
	scenarios := []struct {
		name string
		resp []responses
	}{{
		name: "multiple requests",
		resp: []responses{
			{
				modelName:       "m10",
				targetModelName: "t10",
				respSize:        1200,
				inputToken:      10,
				outputToken:     100,
			},
			{
				modelName:       "m10",
				targetModelName: "t10",
				respSize:        500,
				inputToken:      20,
				outputToken:     200,
			},
			{
				modelName:       "m10",
				targetModelName: "t11",
				respSize:        2480,
				inputToken:      30,
				outputToken:     300,
			},
			{
				modelName:       "m20",
				targetModelName: "t20",
				respSize:        80,
				inputToken:      40,
				outputToken:     400,
			},
		},
	}}
	Register()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, resp := range scenario.resp {
				RecordInputTokens(resp.modelName, resp.targetModelName, resp.inputToken)
				RecordOutputTokens(resp.modelName, resp.targetModelName, resp.outputToken)
				RecordResponseSizes(resp.modelName, resp.targetModelName, resp.respSize)
			}
			wantResponseSize, err := os.Open("testdata/response_sizes_metric")
			defer func() {
				if err := wantResponseSize.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantResponseSize, ResponseSizesMetric); err != nil {
				t.Error(err)
			}

			wantInputToken, err := os.Open("testdata/input_tokens_metric")
			defer func() {
				if err := wantInputToken.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantInputToken, InputTokensMetric); err != nil {
				t.Error(err)
			}

			wantOutputToken, err := os.Open("testdata/output_tokens_metric")
			defer func() {
				if err := wantOutputToken.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, wantOutputToken, OutputTokensMetric); err != nil {
				t.Error(err)
			}
		})
	}
}
