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

package mocks

import (
	"context"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// NoopMetricsRecorder is a no-op implementation of plugin.MetricsRecorder for use in tests.
type NoopMetricsRecorder struct{}

var _ plugin.MetricsRecorder = &NoopMetricsRecorder{}

func (n *NoopMetricsRecorder) RecordRequestTTFT(context.Context, string, string, float64) bool {
	return true
}

func (n *NoopMetricsRecorder) RecordRequestPredictedTTFT(context.Context, string, string, float64) bool {
	return true
}

func (n *NoopMetricsRecorder) RecordRequestTTFTWithSLO(context.Context, string, string, float64, float64) bool {
	return true
}

func (n *NoopMetricsRecorder) RecordRequestTTFTPredictionDuration(context.Context, string, string, float64) bool {
	return true
}

func (n *NoopMetricsRecorder) RecordRequestTPOT(context.Context, string, string, float64) bool {
	return true
}

func (n *NoopMetricsRecorder) RecordRequestPredictedTPOT(context.Context, string, string, float64) bool {
	return true
}

func (n *NoopMetricsRecorder) RecordRequestTPOTWithSLO(context.Context, string, string, float64, float64) bool {
	return true
}

func (n *NoopMetricsRecorder) RecordRequestTPOTPredictionDuration(context.Context, string, string, float64) bool {
	return true
}

func (n *NoopMetricsRecorder) RecordPrefixCacheSize(int64) {}

func (n *NoopMetricsRecorder) RecordPrefixCacheMatch(int, int) {}
