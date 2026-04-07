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

package metrics

import (
	"context"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// Recorder implements plugin.MetricsRecorder by delegating to the package-level
// recording functions.
type Recorder struct{}

var _ plugin.MetricsRecorder = &Recorder{}

func NewRecorder() *Recorder {
	return &Recorder{}
}

func (r *Recorder) RecordRequestTTFT(ctx context.Context, modelName, targetModelName string, ttft float64) bool {
	return RecordRequestTTFT(ctx, modelName, targetModelName, ttft)
}

func (r *Recorder) RecordRequestPredictedTTFT(ctx context.Context, modelName, targetModelName string, predictedTTFT float64) bool {
	return RecordRequestPredictedTTFT(ctx, modelName, targetModelName, predictedTTFT)
}

func (r *Recorder) RecordRequestTTFTWithSLO(ctx context.Context, modelName, targetModelName string, ttft float64, sloThreshold float64) bool {
	return RecordRequestTTFTWithSLO(ctx, modelName, targetModelName, ttft, sloThreshold)
}

func (r *Recorder) RecordRequestTTFTPredictionDuration(ctx context.Context, modelName, targetModelName string, duration float64) bool {
	return RecordRequestTTFTPredictionDuration(ctx, modelName, targetModelName, duration)
}

func (r *Recorder) RecordRequestTPOT(ctx context.Context, modelName, targetModelName string, tpot float64) bool {
	return RecordRequestTPOT(ctx, modelName, targetModelName, tpot)
}

func (r *Recorder) RecordRequestPredictedTPOT(ctx context.Context, modelName, targetModelName string, predictedTPOT float64) bool {
	return RecordRequestPredictedTPOT(ctx, modelName, targetModelName, predictedTPOT)
}

func (r *Recorder) RecordRequestTPOTWithSLO(ctx context.Context, modelName, targetModelName string, tpot float64, sloThreshold float64) bool {
	return RecordRequestTPOTWithSLO(ctx, modelName, targetModelName, tpot, sloThreshold)
}

func (r *Recorder) RecordRequestTPOTPredictionDuration(ctx context.Context, modelName, targetModelName string, duration float64) bool {
	return RecordRequestTPOTPredictionDuration(ctx, modelName, targetModelName, duration)
}

func (r *Recorder) RecordPrefixCacheSize(size int64) {
	RecordPrefixCacheSize(size)
}

func (r *Recorder) RecordPrefixCacheMatch(matchedLength, totalLength int) {
	RecordPrefixCacheMatch(matchedLength, totalLength)
}
