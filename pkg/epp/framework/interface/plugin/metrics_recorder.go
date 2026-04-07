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

package plugin

import "context"

// MetricsRecorder provides an interface for plugins to record metrics without
// depending on the concrete metrics package.
type MetricsRecorder interface {
	RecordRequestTTFT(ctx context.Context, modelName, targetModelName string, ttft float64) bool
	RecordRequestPredictedTTFT(ctx context.Context, modelName, targetModelName string, predictedTTFT float64) bool
	RecordRequestTTFTWithSLO(ctx context.Context, modelName, targetModelName string, ttft float64, sloThreshold float64) bool
	RecordRequestTTFTPredictionDuration(ctx context.Context, modelName, targetModelName string, duration float64) bool
	RecordRequestTPOT(ctx context.Context, modelName, targetModelName string, tpot float64) bool
	RecordRequestPredictedTPOT(ctx context.Context, modelName, targetModelName string, predictedTPOT float64) bool
	RecordRequestTPOTWithSLO(ctx context.Context, modelName, targetModelName string, tpot float64, sloThreshold float64) bool
	RecordRequestTPOTPredictionDuration(ctx context.Context, modelName, targetModelName string, duration float64) bool
	RecordPrefixCacheSize(size int64)
	RecordPrefixCacheMatch(matchedLength, totalLength int)
}
