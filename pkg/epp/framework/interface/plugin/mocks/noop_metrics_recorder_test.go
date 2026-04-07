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

package mocks_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin/mocks"
)

func TestNoopMetricsRecorder(t *testing.T) {
	r := &mocks.NoopMetricsRecorder{}
	ctx := t.Context()

	assert.True(t, r.RecordRequestTTFT(ctx, "m", "t", 1.0))
	assert.True(t, r.RecordRequestPredictedTTFT(ctx, "m", "t", 1.0))
	assert.True(t, r.RecordRequestTTFTWithSLO(ctx, "m", "t", 1.0, 2.0))
	assert.True(t, r.RecordRequestTTFTPredictionDuration(ctx, "m", "t", 1.0))
	assert.True(t, r.RecordRequestTPOT(ctx, "m", "t", 1.0))
	assert.True(t, r.RecordRequestPredictedTPOT(ctx, "m", "t", 1.0))
	assert.True(t, r.RecordRequestTPOTWithSLO(ctx, "m", "t", 1.0, 2.0))
	assert.True(t, r.RecordRequestTPOTPredictionDuration(ctx, "m", "t", 1.0))

	r.RecordPrefixCacheSize(100)
	r.RecordPrefixCacheMatch(10, 20)
}
