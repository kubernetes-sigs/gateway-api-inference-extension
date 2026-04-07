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
	"testing"

	"github.com/stretchr/testify/assert"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

func TestRecorderImplementsInterface(t *testing.T) {
	var _ plugin.MetricsRecorder = &Recorder{}
}

func TestRecorderDelegates(t *testing.T) {
	Register()
	defer Reset()

	r := NewRecorder()
	ctx := t.Context()

	assert.True(t, r.RecordRequestTTFT(ctx, "model", "target", 0.5))
	assert.True(t, r.RecordRequestPredictedTTFT(ctx, "model", "target", 0.5))
	assert.True(t, r.RecordRequestTTFTWithSLO(ctx, "model", "target", 0.5, 1.0))
	assert.True(t, r.RecordRequestTTFTPredictionDuration(ctx, "model", "target", 0.1))
	assert.True(t, r.RecordRequestTPOT(ctx, "model", "target", 0.02))
	assert.True(t, r.RecordRequestPredictedTPOT(ctx, "model", "target", 0.02))
	assert.True(t, r.RecordRequestTPOTWithSLO(ctx, "model", "target", 0.02, 0.05))
	assert.True(t, r.RecordRequestTPOTPredictionDuration(ctx, "model", "target", 0.1))

	// These should not panic
	r.RecordPrefixCacheSize(100)
	r.RecordPrefixCacheMatch(10, 20)
}

func TestRecorderRejectsNegativeValues(t *testing.T) {
	Register()
	defer Reset()

	r := NewRecorder()
	ctx := t.Context()

	assert.False(t, r.RecordRequestTTFT(ctx, "model", "target", -1.0))
	assert.False(t, r.RecordRequestTPOT(ctx, "model", "target", -1.0))
}
