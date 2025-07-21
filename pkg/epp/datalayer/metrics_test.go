/*
Copyright 2025 The Kubernetes Authors.

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

package datalayer_test

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

func TestMetricsClone(t *testing.T) {
	m := &datalayer.Metrics{
		ActiveModels:            map[string]int{"modelA": 1},
		WaitingModels:           map[string]int{"modelB": 2},
		MaxActiveModels:         5,
		RunningQueueSize:        3,
		WaitingQueueSize:        7,
		KVCacheUsagePercent:     42.5,
		KvCacheMaxTokenCapacity: 2048,
		UpdateTime:              time.Now(),
	}

	clone := m.Clone()

	assert.NotNil(t, clone)
	if diff := cmp.Diff(m, clone); diff != "" {
		t.Errorf("Unexpected output (-want +got): %v", diff)
	}

	// ensure maps are cloned and not sharing pointers
	clone.ActiveModels["modelA"] = 42
	assert.Equal(t, 1, m.ActiveModels["modelA"], "mutating clone should not affect original")
	clone.WaitingModels["modelB"] = 99
	assert.Equal(t, 2, m.WaitingModels["modelB"], "mutating clone should not affect original")
}

func TestMetricsCloneOfNil(t *testing.T) {
	var m *datalayer.Metrics
	assert.Nil(t, m.Clone())
}

func TestMetricsToString(t *testing.T) {
	m := datalayer.NewMetrics()
	assert.NotEmpty(t, m.String())

	var none *datalayer.Metrics
	assert.Equal(t, "", none.String())
}
